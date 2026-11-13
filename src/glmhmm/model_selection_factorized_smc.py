
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.nuts as nuts

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import compute_log_marginal_likelihood, gauss_normalization_default
from .gaussian_evidence import (gaussian_factor_log_evidence,
                                gaussian_sample_weights,
                                resolve_iw_prior)
from .emissions import (
    bernoulli_glmhmm_loglikelihood,
    multinomial_glmhmm_loglikelihood,
    negbin_glmhmm_loglikelihood,
    gamma_glmhmm_loglikelihood,
    gaussian_glmhmm_loglikelihoods,
    normalized_gauss_loglikelihoods
)

def fast_smc_evidence(smc_algorithm, initial_particles, key):
    """
    Runs the SMC loop using jax.lax.while_loop for JIT compilation.
    """
    init_state = smc_algorithm.init(initial_particles)

    def cond_fn(carry):
        state, _, _ = carry
        return state.tempering_param < 1.0

    def body_fn(carry):
        state, k, current_evidence = carry
        k, subk = jax.random.split(k)
        new_state, info = smc_algorithm.step(subk, state)
        new_evidence = current_evidence + info.log_likelihood_increment
        return new_state, k, new_evidence

    final_state, _, final_evidence = jax.lax.while_loop(cond_fn, body_fn, (init_state, key, 0.0))
    return final_evidence

def diagonal_gaussian_prior_sampler(scales):
    """
    Builds an exact sampler for a zero-mean diagonal Gaussian prior.

    Every prior in this module is a product of independent zero-mean Gaussians,
    so the prior can be sampled directly with no MCMC.

    Args:
        scales: Per-coordinate standard deviations, (dim,).

    Returns:
        Callable (key, n) -> (n, dim) samples.
    """
    def sampler(key, n):
        return jax.random.normal(key, (n, scales.shape[0])) * scales
    return sampler


def run_smc_factor(
    logprior_fn,
    loglikelihood_fn,
    prior_sampler,
    dim,
    num_particles=500,
    num_mcmc_steps=10,
    key_seed=0,
    mcmc_kernel='nuts'  # 'nuts' (accurate) or 'hmc' (faster)
):
    """
    Runs SMC to estimate the normalizing constant (evidence) for a single factor.
    Z = \\int exp(loglikelihood_fn(theta)) * exp(logprior_fn(theta)) dtheta

    The adaptive tempered SMC evidence estimator telescopes
    Z = prod_m E_{m-1}[lik^(lambda_m - lambda_{m-1})], which is only valid when
    the lambda = 0 population is distributed according to the prior. Particles
    are therefore drawn from the prior, not placed at the MAP.

    Args:
        logprior_fn: Log prior function
        loglikelihood_fn: Log likelihood function
        prior_sampler: Callable (key, n) -> (n, dim) drawing exact prior samples
        dim: Dimension of parameter space
        num_particles: Number of SMC particles (default 500, use 50-100 for speed)
        num_mcmc_steps: MCMC steps per temperature (default 10, use 3-5 for speed)
        key_seed: Random seed
        mcmc_kernel: 'hmc' for speed, 'nuts' for accuracy
    """
    key = jax.random.PRNGKey(key_seed)
    key, init_key = jax.random.split(key)

    initial_particles = prior_sampler(init_key, num_particles)

    # Use step size based on dimension (heuristic from Stan)
    step_size = 0.5 / jnp.sqrt(dim)
    
    mcmc_params = {
        'step_size': jnp.array([step_size]), 
        'inverse_mass_matrix': jnp.expand_dims(jnp.eye(dim), 0)
    }
    
    if mcmc_kernel == 'hmc':
        # HMC with fixed trajectory length (faster than NUTS)
        import blackjax.mcmc.hmc as hmc
        mcmc_kernel_fn = hmc.build_kernel()
        mcmc_init_fn = hmc.init
        # Add num_integration_steps for HMC (must be array for blackjax)
        mcmc_params['num_integration_steps'] = jnp.array([max(3, min(10, dim // 2))])
    else:
        # NUTS with adaptive trajectory (more accurate but slower)  
        mcmc_kernel_fn = nuts.build_kernel()
        mcmc_init_fn = nuts.init

    smc_loop = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=mcmc_kernel_fn,
        mcmc_init_fn=mcmc_init_fn,
        mcmc_parameters=mcmc_params,
        resampling_fn=resampling.systematic,
        num_mcmc_steps=num_mcmc_steps, 
        target_ess=0.5
    )
    
    log_evidence = fast_smc_evidence(smc_loop, initial_particles, key)
    return float(log_evidence)

def compute_factorized_evidence_smc(
    data: GLMHMMData,
    params: GLMHMMParams,
    config: GLMHMMConfig,
    gammas: jnp.ndarray, 
    xis: jnp.ndarray,
    num_particles: int = 150,
    num_mcmc_steps: int = 10,
    seed: int = 0,
    mcmc_kernel: str = 'nuts',  # 'nuts' (default, accurate) or 'hmc' (faster)
    static_trans_alpha: float = 1.0,  # Dirichlet prior for static transitions
    include_label_switching: bool = True,
    verbose: bool = True
):
    """
    Computes evidence using Factorized SMC.

    Same decomposition as compute_factorized_evidence (BQ); see
    docs/model_and_model_selection.qmd for the derivation, assumptions and
    approximations. Particles for each factor are drawn from that factor's prior,
    which is what makes the tempered-SMC normalising constant valid.

    The Gaussian factor is handled analytically (matrix-normal / inverse-Wishart
    conjugacy) rather than by SMC, so it also integrates over Sigma.

    Speed tips:
        - Use mcmc_kernel='hmc' (~2-5x faster than 'nuts')
        - Reduce num_particles to 50-100 for fast approximation
        - Reduce num_mcmc_steps to 3-5 for speed

    Args:
        static_trans_alpha: Dirichlet prior concentration for static transitions.
            Default 1.0 (uniform). Use smaller values (e.g., 0.5) for weaker priors.
            MUST match the value used in compute_factorized_evidence (BQ).
        include_label_switching: Add log K! for the K! equivalent state
            labelings. MUST match the value used for BQ when comparing them.
        verbose: Print the per-term breakdown.

    Returns:
        Scalar log evidence. Use compute_factorized_evidence_smc_replicated for a
        Monte Carlo error estimate.
    """
    K = config.num_states
    
    def flatten_list(l):
        if l is None: 
            return None
        if isinstance(l, (list, tuple)):
            if len(l) == 0: 
                return None
            if isinstance(l[0], list):
                 flat_items = []
                 for sublist in l:
                     flat_items.extend(sublist)
                 return jnp.concatenate(flat_items, axis=0)
            return jnp.concatenate(l, axis=0)
        
        if hasattr(l, 'shape'):
             return jnp.concatenate(l, axis=0)
             
        return None

    gammas_flat = flatten_list(gammas)
    
    if isinstance(xis, list):
         counts_trans = sum([jnp.sum(x, axis=0) for x in xis]) # (K, K)
    else:
         counts_trans = jnp.sum(xis, axis=(0, 1))
         
    log_Z_trans = 0.0
    
    logit_trans_evidence = 0.0
    
    if data.x_trans is not None:
        if isinstance(xis, list):
             x_trans_inputs = jnp.concatenate([x for x in data.x_trans], axis=0)
             xis_flat = jnp.concatenate(xis, axis=0) 
        else:
             x_trans_inputs = data.x_trans.reshape(-1, data.x_trans.shape[-1])
             xis_flat = xis.reshape(-1, K, K)

        D_trans = x_trans_inputs.shape[1]
        
        for j in range(K): 
            dim_theta_j = D_trans * (K - 1)
            prior_var_trans = config.prior_var_bias_trans
            iteration_seed = seed + 200 + j
            
            targets_j = xis_flat[:, j, :] # (N, K)
            
            def prior_theta(theta_flat):
                dim_t = theta_flat.shape[0]
                indices = jnp.arange(dim_t)
                row_indices = indices // (K-1)
                is_bias = (row_indices == 0)
                lp_bias = jnp.sum(stats.norm.logpdf(theta_flat, 0.0, jnp.sqrt(prior_var_trans)), where=is_bias)
                scale = 1.0 / jnp.sqrt(config.l2_reg_trans)
                lp_reg = jnp.sum(stats.norm.logpdf(theta_flat, 0.0, scale), where=(~is_bias))
                
                return lp_bias + lp_reg

            def lik_theta(theta_flat):
                Theta_reduced = theta_flat.reshape(D_trans, K - 1)
                zeros = jnp.zeros((D_trans, 1))
                Theta_j = jnp.concatenate([Theta_reduced, zeros], axis=1) # (D, K)
                
                logits = x_trans_inputs @ Theta_j # (N, K)
                log_probs = jax.nn.log_softmax(logits) # (N, K)
                
                return jnp.sum(targets_j * log_probs)

            theta_is_bias = (jnp.arange(dim_theta_j) // (K - 1)) == 0
            theta_scales = jnp.where(theta_is_bias,
                                     jnp.sqrt(prior_var_trans),
                                     1.0 / jnp.sqrt(config.l2_reg_trans))
            lz = run_smc_factor(prior_theta, lik_theta,
                                diagonal_gaussian_prior_sampler(theta_scales),
                                dim_theta_j, num_particles, num_mcmc_steps,
                                iteration_seed, mcmc_kernel)
            logit_trans_evidence += lz
            
    else:
        for j in range(K):
            alpha_prior = static_trans_alpha * jnp.ones(K)  # Configurable Dirichlet concentration
            if config.transition_stickiness > 0.0:
                 alpha_prior = alpha_prior.at[j].add(config.transition_stickiness)
            
            row_counts = counts_trans[j]
            term1 = jax.scipy.special.gammaln(jnp.sum(alpha_prior)) - jax.scipy.special.gammaln(jnp.sum(alpha_prior + row_counts))
            term2 = jnp.sum(jax.scipy.special.gammaln(alpha_prior + row_counts) - jax.scipy.special.gammaln(alpha_prior))
            
            log_Z_trans += term1 + term2
            
        logit_trans_evidence = log_Z_trans
            
    log_Z_trans = logit_trans_evidence

    log_Z_glm = 0.0
    
    x_b_flat = flatten_list(data.x_bern)
    y_b_flat = flatten_list(data.y_bern)
    x_g_flat = flatten_list(data.x_gauss)
    y_g_flat = flatten_list(data.y_gauss)
    
    valid_bern = (params.W_bern is not None) and (x_b_flat is not None)
    valid_gauss = (params.W_gauss is not None) and (x_g_flat is not None)
    valid_multinom = (params.W_multinom is not None) and (data.x_multinom is not None)
    valid_negbin = (params.W_negbin is not None) and (data.x_negbin is not None)
    valid_gamma = (params.W_gamma is not None) and (data.x_gamma is not None)

    # The factorization log Z = log Z_init + log Z_trans + sum_k log Z_glm_k + entropy
    # only telescopes if every term uses the same emission log-likelihood. Resolve the
    # Gaussian rescaling once here and apply it to the per-state factors, to the
    # expected complete log-likelihood, and to log p(Y) alike.
    do_normalize_gauss = (
        config.normalize_gauss_loglik
        if config.normalize_gauss_loglik is not None
        else gauss_normalization_default(data)
    )

    gauss_segments = None
    gauss_seg_lengths = None
    if valid_gauss and data.gauss_first_idx is not None:
        first_flat = flatten_list(data.gauss_first_idx)
        last_flat = flatten_list(data.gauss_last_idx)
        counts = last_flat - first_flat
        seg_ids = jnp.repeat(jnp.arange(len(counts)), counts)
        gauss_segments = (first_flat, last_flat, seg_ids)
        gauss_seg_lengths = counts

    if valid_gauss:
        M_gauss = y_g_flat.shape[1]
        iw_scale, iw_dof = resolve_iw_prior(config, M_gauss)
        D_gauss = x_g_flat.shape[1]
        row_precisions = jnp.concatenate([
            jnp.array([1.0 / config.prior_var_bias_emit]),
            jnp.full(D_gauss - 1, config.l2_reg_gauss)
        ])
        seg_ids_g = gauss_segments[2] if gauss_segments is not None else None

    def weighted_gauss_ll(ll_point, state_idx):
        """
        Responsibility-weighted Gaussian log-lik for one state, matching exactly
        what compute_total_emissions contributes to the forward pass.
        """
        if gauss_segments is None:
            return jnp.sum(gammas_flat[:, state_idx] * ll_point)
        first_flat, last_flat, seg_ids = gauss_segments
        ll_seg = normalized_gauss_loglikelihoods(
            ll_point, first_flat, last_flat, y_g_flat.shape[1],
            seg_ids, do_normalize_gauss
        )
        return jnp.sum(gammas_flat[:, state_idx] * ll_seg)

    for k in range(K):
        iteration_seed = seed + 100 + k
        
        if valid_bern:
            gs = gammas_flat[:, k]
            center = params.W_bern[:, k]
            dim_w = center.shape[0]
            
            def prior_bern(w):
                prior_var_emit = config.prior_var_bias_emit
                lp_bias = jnp.sum(stats.norm.logpdf(w[0], 0.0, jnp.sqrt(prior_var_emit)))                
                l2_lambda = config.l2_reg_bern
                scale = 1.0 / jnp.sqrt(l2_lambda)
                lp_feat = jnp.sum(stats.norm.logpdf(w[1:], 0.0, scale))
                
                return lp_bias + lp_feat

            def lik_bern(w):
                ll_vec = bernoulli_glmhmm_loglikelihood(x_b_flat, y_b_flat, w)
                return jnp.sum(gs * ll_vec.squeeze())

            bern_scales = jnp.concatenate([
                jnp.array([jnp.sqrt(config.prior_var_bias_emit)]),
                jnp.full(dim_w - 1, 1.0 / jnp.sqrt(config.l2_reg_bern))
            ])
            lz = run_smc_factor(prior_bern, lik_bern,
                                diagonal_gaussian_prior_sampler(bern_scales),
                                dim_w, num_particles, num_mcmc_steps,
                                iteration_seed, mcmc_kernel)
            log_Z_glm += lz
            
        if valid_gauss:
             # conjugate: integrated in closed form over both W and Sigma,
             # so this factor needs no SMC at all
             w_samples = gaussian_sample_weights(
                 gammas_flat, k, M_gauss, seg_ids_g, gauss_seg_lengths, do_normalize_gauss
             )
             log_Z_glm += float(gaussian_factor_log_evidence(
                 x_g_flat, y_g_flat, w_samples, row_precisions, iw_scale, iw_dof
             ))

        if valid_multinom:
             x_m_flat = flatten_list(data.x_multinom)
             y_m_flat = flatten_list(data.y_multinom)
             
             center_full = params.W_multinom[:, :, k] # (D, C)
             D_mn, C_mn = center_full.shape
             
             center_reduced = center_full[:, :-1]
             w_flat_center = center_reduced.reshape(-1)
             dim_flat = w_flat_center.shape[0]
             gs = gammas_flat[:, k]
             
             def prior_multinom(w_flat):
                 reg = config.l2_reg_multinom
                 scale = 1.0 / jnp.sqrt(reg)
                 num_classes_reduced = C_mn - 1
                 indices = jnp.arange(dim_flat)
                 row_indices = indices // num_classes_reduced
                 is_bias = (row_indices == 0)
                 
                 prior_var_emit = config.prior_var_bias_emit
                 lp_bias = jnp.sum(stats.norm.logpdf(w_flat, 0.0, jnp.sqrt(prior_var_emit)), where=is_bias)
                 lp_reg = jnp.sum(stats.norm.logpdf(w_flat, 0.0, scale), where=(~is_bias))
                 return lp_bias + lp_reg

             def lik_multinom(w_flat):
                 W_reduced = w_flat.reshape(D_mn, C_mn - 1)
                 zeros = jnp.zeros((D_mn, 1))
                 W_full = jnp.concatenate([W_reduced, zeros], axis=1)
                 ll_vec = multinomial_glmhmm_loglikelihood(x_m_flat, y_m_flat, W_full)
                 return jnp.sum(gs * ll_vec)
             
             mn_is_bias = (jnp.arange(dim_flat) // (C_mn - 1)) == 0
             mn_scales = jnp.where(mn_is_bias,
                                   jnp.sqrt(config.prior_var_bias_emit),
                                   1.0 / jnp.sqrt(config.l2_reg_multinom))
             lz = run_smc_factor(prior_multinom, lik_multinom,
                                 diagonal_gaussian_prior_sampler(mn_scales),
                                 dim_flat, num_particles, num_mcmc_steps,
                                 iteration_seed + 2, mcmc_kernel)
             log_Z_glm += lz

        if valid_negbin:
             x_nb_flat = flatten_list(data.x_negbin)
             y_nb_flat = flatten_list(data.y_negbin)
             
             D_nb = params.W_negbin[:, k].shape[0]
             dim_flat = D_nb + 1
             gs = gammas_flat[:, k]

             def prior_negbin(theta):
                 w = theta[:-1]
                 log_phi = theta[-1]
                 reg = config.l2_reg_negbin
                 scale = 1.0 / jnp.sqrt(reg)
                 prior_var_emit = config.prior_var_bias_emit

                 lp_w_bias = stats.norm.logpdf(w[0], 0.0, jnp.sqrt(prior_var_emit))
                 lp_w_feat = jnp.sum(stats.norm.logpdf(w[1:], 0.0, scale))
                 lp_w = lp_w_bias + lp_w_feat
                 lp_phi = stats.norm.logpdf(log_phi, 0.0, 1.0)
                 return lp_w + lp_phi

             def lik_negbin(theta):
                 w = theta[:-1]
                 log_phi = theta[-1]
                 phi = jnp.exp(log_phi)
                 # the shared emission fn expects (D, K) weights and (K,) dispersion;
                 # this factor covers a single state, so promote to K=1 and drop the axis
                 ll_vec = negbin_glmhmm_loglikelihood(
                     x_nb_flat, y_nb_flat, w[:, jnp.newaxis], jnp.atleast_1d(phi)
                 )[:, 0]
                 return jnp.sum(gs * ll_vec)

             nb_scales = jnp.concatenate([
                 jnp.array([jnp.sqrt(config.prior_var_bias_emit)]),
                 jnp.full(D_nb - 1, 1.0 / jnp.sqrt(config.l2_reg_negbin)),
                 jnp.array([1.0])          # log phi ~ N(0, 1)
             ])
             lz = run_smc_factor(prior_negbin, lik_negbin,
                                 diagonal_gaussian_prior_sampler(nb_scales),
                                 dim_flat, num_particles, num_mcmc_steps,
                                 iteration_seed + 3, mcmc_kernel)
             log_Z_glm += lz

        if valid_gamma:
             x_gm_flat = flatten_list(data.x_gamma)
             y_gm_flat = flatten_list(data.y_gamma)
             
             D_gm = params.W_gamma[:, k].shape[0]
             dim_flat = D_gm + 1
             gs = gammas_flat[:, k]

             def prior_gamma(theta):
                 w = theta[:-1]
                 log_alpha = theta[-1]
                 
                 reg = config.l2_reg_gamma
                 scale = 1.0 / jnp.sqrt(reg)
                 prior_var_emit = config.prior_var_bias_emit

                 lp_w_bias = stats.norm.logpdf(w[0], 0.0, jnp.sqrt(prior_var_emit))
                 lp_w_feat = jnp.sum(stats.norm.logpdf(w[1:], 0.0, scale))
                 lp_w = lp_w_bias + lp_w_feat
                 
                 lp_alpha = stats.norm.logpdf(log_alpha, 0.0, 1.0)
                 return lp_w + lp_alpha

             def lik_gamma(theta):
                 w = theta[:-1]
                 log_alpha = theta[-1]
                 alpha = jnp.exp(log_alpha)
                 # single-state factor; promote to the (D, K) / (K,) shapes the
                 # shared emission fn expects, then drop the state axis
                 ll_vec = gamma_glmhmm_loglikelihood(
                     x_gm_flat, y_gm_flat, w[:, jnp.newaxis], jnp.atleast_1d(alpha)
                 )[:, 0]
                 return jnp.sum(gs * ll_vec)

             gm_scales = jnp.concatenate([
                 jnp.array([jnp.sqrt(config.prior_var_bias_emit)]),
                 jnp.full(D_gm - 1, 1.0 / jnp.sqrt(config.l2_reg_gamma)),
                 jnp.array([1.0])          # log alpha ~ N(0, 1)
             ])
             lz = run_smc_factor(prior_gamma, lik_gamma,
                                 diagonal_gaussian_prior_sampler(gm_scales),
                                 dim_flat, num_particles, num_mcmc_steps,
                                 iteration_seed + 4, mcmc_kernel)
             log_Z_glm += lz
    
    log_P_Y = compute_log_marginal_likelihood(params, data, do_normalize_gauss)

    if isinstance(gammas, list):
         expected_ll_init = sum([jnp.sum(g[0] * params.log_pi0) for g in gammas])
    else:
         expected_ll_init = jnp.sum(gammas[:, 0, :] * params.log_pi0)

    if data.x_trans is not None and params.theta is not None:
         expected_ll_trans = 0.0
         if isinstance(xis, list):
             for i, x_seq in enumerate(data.x_trans):
                 x_in = x_seq 
                 logits_t = jnp.tensordot(x_in, params.theta, axes=(1, 0))
                 log_A_t = jax.nn.log_softmax(logits_t, axis=-1)
                 expected_ll_trans += jnp.sum(xis[i] * log_A_t)
         else:
             x_in = data.x_trans
             logits_t = jnp.tensordot(x_in, params.theta, axes=(2, 0)) 
             log_A_t = jax.nn.log_softmax(logits_t, axis=-1)
             expected_ll_trans = jnp.sum(xis * log_A_t)
             
    elif isinstance(xis, list):
         expected_ll_trans = sum([jnp.sum(x * params.log_A) for x in xis])
    else:
         expected_ll_trans = jnp.sum(xis * params.log_A)
         
    expected_ll_emit = 0.0
    if valid_bern:
        ll_bern = bernoulli_glmhmm_loglikelihood(x_b_flat, y_b_flat, params.W_bern)
        expected_ll_emit += jnp.sum(gammas_flat * ll_bern)
        
    if valid_gauss:
        total_gauss_ll = 0.0
        for k in range(K):
             ll_k = gaussian_glmhmm_loglikelihoods(
                 x_g_flat, y_g_flat, params.W_gauss[:, :, k], params.Sigma_gauss[:, :, k]
             )
             total_gauss_ll += weighted_gauss_ll(ll_k, k)
        expected_ll_emit += total_gauss_ll

    if valid_multinom:
        ll_unweighted = multinomial_glmhmm_loglikelihood(flatten_list(data.x_multinom), flatten_list(data.y_multinom), params.W_multinom)
        expected_ll_emit += jnp.sum(gammas_flat * ll_unweighted)

    if valid_negbin:
        ll_nb = negbin_glmhmm_loglikelihood(flatten_list(data.x_negbin), flatten_list(data.y_negbin), params.W_negbin, params.phi_negbin)
        expected_ll_emit += jnp.sum(gammas_flat * ll_nb)

    if valid_gamma:
        ll_gm = gamma_glmhmm_loglikelihood(flatten_list(data.x_gamma), flatten_list(data.y_gamma), params.W_gamma, params.alpha_gamma)
        expected_ll_emit += jnp.sum(gammas_flat * ll_gm)

    expected_complete_ll = expected_ll_init + expected_ll_trans + expected_ll_emit
    entropy = log_P_Y - expected_complete_ll
    
    if isinstance(gammas, list):
         n_sequences = len(gammas)
         init_counts = sum([g[0] for g in gammas]) # (K,)
    else:
         n_sequences = gammas.shape[0]
         if gammas.ndim == 3:
             init_counts = jnp.sum(gammas[:, 0, :], axis=0)
         else:
             init_counts = gammas[0]

    alpha_prior = static_trans_alpha  # Use same prior as transitions
    term1 = jnp.sum(jax.scipy.special.gammaln(alpha_prior + init_counts))
    term2 = jax.scipy.special.gammaln(jnp.sum(alpha_prior + init_counts))
    term3 = jnp.sum(jax.scipy.special.gammaln(jnp.array([alpha_prior] * K)))
    term4 = jax.scipy.special.gammaln(K * alpha_prior)
    
    log_Z_init = (term1 - term2) - (term3 - term4)
    
    # The factorisation expands around one labeling of the states, but the
    # parameter posterior has K! symmetric modes; log K! restores the rest.
    log_multiplicity = jax.scipy.special.gammaln(K + 1.0) if include_label_switching else 0.0

    total_smc_evidence = log_Z_init + log_Z_trans + log_Z_glm + entropy + log_multiplicity

    if verbose:
        print(f"  [SMC K={K}] log_Z_init:  {float(log_Z_init):.2f}")
        print(f"  [SMC K={K}] log_Z_trans: {float(log_Z_trans):.2f}")
        print(f"  [SMC K={K}] log_Z_glm:   {float(log_Z_glm):.2f}")
        print(f"  [SMC K={K}] entropy:     {float(entropy):.2f}")
        if include_label_switching:
            print(f"  [SMC K={K}] log K!:      {float(log_multiplicity):.2f}")
        print(f"  [SMC K={K}] total:       {float(total_smc_evidence):.2f}")

    return total_smc_evidence


def compute_factorized_evidence_smc_replicated(
    data: GLMHMMData,
    params: GLMHMMParams,
    config: GLMHMMConfig,
    gammas,
    xis,
    n_replicates: int = 5,
    seed: int = 0,
    verbose: bool = True,
    **kwargs
):
    """
    Runs the factorized SMC estimator several times and reports its spread.

    SMC returns a stochastic estimate whose error is not available from a single
    run, unlike BQ which reports a quadrature variance. Independent replicates
    with different seeds give the Monte Carlo standard deviation, which is the
    number to compare against differences in log evidence between models.

    Args:
        n_replicates: Number of independent runs.
        seed: Base seed; replicate r uses seed + 1000 * r.
        verbose: Print a summary (per-replicate breakdowns are suppressed).
        **kwargs: Forwarded to compute_factorized_evidence_smc.

    Returns:
        (mean_log_evidence, std_log_evidence, all_estimates)
    """
    import numpy as np

    estimates = []
    for r in range(n_replicates):
        lz = compute_factorized_evidence_smc(
            data, params, config, gammas, xis,
            seed=seed + 1000 * r, verbose=False, **kwargs
        )
        estimates.append(float(lz))

    mean_lz = float(np.mean(estimates))
    std_lz = float(np.std(estimates, ddof=1)) if n_replicates > 1 else 0.0

    if verbose:
        print(f"  [SMC K={config.num_states}] {n_replicates} replicates: "
              f"{mean_lz:.2f} +/- {std_lz:.2f}")
        print(f"  [SMC K={config.num_states}] estimates: "
              f"{[round(e, 2) for e in estimates]}")

    return mean_lz, std_lz, estimates
