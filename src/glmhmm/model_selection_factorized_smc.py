
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import flatten_util
from functools import partial
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.nuts as nuts

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import compute_log_marginal_likelihood
from .emissions import (
    bernoulli_glmhmm_loglikelihood,
    multinomial_glmhmm_loglikelihood,
    negbin_glmhmm_loglikelihood,
    gamma_glmhmm_loglikelihood,
    gaussian_glmhmm_loglikelihoods
)

def fast_smc_evidence(smc_algorithm, initial_particles, key):
    """
    Runs the SMC loop using jax.lax.while_loop for JIT compilation.
    """
    init_state = smc_algorithm.init(initial_particles)
    key, subk = jax.random.split(key)
    _, info = smc_algorithm.step(subk, init_state)
    
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

def run_smc_factor(
    logprior_fn,
    loglikelihood_fn,
    initial_center,
    dim,
    num_particles=500,
    num_mcmc_steps=10,
    key_seed=0
):
    """
    Runs SMC to estimate the normalizing constant (evidence) for a single factor.
    Z = \int exp(loglikelihood_fn(theta)) * exp(logprior_fn(theta)) dtheta
    """
    key = jax.random.PRNGKey(key_seed)
    key, init_key = jax.random.split(key)
    initial_particles = jax.random.normal(init_key, (num_particles, dim))
    
    mcmc_params = {
        'step_size': jnp.array([1e-2]), 
        'inverse_mass_matrix': jnp.expand_dims(jnp.eye(dim), 0)
    }

    smc_loop = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=nuts.build_kernel(),
        mcmc_init_fn=nuts.init,
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
    num_particles: int = 500,
    num_mcmc_steps: int = 20,
    seed: int = 0
):
    """
    Computes evidence using Factorized SMC.
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
        x_trans_flat = flatten_list(data.x_trans) 
        if isinstance(xis, list):
             x_trans_inputs = jnp.concatenate([x for x in data.x_trans], axis=0)
             xis_flat = jnp.concatenate(xis, axis=0) 
        else:
             x_trans_inputs = data.x_trans.reshape(-1, data.x_trans.shape[-1])
             xis_flat = xis.reshape(-1, K, K)

        D_trans = x_trans_inputs.shape[1]
        
        for j in range(K): 
            center_theta_j_full = params.theta[:, j, :] 
            center_theta_j_reduced = center_theta_j_full[:, :-1]
            
            dim_theta_j = D_trans * (K - 1)
            iteration_seed = seed + 200 + j
            
            targets_j = xis_flat[:, j, :] # (N, K)
            
            def prior_theta(theta_flat):
                reg = config.l2_reg_trans
                scale = 1.0 / jnp.sqrt(reg)
                return jnp.sum(stats.norm.logpdf(theta_flat, 0.0, scale))

            def lik_theta(theta_flat):
                Theta_reduced = theta_flat.reshape(D_trans, K - 1)
                zeros = jnp.zeros((D_trans, 1))
                Theta_j = jnp.concatenate([Theta_reduced, zeros], axis=1) # (D, K)
                
                logits = x_trans_inputs @ Theta_j # (N, K)
                log_probs = jax.nn.log_softmax(logits) # (N, K)
                
                return jnp.sum(targets_j * log_probs)

            lz = run_smc_factor(prior_theta, lik_theta, center_theta_j_reduced.flatten(), dim_theta_j, num_particles, num_mcmc_steps, iteration_seed)
            logit_trans_evidence += lz
            
    else:
        # static transitions (log_A)
        for j in range(K):
            row_counts = counts_trans[j]
            dim_trans = K
            
            def prior_fn(logit_row):
                reg = config.l2_reg_trans
                scale = 1.0 / jnp.sqrt(reg)
                return jnp.sum(stats.norm.logpdf(logit_row, 0.0, scale))

            def lik_fn(logit_row):
                log_probs = jax.nn.log_softmax(logit_row)
                return jnp.dot(row_counts, log_probs)
                
            lz = run_smc_factor(prior_fn, lik_fn, params.log_A[j], dim_trans, num_particles, num_mcmc_steps, seed + j)
            logit_trans_evidence += lz
            
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
    
    for k in range(K):
        iteration_seed = seed + 100 + k
        
        if valid_bern:
            gs = gammas_flat[:, k]
            center = params.W_bern[:, k]
            dim_w = center.shape[0]
            
            def prior_bern(w):
                l2_lambda = config.l2_reg_bern
                lp = jnp.sum(stats.norm.logpdf(w[0], 0.0, 10.0))
                if l2_lambda > 0:
                    scale = 1.0 / jnp.sqrt(l2_lambda)
                    lp += jnp.sum(stats.norm.logpdf(w[1:], 0.0, scale))
                else:
                    lp += jnp.sum(stats.norm.logpdf(w[1:], 0.0, 10.0))
                return lp

            def lik_bern(w):
                logits = x_b_flat @ w
                ll_point = bernoulli_glmhmm_loglikelihood(x_b_flat, y_b_flat, w) 
                ll_vec = bernoulli_glmhmm_loglikelihood(x_b_flat, y_b_flat, w) 
                return jnp.sum(gs * ll_vec.squeeze())

            lz = run_smc_factor(prior_bern, lik_bern, center, dim_w, num_particles, num_mcmc_steps, iteration_seed)
            log_Z_glm += lz
            
        if valid_gauss:
             W_center = params.W_gauss[:, :, k]
             D_in, M_out = W_center.shape
             dim_flat = D_in * M_out
             
             if data.gauss_first_idx is not None:
                first_flat = flatten_list(data.gauss_first_idx)
                last_flat = flatten_list(data.gauss_last_idx)
                counts = last_flat - first_flat
                seg_ids = jnp.repeat(jnp.arange(len(counts)), counts)
                gs_expanded = gammas_flat[seg_ids, k]
             else:
                gs_expanded = gammas_flat[:, k]
             
             def prior_gauss(w_flat):
                 reg = config.l2_reg_gauss
                 scale = 1.0 / jnp.sqrt(reg)
                 return jnp.sum(stats.norm.logpdf(w_flat, 0.0, scale))
                 
             def lik_gauss(w_flat):
                 W = w_flat.reshape(D_in, M_out)
                 Sigma_k = params.Sigma_gauss[:, :, k]
                 ll_vec = gaussian_glmhmm_loglikelihoods(x_g_flat, y_g_flat, W, Sigma_k)
                 return jnp.sum(gs_expanded * ll_vec)
                 
             lz = run_smc_factor(prior_gauss, lik_gauss, W_center.flatten(), dim_flat, num_particles, num_mcmc_steps, iteration_seed + 1)
             log_Z_glm += lz

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
                 return jnp.sum(stats.norm.logpdf(w_flat, 0.0, scale))

             def lik_multinom(w_flat):
                 W_reduced = w_flat.reshape(D_mn, C_mn - 1)
                 zeros = jnp.zeros((D_mn, 1))
                 W_full = jnp.concatenate([W_reduced, zeros], axis=1)
                 ll_vec = multinomial_glmhmm_loglikelihood(x_m_flat, y_m_flat, W_full)
                 return jnp.sum(gs * ll_vec)
             
             lz = run_smc_factor(prior_multinom, lik_multinom, w_flat_center, dim_flat, num_particles, num_mcmc_steps, iteration_seed + 2)
             log_Z_glm += lz

        if valid_negbin:
             x_nb_flat = flatten_list(data.x_negbin)
             y_nb_flat = flatten_list(data.y_negbin)
             
             w_center = params.W_negbin[:, k]
             phi_center = params.phi_negbin[k]
             center_combined = jnp.concatenate([w_center, jnp.log(jnp.array([phi_center]))])
             D_nb = w_center.shape[0]
             dim_flat = D_nb + 1
             gs = gammas_flat[:, k]

             def prior_negbin(theta):
                 w = theta[:-1]
                 log_phi = theta[-1]
                 reg = config.l2_reg_negbin
                 scale = 1.0 / jnp.sqrt(reg)
                 lp_w = jnp.sum(stats.norm.logpdf(w, 0.0, scale))
                 lp_phi = stats.norm.logpdf(log_phi, 0.0, 1.0)
                 return lp_w + lp_phi

             def lik_negbin(theta):
                 w = theta[:-1]
                 log_phi = theta[-1]
                 phi = jnp.exp(log_phi)
                 ll_vec = negbin_glmhmm_loglikelihood(x_nb_flat, y_nb_flat, w, phi)
                 return jnp.sum(gs * ll_vec)

             lz = run_smc_factor(prior_negbin, lik_negbin, center_combined, dim_flat, num_particles, num_mcmc_steps, iteration_seed + 3)
             log_Z_glm += lz

        if valid_gamma:
             x_gm_flat = flatten_list(data.x_gamma)
             y_gm_flat = flatten_list(data.y_gamma)
             
             w_center = params.W_gamma[:, k]
             alpha_center = params.alpha_gamma[k]
             center_combined = jnp.concatenate([w_center, jnp.log(jnp.array([alpha_center]))])
             D_gm = w_center.shape[0]
             dim_flat = D_gm + 1
             gs = gammas_flat[:, k]

             def prior_gamma(theta):
                 w = theta[:-1]
                 log_alpha = theta[-1]
                 
                 reg = config.l2_reg_gamma
                 scale = 1.0 / jnp.sqrt(reg)
                 lp_w = jnp.sum(stats.norm.logpdf(w, 0.0, scale))
                 lp_alpha = stats.norm.logpdf(log_alpha, 0.0, 1.0)
                 return lp_w + lp_alpha

             def lik_gamma(theta):
                 w = theta[:-1]
                 log_alpha = theta[-1]
                 alpha = jnp.exp(log_alpha)
                 ll_vec = gamma_glmhmm_loglikelihood(x_gm_flat, y_gm_flat, w, alpha)
                 return jnp.sum(gs * ll_vec)

             lz = run_smc_factor(prior_gamma, lik_gamma, center_combined, dim_flat, num_particles, num_mcmc_steps, iteration_seed + 4)
             log_Z_glm += lz
    
    log_P_Y = compute_log_marginal_likelihood(params, data)
    
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
             if data.gauss_first_idx is not None:
                first_flat = flatten_list(data.gauss_first_idx)
                last_flat = flatten_list(data.gauss_last_idx)
                counts = last_flat - first_flat
                seg_ids = jnp.repeat(jnp.arange(len(counts)), counts)
                gs_expanded = gammas_flat[seg_ids, k]
             else:
                gs_expanded = gammas_flat[:, k]
             total_gauss_ll += jnp.sum(gs_expanded * ll_k)
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

    alpha_prior = 1.0
    term1 = jnp.sum(jax.scipy.special.gammaln(alpha_prior + init_counts))
    term2 = jax.scipy.special.gammaln(jnp.sum(alpha_prior + init_counts))
    term3 = jnp.sum(jax.scipy.special.gammaln(alpha_prior))
    term4 = jax.scipy.special.gammaln(jnp.sum(jnp.array([alpha_prior] * K)))
    
    log_Z_init = (term1 - term2) - (term3 - term4)
    
    total_smc_evidence = log_Z_init + log_Z_trans + log_Z_glm + entropy
    return total_smc_evidence
