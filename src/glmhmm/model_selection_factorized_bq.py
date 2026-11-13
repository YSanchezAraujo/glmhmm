
import jax
import jax.numpy as jnp
from jax import random, hessian, vmap, jit
from tinygp import kernels
from typing import Callable, Tuple, Optional
from functools import partial

import warnings

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import compute_log_marginal_likelihood, gauss_normalization_default
from .m_step_utils import run_lbfgs_optimizer
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

@jit
def compute_kernel_expectations(
    kernel_variance: float,
    lengthscale: jnp.ndarray, # (D,) or scalar
    X_train: jnp.ndarray,     # (N, D)
    q_mean: jnp.ndarray,      # (D,)
    q_cov: jnp.ndarray        # (D, D)
) -> jnp.ndarray:
    r"""
    Computes vector z where z_i = \int k(x_i, x) q(x) dx
    """
    D = q_mean.shape[0]
    
    if lengthscale.ndim == 0 or lengthscale.size == 1:
        ls_val = jnp.squeeze(lengthscale)
        Lambda = jnp.eye(D) * (ls_val**2)
    else:
        Lambda = jnp.diag(lengthscale**2)
        
    S_plus_Lambda = q_cov + Lambda
    
    diff = X_train - q_mean # (N, D)
    
    L = jnp.linalg.cholesky(S_plus_Lambda)
    z_solve = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True) # (D, N)
    mahal = jnp.sum(z_solve**2, axis=0) # (N,)
    
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    det_Lambda = jnp.prod(jnp.diag(Lambda))
    det_S_plus_Lambda = jnp.exp(log_det)
    
    prefactor = jnp.sqrt(det_Lambda / det_S_plus_Lambda)
    
    z = kernel_variance * prefactor * jnp.exp(-0.5 * mahal)
    return z

@partial(jit, static_argnums=(3,))
def compute_prior_integral_variance(
    kernel_variance: float,
    lengthscale: jnp.ndarray,
    q_cov: jnp.ndarray,
    dim: int
) -> float:
    r"""
    Computes \iint k(x, x') q(x) q(x') dx dx'
    For RBF kernel and q=N(m, S).
    Equivalent to E[k(x, x')] where x, x' ~ iid q.
    Let delta = x - x' ~ N(0, 2S).
    k(delta) = var * exp(-0.5 delta^T Lambda^-1 delta).
    Integral is var * det(I + 2S Lambda^-1)^-1/2.
    """
    if lengthscale.ndim == 0 or lengthscale.size == 1:
        ls_val = jnp.squeeze(lengthscale)
        inv_Lambda = jnp.eye(dim) / (ls_val**2)
    else:
        inv_Lambda = jnp.diag(1.0 / lengthscale**2)
        
    M = jnp.eye(dim) + 2.0 * q_cov @ inv_Lambda
    sign, logdet = jnp.linalg.slogdet(M)
    
    return kernel_variance * jnp.exp(-0.5 * logdet)

@jit
def integrate_rbf_against_gaussian(
    kernel_variance: float,
    lengthscale: jnp.ndarray,
    X_train: jnp.ndarray,
    alpha: jnp.ndarray,
    q_mean: jnp.ndarray,
    q_cov: jnp.ndarray
) -> float:
    z = compute_kernel_expectations(kernel_variance, lengthscale, X_train, q_mean, q_cov)
    return jnp.dot(alpha, z)


def stable_cholesky(K_mat: jnp.ndarray, base_jitter: float, max_tries: int = 8):
    """
    Cholesky factor of a kernel matrix, escalating the jitter until it succeeds.

    A fitted lengthscale can be much larger than the default of 1, which makes the
    Gram matrix strongly correlated and numerically indefinite at a fixed small
    jitter. JAX returns NaN rather than raising in that case, so the failure has to
    be detected explicitly. Growing the jitter damps the GP correction toward the
    plain Monte Carlo mean, which is the right way to degrade.

    Args:
        K_mat: Kernel matrix, (n, n).
        base_jitter: Starting jitter, in the units of the kernel.
        max_tries: How many times to multiply the jitter by 10.

    Returns:
        (L, jitter_used) or (None, jitter_tried) if it never succeeded.
    """
    eye = jnp.eye(K_mat.shape[0], dtype=K_mat.dtype)
    jitter = base_jitter
    for _ in range(max_tries):
        L = jnp.linalg.cholesky(K_mat + jitter * eye)
        if bool(jnp.all(jnp.isfinite(L))):
            return L, jitter
        jitter *= 10.0
    return None, jitter


def fit_gp_hyperparameters(
    z_samples: jnp.ndarray,
    y_train: jnp.ndarray,
    init_lengthscale: float = 1.0,
    init_variance: float = 1.0,
    jitter: float = 1e-5,
    max_points: int = 512,
    key: Optional[random.PRNGKey] = None
) -> Tuple[float, float]:
    """
    Fits the RBF lengthscale and variance by maximising the GP marginal
    likelihood, rather than leaving them at their initial values.

    Fitting is done on a random subset, since each objective evaluation costs a
    dense Cholesky and the full sample count is O(1000s).

    Args:
        z_samples: Whitened sample locations, (n, dim).
        y_train: Centred importance ratios, (n,).
        init_lengthscale: Starting lengthscale.
        init_variance: Starting kernel variance.
        jitter: Diagonal jitter.
        max_points: Subset size used for fitting.
        key: PRNG key for subsetting.

    Returns:
        (lengthscale, variance), both positive.
    """
    n = z_samples.shape[0]
    if n > max_points:
        if key is None:
            key = random.PRNGKey(0)
        idx = random.choice(key, n, shape=(max_points,), replace=False)
        z_samples, y_train = z_samples[idx], y_train[idx]

    n_fit = z_samples.shape[0]
    eye = jnp.eye(n_fit)

    def neg_log_marginal(log_params):
        # clip to keep the Cholesky well posed under an unconstrained optimiser
        log_ls, log_var = jnp.clip(log_params[0], -5.0, 5.0), jnp.clip(log_params[1], -5.0, 5.0)
        var = jnp.exp(log_var)
        K_mat = var * kernels.ExpSquared(scale=jnp.exp(log_ls))(z_samples, z_samples)
        # jitter relative to the kernel scale, so it stays meaningful as var moves
        K_mat = K_mat + jitter * var * eye
        L = jnp.linalg.cholesky(K_mat)
        a = jax.scipy.linalg.solve_triangular(L, y_train, lower=True)
        return 0.5 * jnp.sum(a ** 2) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * n_fit * jnp.log(2 * jnp.pi)

    init = jnp.array([jnp.log(init_lengthscale), jnp.log(init_variance)])
    fitted, _ = run_lbfgs_optimizer(neg_log_marginal, init, max_iter=100, tol=1e-4)
    fitted = jnp.clip(fitted, -5.0, 5.0)
    return jnp.exp(fitted[0]), jnp.exp(fitted[1])


def run_bq_tinygp(
    log_prob_fn: Callable[[jnp.ndarray], float],
    center: jnp.ndarray,
    dim: int,
    n_samples: int = 150,
    key: Optional[random.PRNGKey] = None,
    lengthscale_init: float = 1.0,
    tune_hyperparameters: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Estimates a single factor's log evidence by Laplace-plus-Bayesian-quadrature.

    A Laplace approximation at `center` provides the base measure; importance
    ratios against it are then integrated with a GP quadrature rule rather than a
    plain Monte Carlo mean.

    Args:
        log_prob_fn: Unnormalised log density of the factor.
        center: Expansion point. Assumed to be the mode of log_prob_fn.
        dim: Dimension of the factor.
        n_samples: Number of importance samples.
        key: PRNG key.
        lengthscale_init: Initial (and, if not tuning, final) RBF lengthscale.
        tune_hyperparameters: Fit the GP lengthscale and variance to the ratios.

    Returns:
        log_evidence, var_log_evidence, diagnostics. Diagnostics reports the
        fraction of non-finite importance ratios that had to be sanitised and the
        kernel hyperparameters actually used.
    """
    if key is None: key = random.PRNGKey(0)

    key, subkey = random.split(key)
    H = hessian(log_prob_fn)(center)
    H_reg = H - 1e-4 * jnp.eye(dim)
    Sigma_laplace = jnp.linalg.inv(-H_reg)
    
    val_map = log_prob_fn(center)
    sign, logdet = jnp.linalg.slogdet(-H_reg)
    log_z_laplace = val_map + 0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * logdet
    
    key, subkey = random.split(key)
    L_chol = jnp.linalg.cholesky(Sigma_laplace)
    z = random.normal(subkey, (n_samples, dim))
    samples = center + z @ L_chol.T
    
    def get_log_q(theta):
        d = theta - center
        sol = jax.scipy.linalg.solve_triangular(L_chol, d, lower=True)
        return -0.5*dim*jnp.log(2*jnp.pi) - jnp.sum(jnp.log(jnp.diag(L_chol))) - 0.5 * jnp.sum(sol**2)
        
    v_log_prob = vmap(log_prob_fn)
    v_log_q = vmap(get_log_q)
    
    log_probs = v_log_prob(samples)
    log_qs = v_log_q(samples)
    
    log_ratios = log_probs - log_qs - log_z_laplace
    ratios = jnp.exp(log_ratios)
    # a diverging importance ratio means the Laplace base measure is a poor fit;
    # it still has to be made finite, but the caller is told how often it happened
    n_bad = int(jnp.sum(~jnp.isfinite(ratios)))
    ratios = jnp.nan_to_num(ratios, nan=0.0, posinf=1e6, neginf=0.0)

    mean_ratio = jnp.mean(ratios)
    y_train = ratios - mean_ratio
    z_samples = jax.scipy.linalg.solve_triangular(L_chol, (samples - center).T, lower=True).T

    if tune_hyperparameters:
        key, subkey = random.split(key)
        lengthscale, kernel_var = fit_gp_hyperparameters(
            z_samples, y_train, init_lengthscale=lengthscale_init, key=subkey
        )
    else:
        lengthscale, kernel_var = jnp.asarray(lengthscale_init), jnp.asarray(1.0)

    kernel = kernel_var * kernels.ExpSquared(scale=lengthscale)
    K_mat = kernel(z_samples, z_samples)
    L_K, used_jitter = stable_cholesky(K_mat, base_jitter=1e-5 * float(kernel_var))

    if L_K is None:
        # GP is unusable: fall back to plain Monte Carlo, which is what the
        # quadrature is correcting in the first place
        bq_integral_scaled = mean_ratio
        posterior_var_dist = jnp.var(ratios) / n_samples
        gp_failed = True
    else:
        gp_failed = False
        alpha = jax.scipy.linalg.solve_triangular(
            L_K.T, jax.scipy.linalg.solve_triangular(L_K, y_train, lower=True), lower=False)

        z_vec = compute_kernel_expectations(
            kernel_variance=kernel_var,
            lengthscale=jnp.atleast_1d(lengthscale),
            X_train=z_samples,
            q_mean=jnp.zeros(dim),
            q_cov=jnp.eye(dim)
        )

        integral_correction_z = jnp.dot(z_vec, alpha)
        bq_integral_scaled = integral_correction_z + mean_ratio

        prior_var_integral = compute_prior_integral_variance(
            kernel_variance=kernel_var,
            lengthscale=jnp.atleast_1d(lengthscale),
            q_cov=jnp.eye(dim),
            dim=dim
        )

        K_inv_z = jax.scipy.linalg.solve_triangular(
            L_K.T, jax.scipy.linalg.solve_triangular(L_K, z_vec, lower=True), lower=False)
        var_reduction = jnp.dot(z_vec, K_inv_z)
        posterior_var_dist = prior_var_integral - var_reduction
        posterior_var_dist = jnp.maximum(posterior_var_dist, 0.0)
    bq_val_safe = jnp.maximum(bq_integral_scaled, 1e-4)
    var_log_Z = posterior_var_dist / (bq_val_safe**2)

    diagnostics = {
        "nonfinite_ratio_fraction": n_bad / n_samples,
        "lengthscale": float(lengthscale),
        "kernel_variance": float(kernel_var),
        "n_samples": n_samples,
        "gp_jitter": float(used_jitter),
        "gp_failed": gp_failed,
    }
    return log_z_laplace + jnp.log(bq_val_safe), var_log_Z, diagnostics

def compute_factorized_evidence(
    data: GLMHMMData,
    params: GLMHMMParams,
    config: GLMHMMConfig,
    gammas: jnp.ndarray, # (N_seq, T, K)
    xis: jnp.ndarray,    # (N_seq, T-1, K, K)
    key: Optional[random.PRNGKey] = None,
    static_trans_alpha: float = 1.0,  # Dirichlet prior for static transitions (lower = weaker prior)
    min_samples: int = 2000,
    samples_per_dim: int = 500,
    max_samples: int = 8000,
    tune_gp_hyperparameters: bool = True,
    include_label_switching: bool = True,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Computes total log evidence using the factorization

        log Z = log Z_init + log Z_trans + sum_k log Z_glm_k + H[q(Z)]

    See docs/model_and_model_selection.qmd for the derivation and for the
    assumptions and approximations involved. In particular the per-factor
    integrals are evaluated at *expected* latent counts, which is the dominant
    approximation and is not a bound.

    The Gaussian factor is handled analytically (matrix-normal / inverse-Wishart
    conjugacy) rather than by quadrature, so it also integrates over Sigma.

    Args:
        data: Dataset.
        params: Fitted parameters, assumed to be at a converged MAP optimum --
                the Laplace expansion is taken at these values.
        config: Configuration; supplies the priors implied by the regularisation.
        gammas: State responsibilities from the E-step.
        xis: Transition responsibilities from the E-step.
        key: PRNG key.
        static_trans_alpha: Dirichlet prior concentration for static transitions.
            Default 1.0 (uniform). Use smaller values (e.g., 0.5) for weaker priors.
        min_samples: Floor on importance samples per factor.
        samples_per_dim: Importance samples per parameter dimension.
        max_samples: Ceiling on importance samples per factor. Quadrature costs
            O(n^2) memory and O(n^3) time per factor per state, so this bounds
            the dominant runtime term.
        tune_gp_hyperparameters: Fit the quadrature kernel to the ratios instead
            of using fixed unit lengthscale and variance.
        include_label_switching: Add log K! to account for the K! equivalent
            state labelings, which the mode-based factorisation covers only once.
            Set False to compare specific labelled solutions.
        verbose: Print the per-term breakdown.

    Returns:
        (log_evidence, var_log_evidence)
    """
    def n_bq_samples(d):
        return int(min(max_samples, max(min_samples, d * samples_per_dim)))

    bq_warnings = []

    def run_factor(fn, centre, d, subkey, label):
        lz, var, diag = run_bq_tinygp(
            fn, centre, dim=d, n_samples=n_bq_samples(d), key=subkey,
            tune_hyperparameters=tune_gp_hyperparameters
        )
        if diag["nonfinite_ratio_fraction"] > 0.01:
            bq_warnings.append(
                f"{label}: {diag['nonfinite_ratio_fraction']:.1%} of importance ratios "
                f"were non-finite; the Laplace approximation is a poor fit for this factor"
            )
        if diag["gp_failed"]:
            bq_warnings.append(
                f"{label}: GP quadrature was ill-conditioned even after jitter escalation; "
                f"fell back to the plain Monte Carlo estimate"
            )
        if not bool(jnp.isfinite(lz)):
            bq_warnings.append(f"{label}: log evidence is not finite")
        return lz, var

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
    
    if key is None: 
        key = random.PRNGKey(0)
    
    K = config.num_states

    if isinstance(xis, list):
         counts_trans = sum([jnp.sum(x, axis=0) for x in xis]) # (K, K)
    else:
         counts_trans = jnp.sum(xis, axis=(0, 1))
         
    log_Z_trans = 0.0
    var_log_Z_trans = 0.0
    
    prior_var_trans = config.prior_var_bias_trans
    prior_var_emit = config.prior_var_bias_emit
    
    logit_trans_evidence = 0.0
    var_logit_trans_evidence = 0.0
    
    if data.x_trans is not None:
         logit_theta_full = params.theta # (D, K, K)

         if isinstance(xis, list):
             x_trans_inputs = jnp.concatenate([x for x in data.x_trans], axis=0) # (N, D)
             xis_flat = jnp.concatenate(xis, axis=0) # (N, K, K)
         else:
             x_trans_inputs = data.x_trans.reshape(-1, data.x_trans.shape[-1])
             xis_flat = xis.reshape(-1, K, K)
             
         D_trans = x_trans_inputs.shape[1]
         
         for j in range(K):
             key, subkey = random.split(key)
             center_j_full = logit_theta_full[:, j, :] # (D, K)
             center_j_reduced = center_j_full[:, :-1] - center_j_full[:, -1:]  # (D, K-1)
             center_j = center_j_reduced.flatten()     # (D * (K-1),)
             targets_j = xis_flat[:, j, :] # (N, K)
             
             def lp_theta(theta_flat):
                 Theta_reduced = theta_flat.reshape(D_trans, K-1)
                 zeros = jnp.zeros((D_trans, 1))
                 Theta_j = jnp.concatenate([Theta_reduced, zeros], axis=1) # (D, K)
                 
                 logits = x_trans_inputs @ Theta_j
                 log_probs = jax.nn.log_softmax(logits)
                 ll = jnp.sum(targets_j * log_probs)
                 
                 dim_t = theta_flat.shape[0]
                 indices = jnp.arange(dim_t)
                 row_indices = indices // (K-1)

                 is_bias = (row_indices == 0)

                 reg_prec = config.l2_reg_trans
                 log_prob_bias = -0.5 * (theta_flat**2) / prior_var_trans - 0.5 * jnp.log(2 * jnp.pi * prior_var_trans)
                 log_prob_reg = -0.5 * reg_prec * (theta_flat**2) + 0.5 * (jnp.log(reg_prec) - jnp.log(2 * jnp.pi))
                 log_prior_elements = jnp.where(is_bias, log_prob_bias, log_prob_reg)
                 
                 return ll + jnp.sum(log_prior_elements)
                 
             lz, var_lz = run_factor(lp_theta, center_j, D_trans*(K-1), subkey, f"transitions[{j}]")
             logit_trans_evidence += lz
             var_logit_trans_evidence += var_lz
         
         # Assign the computed input-driven evidence to log_Z_trans
         log_Z_trans = logit_trans_evidence
         var_log_Z_trans = var_logit_trans_evidence
              
    else:
        for j in range(K):
            alpha_prior = static_trans_alpha * jnp.ones(K)  # Configurable Dirichlet concentration
            if config.transition_stickiness > 0.0:
                 alpha_prior = alpha_prior.at[j].add(config.transition_stickiness)
            
            row_counts = counts_trans[j]
            term1 = jax.scipy.special.gammaln(jnp.sum(alpha_prior)) - jax.scipy.special.gammaln(jnp.sum(alpha_prior + row_counts))
            term2 = jnp.sum(jax.scipy.special.gammaln(alpha_prior + row_counts) - jax.scipy.special.gammaln(alpha_prior))
            
            log_Z_trans += term1 + term2
        
    log_Z_glm = 0.0
    var_log_Z_glm = 0.0
    
    gammas_flat = flatten_list(gammas)
    x_b_flat = flatten_list(data.x_bern)
    y_b_flat = flatten_list(data.y_bern)
    x_g_flat = flatten_list(data.x_gauss)
    y_g_flat = flatten_list(data.y_gauss)
    
    valid_bern = (params.W_bern is not None) and (x_b_flat is not None)
    valid_gauss = (params.W_gauss is not None) and (x_g_flat is not None)

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

    # The Gaussian factor is conjugate, so it is integrated in closed form over
    # both W and Sigma rather than by quadrature. The matrix-normal prior used
    # here (column covariance Sigma, row precisions Lambda) is exactly the prior
    # the closed-form Gaussian M-step optimises against.
    if valid_gauss:
        M_gauss = y_g_flat.shape[1]
        iw_scale, iw_dof = resolve_iw_prior(config, M_gauss)
        D_gauss = x_g_flat.shape[1]
        row_precisions = jnp.concatenate([
            jnp.array([1.0 / config.prior_var_bias_emit]),
            jnp.full(D_gauss - 1, config.l2_reg_gauss)
        ])
        seg_ids_g = gauss_segments[2] if gauss_segments is not None else None

    for k in range(K):
        key, subkey = random.split(key)

        if valid_gauss:
            w_samples = gaussian_sample_weights(
                gammas_flat, k, M_gauss, seg_ids_g, gauss_seg_lengths, do_normalize_gauss
            )
            log_Z_glm += gaussian_factor_log_evidence(
                x_g_flat, y_g_flat, w_samples, row_precisions, iw_scale, iw_dof
            )
            # exact: contributes no Monte Carlo variance

        if valid_bern:
            center = params.W_bern[:, k]
            gs = gammas_flat[:, k]
            
            def lp_bern(w):
                logits = x_b_flat @ w
                ll_point = y_b_flat * logits - jnp.logaddexp(0.0, logits)
                
                D_w = w.shape[0]
                reg = config.l2_reg_bern
                log_prior_bias = -0.5 * (w[0]**2) / prior_var_emit - 0.5 * jnp.log(2 * jnp.pi * prior_var_emit)
                
                D_feat = D_w - 1
                if D_feat > 0:
                    log_prior_const_feat = 0.5 * D_feat * (jnp.log(reg) - jnp.log(2 * jnp.pi))
                    log_prior_feat = -0.5 * reg * jnp.sum(w[1:]**2)
                    log_prior_val = log_prior_bias + log_prior_feat + log_prior_const_feat
                else:
                    log_prior_val = log_prior_bias
                
                return jnp.sum(gs * ll_point) + log_prior_val
            
            lz, var_lz = run_factor(lp_bern, center, center.shape[0], subkey, f"bernoulli[{k}]")
            log_Z_glm += lz
            var_log_Z_glm += var_lz

        if (params.W_multinom is not None) and (data.x_multinom is not None):
             x_m_flat = flatten_list(data.x_multinom)
             y_m_flat = flatten_list(data.y_multinom)
             
             center_full = params.W_multinom[:, :, k]
             D_mn, C_mn = center_full.shape
             
             center_reduced = center_full[:, :-1] - center_full[:, -1:]
             w_flat_center = center_reduced.reshape(-1)   
             gs = gammas_flat[:, k]
             y_m_onehot = jax.nn.one_hot(y_m_flat, C_mn)
             
             def lp_multinom(w_flat):
                 W_reduced = w_flat.reshape(D_mn, C_mn - 1)
                 zeros = jnp.zeros((D_mn, 1))
                 W_full = jnp.concatenate([W_reduced, zeros], axis=1)
                 
                 logits = x_m_flat @ W_full
                 log_probs = jax.nn.log_softmax(logits)
                 ll_point = jnp.sum(y_m_onehot * log_probs, axis=1)
                 
                 D_w = w_flat.shape[0] # D * (C-1)
                 num_classes_reduced = C_mn - 1
                 indices = jnp.arange(D_w)
                 row_indices = indices // num_classes_reduced
                 is_bias = (row_indices == 0)
                 
                 reg = config.l2_reg_multinom
                 log_prob_bias = -0.5 * (w_flat**2) / prior_var_emit - 0.5 * jnp.log(2 * jnp.pi * prior_var_emit)
                 log_prob_reg = -0.5 * reg * (w_flat**2) + 0.5 * (jnp.log(reg) - jnp.log(2 * jnp.pi))
                 log_prior_elements = jnp.where(is_bias, log_prob_bias, log_prob_reg)
                 return jnp.sum(gs * ll_point) + jnp.sum(log_prior_elements)

             lz, var_lz = run_factor(lp_multinom, w_flat_center, D_mn*(C_mn-1), subkey, f"multinomial[{k}]")
             log_Z_glm += lz
             var_log_Z_glm += var_lz

        if (params.W_negbin is not None) and (data.x_negbin is not None):
             x_nb_flat = flatten_list(data.x_negbin)
             y_nb_flat = flatten_list(data.y_negbin)
             
             w_center = params.W_negbin[:, k]
             phi_center = params.phi_negbin[k]

             center_combined = jnp.concatenate([w_center, jnp.log(jnp.array([phi_center]))])
             D_nb = w_center.shape[0]
             gs = gammas_flat[:, k]
             
             def lp_negbin(theta):
                 w = theta[:-1]
                 log_phi = theta[-1]
                 phi = jnp.exp(log_phi)
                 mu = jnp.exp(x_nb_flat @ w)
                 
                 t1 = jax.scipy.special.gammaln(y_nb_flat + phi)
                 t2 = -jax.scipy.special.gammaln(y_nb_flat + 1.0)
                 t3 = -jax.scipy.special.gammaln(phi)
                 t4 = y_nb_flat * jnp.log(mu + 1e-10)
                 t5 = phi * log_phi
                 t6 = -(y_nb_flat + phi) * jnp.log(mu + phi + 1e-10)
                 
                 ll_point = t1 + t2 + t3 + t4 + t5 + t6
                 
                 reg = config.l2_reg_negbin
                 D_w = w.shape[0]
                 log_prior_bias = -0.5 * (w[0]**2) / prior_var_emit - 0.5 * jnp.log(2 * jnp.pi * prior_var_emit)
                 
                 D_feat = D_w - 1
                 if D_feat > 0:
                     log_prior_const_feat = 0.5 * D_feat * (jnp.log(reg) - jnp.log(2 * jnp.pi))
                     log_prior_feat = -0.5 * reg * jnp.sum(w[1:]**2)
                     log_prior_w = log_prior_bias + log_prior_feat + log_prior_const_feat
                 else:
                     log_prior_w = log_prior_bias

                 log_prior_phi = -0.5 * (log_phi**2) - 0.5 * jnp.log(2 * jnp.pi)
                 
                 return jnp.sum(gs * ll_point) + log_prior_w + log_prior_phi
                 
             lz, var_lz = run_factor(lp_negbin, center_combined, D_nb+1, subkey, f"negbin[{k}]")
             log_Z_glm += lz
             var_log_Z_glm += var_lz
             
        if (params.W_gamma is not None) and (data.x_gamma is not None):
             x_gm_flat = flatten_list(data.x_gamma)
             y_gm_flat = flatten_list(data.y_gamma)
             
             w_center = params.W_gamma[:, k]
             alpha_center = params.alpha_gamma[k]
             
             center_combined = jnp.concatenate([w_center, jnp.log(jnp.array([alpha_center]))])
             D_gm = w_center.shape[0]
             gs = gammas_flat[:, k]
             
             def lp_gamma(theta):
                 w = theta[:-1]
                 log_alpha = theta[-1]
                 alpha = jnp.exp(log_alpha)
                 
                 mu = jnp.exp(x_gm_flat @ w)
                 log_beta = log_alpha - jnp.log(mu + 1e-10)
                 beta = alpha / (mu + 1e-10)
                 
                 t1 = alpha * log_beta
                 t2 = -jax.scipy.special.gammaln(alpha)
                 t3 = (alpha - 1.0) * jnp.log(y_gm_flat + 1e-10)
                 t4 = -beta * y_gm_flat
                 
                 ll_point = t1 + t2 + t3 + t4
                 
                 reg = config.l2_reg_gamma
                 D_w = w.shape[0]                 
                 log_prior_bias = -0.5 * (w[0]**2) / prior_var_emit - 0.5 * jnp.log(2 * jnp.pi * prior_var_emit)
                 
                 D_feat = D_w - 1
                 if D_feat > 0:
                     log_prior_const_feat = 0.5 * D_feat * (jnp.log(reg) - jnp.log(2 * jnp.pi))
                     log_prior_feat = -0.5 * reg * jnp.sum(w[1:]**2)
                     log_prior_w = log_prior_bias + log_prior_feat + log_prior_const_feat
                 else:
                     log_prior_w = log_prior_bias

                 log_prior_alpha = -0.5 * (log_alpha**2) - 0.5 * jnp.log(2 * jnp.pi)
                 
                 return jnp.sum(gs * ll_point) + log_prior_w + log_prior_alpha

             lz, var_lz = run_factor(lp_gamma, center_combined, D_gm+1, subkey, f"gamma[{k}]")
             log_Z_glm += lz
             var_log_Z_glm += var_lz


    log_P_Y = compute_log_marginal_likelihood(params, data, do_normalize_gauss)

    if isinstance(gammas, list):
         expected_ll_init = sum([jnp.sum(g[0] * params.log_pi0) for g in gammas])
    else:
         expected_ll_init = jnp.sum(gammas[:, 0, :] * params.log_pi0)
         
    if data.x_trans is not None and params.theta is not None:
         expected_ll_trans = 0.0
         if isinstance(xis, list):
             for i, x_seq in enumerate(data.x_trans):
                 x_in = x_seq # (T-1, D)
                 logits_t = jnp.tensordot(x_in, params.theta, axes=(1, 0))
                 log_A_t = jax.nn.log_softmax(logits_t, axis=-1)
                 expected_ll_trans += jnp.sum(xis[i] * log_A_t)
         else:
             x_in = data.x_trans
             logits_t = jnp.tensordot(x_in, params.theta, axes=(2, 0)) # (N, T-1, D) @ (D, K, K) -> (N, T-1, K, K)
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
         
    if (params.W_multinom is not None) and (data.x_multinom is not None):
        ll_unweighted = multinomial_glmhmm_loglikelihood(flatten_list(data.x_multinom), flatten_list(data.y_multinom), params.W_multinom)
        expected_ll_emit += jnp.sum(gammas_flat * ll_unweighted)
        
    if (params.W_negbin is not None) and (data.x_negbin is not None):
        ll_nb = negbin_glmhmm_loglikelihood(flatten_list(data.x_negbin), flatten_list(data.y_negbin), params.W_negbin, params.phi_negbin)
        expected_ll_emit += jnp.sum(gammas_flat * ll_nb)
        
    if (params.W_gamma is not None) and (data.x_gamma is not None):
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
             n_sequences = 1
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

    total_log_Z = log_Z_init + log_Z_trans + log_Z_glm + entropy + log_multiplicity
    total_var = var_log_Z_trans + var_log_Z_glm

    for msg in bq_warnings:
        warnings.warn(f"[BQ K={K}] {msg}", RuntimeWarning)

    if verbose:
        print(f"  [BQ K={K}] log_Z_init:  {float(log_Z_init):.2f}")
        print(f"  [BQ K={K}] log_Z_trans: {float(log_Z_trans):.2f}")
        print(f"  [BQ K={K}] log_Z_glm:   {float(log_Z_glm):.2f}")
        print(f"  [BQ K={K}] entropy:     {float(entropy):.2f}")
        if include_label_switching:
            print(f"  [BQ K={K}] log K!:      {float(log_multiplicity):.2f}")
        print(f"  [BQ K={K}] total:       {float(total_log_Z):.2f}  (sd {float(jnp.sqrt(total_var)):.2f})")

    return total_log_Z, total_var



