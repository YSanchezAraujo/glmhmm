
import jax
import jax.numpy as jnp
from jax import random, grad, hessian, vmap, jit
import numpy as np
import tinygp
from tinygp import kernels
import optax
from typing import Callable, Tuple, Optional, Any
from functools import partial

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import compute_log_marginal_likelihood
from .emissions import (
    bernoulli_glmhmm_loglikelihood,
    multinomial_glmhmm_loglikelihood,
    negbin_glmhmm_loglikelihood,
    gamma_glmhmm_loglikelihood,
    gaussian_glmhmm_loglikelihoods
)

@jit
def compute_kernel_expectations(
    kernel_variance: float,
    lengthscale: jnp.ndarray, # (D,) or scalar
    X_train: jnp.ndarray,     # (N, D)
    q_mean: jnp.ndarray,      # (D,)
    q_cov: jnp.ndarray        # (D, D)
) -> jnp.ndarray:
    """
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
    """
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


class BQResult(Any):
    log_evidence: float
    bq_correction: float
    laplace_evidence: float

def run_bq_tinygp(
    log_prob_fn: Callable[[jnp.ndarray], float],
    center: jnp.ndarray,
    dim: int,
    n_samples: int = 150,
    key: Optional[random.PRNGKey] = None,
    lengthscale_init: float = 1.0
) -> float:
    
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
    ratios = jnp.nan_to_num(ratios, nan=0.0, posinf=1e6, neginf=0.0)
    
    mean_ratio = jnp.mean(ratios)
    y_train = ratios - mean_ratio
    z_samples = jax.scipy.linalg.solve_triangular(L_chol, (samples - center).T, lower=True).T
    
    kernel = kernels.ExpSquared(scale=lengthscale_init)
    K_mat = kernel(z_samples, z_samples) + 1e-5 * jnp.eye(n_samples)
    L_K = jnp.linalg.cholesky(K_mat)
    alpha = jax.scipy.linalg.solve_triangular(L_K.T, jax.scipy.linalg.solve_triangular(L_K, y_train, lower=True), lower=False)

    z_vec = compute_kernel_expectations(
        kernel_variance=1.0,
        lengthscale=jnp.array([lengthscale_init]),
        X_train=z_samples,
        q_mean=jnp.zeros(dim),
        q_cov=jnp.eye(dim)
    )
    
    integral_correction_z = jnp.dot(z_vec, alpha)
    bq_integral_scaled = integral_correction_z + mean_ratio
    
    prior_var_integral = compute_prior_integral_variance(
        kernel_variance=1.0,
        lengthscale=jnp.array([lengthscale_init]),
        q_cov=jnp.eye(dim),
        dim=dim
    )
    
    K_inv_z = jax.scipy.linalg.solve_triangular(L_K.T, jax.scipy.linalg.solve_triangular(L_K, z_vec, lower=True), lower=False)
    var_reduction = jnp.dot(z_vec, K_inv_z)
    posterior_var_dist = prior_var_integral - var_reduction
    posterior_var_dist = jnp.maximum(posterior_var_dist, 0.0)
    bq_val_safe = jnp.maximum(bq_integral_scaled, 1e-4)
    var_log_Z = posterior_var_dist / (bq_val_safe**2)
    
    return log_z_laplace + jnp.log(bq_val_safe), var_log_Z

def compute_factorized_evidence(
    data: GLMHMMData,
    params: GLMHMMParams,
    config: GLMHMMConfig,
    gammas: jnp.ndarray, # (N_seq, T, K)
    xis: jnp.ndarray,    # (N_seq, T-1, K, K)
    key: Optional[random.PRNGKey] = None
) -> Tuple[float, float]:
    """
    Computes total log evidence using factorization:
    log Z = log Z_trans + sum_k log Z_glm_k
    Returns: (log_evidence, var_log_evidence)
    """
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
    min_samples = 2000
    samples_per_dim = 500
    
    if isinstance(xis, list):
         counts_trans = sum([jnp.sum(x, axis=0) for x in xis]) # (K, K)
    else:
         counts_trans = jnp.sum(xis, axis=(0, 1))
         
    log_Z_trans = 0.0
    var_log_Z_trans = 0.0
    
    prior_var_trans = config.prior_var_bias_trans
    prior_var_emit = config.prior_var_bias_emit
    
    def make_log_prob_trans_row(row_counts, state_idx, prior_dim=None, row_prior_alpha=1.1):
        def _lp(logit_row):
            log_probs = jax.nn.log_softmax(logit_row)
            ll = jnp.dot(row_counts, log_probs)
            
            D_actual = logit_row.shape[0]
            D_prior = prior_dim if prior_dim is not None else D_actual
            weak_prior_prec = 1.0 / prior_var_trans
            log_prior_const = 0.5 * D_prior * (jnp.log(weak_prior_prec) - jnp.log(2 * jnp.pi))
            prior_l2 = -0.5 * weak_prior_prec * jnp.sum(logit_row**2)
            return ll + prior_l2 + log_prior_const
        return _lp

    logit_trans_evidence = 0.0
    var_logit_trans_evidence = 0.0
    
    if data.x_trans is not None:
         logit_theta_full = params.theta # (D, K, K)
         
         x_trans_flat = flatten_list(data.x_trans)
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
                 prior = jnp.where(is_bias, 
                                   -0.5 * config.transition_stickiness * (theta_flat**2), 
                                   -0.5 * reg_prec * (theta_flat**2)
                                  )
                 
                 log_prob_bias = -0.5 * (theta_flat**2) / prior_var_trans - 0.5 * jnp.log(2 * jnp.pi * prior_var_trans)
                 log_prob_reg = -0.5 * reg_prec * (theta_flat**2) + 0.5 * (jnp.log(reg_prec) - jnp.log(2 * jnp.pi))
                 log_prior_elements = jnp.where(is_bias, log_prob_bias, log_prob_reg)
                 
                 return ll + jnp.sum(log_prior_elements)
                 
             current_dim = D_trans*(K-1)
             n_dynamic = max(min_samples, current_dim * samples_per_dim)
             
             lz, var_lz = run_bq_tinygp(lp_theta, center_j, dim=current_dim, n_samples=n_dynamic, key=subkey)
             logit_trans_evidence += lz
             var_logit_trans_evidence += var_lz
             
    else:
        for j in range(K):
            alpha_prior = jnp.ones(K)
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

    for k in range(K):
        key, subkey = random.split(key)
        
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
            
            current_dim = center.shape[0]
            n_dynamic = max(min_samples, current_dim * samples_per_dim)
            
            lz, var_lz = run_bq_tinygp(lp_bern, center, dim=current_dim, n_samples=n_dynamic, key=subkey)
            log_Z_glm += lz
            var_log_Z_glm += var_lz
            
        if valid_gauss:
            W_center = params.W_gauss[:, :, k]
            D_in, M_out = W_center.shape
            w_flat_center = W_center.reshape(-1)
            
            Sigma_k = params.Sigma_gauss[:, :, k]
            Sigma_safe = Sigma_k + 1e-3 * jnp.eye(M_out)
            Prec_k = jnp.linalg.inv(Sigma_safe)
            
            if data.gauss_first_idx is not None:
                first_flat = flatten_list(data.gauss_first_idx)
                last_flat = flatten_list(data.gauss_last_idx)
                counts = last_flat - first_flat
                seg_ids = jnp.repeat(jnp.arange(len(counts)), counts)
                gs_expanded = gammas_flat[seg_ids, k]
            else:
                 gs_expanded = gammas_flat[:, k]

            def lp_gauss(w_flat):
                W = w_flat.reshape(D_in, M_out)
                mu = x_g_flat @ W 
                diff = y_g_flat - mu
                quad = jnp.sum((diff @ Prec_k) * diff, axis=1)
                ll_point = -0.5 * quad - 0.5 * M_out * jnp.log(2*jnp.pi) + 0.5 * jnp.linalg.slogdet(Prec_k)[1]
                D_w = w_flat.shape[0]       
                indices = jnp.arange(D_w)
                row_indices = indices // M_out
                is_bias = (row_indices == 0)
                reg = config.l2_reg_gauss
                log_prob_bias = -0.5 * (w_flat**2) / prior_var_emit - 0.5 * jnp.log(2 * jnp.pi * prior_var_emit)
                log_prob_reg = -0.5 * reg * (w_flat**2) + 0.5 * (jnp.log(reg) - jnp.log(2 * jnp.pi))
                log_prior_elements = jnp.where(is_bias, log_prob_bias, log_prob_reg)
                return jnp.sum(gs_expanded * ll_point) + jnp.sum(log_prior_elements)

            current_dim = D_in*M_out
            n_dynamic = max(min_samples, current_dim * samples_per_dim)
            
            lz, var_lz = run_bq_tinygp(lp_gauss, w_flat_center, dim=current_dim, n_samples=n_dynamic, key=subkey)
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

             current_dim = D_mn*(C_mn-1)
             n_dynamic = max(min_samples, current_dim * samples_per_dim)
             
             lz, var_lz = run_bq_tinygp(lp_multinom, w_flat_center, dim=current_dim, n_samples=n_dynamic, key=subkey)
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

                 log_prior_phi = -0.5 * 1.0 * (log_phi**2)
                 
                 return jnp.sum(gs * ll_point) + log_prior_w + log_prior_phi
                 
             current_dim = D_nb+1
             n_dynamic = max(min_samples, current_dim * samples_per_dim)
             
             lz, var_lz = run_bq_tinygp(lp_negbin, center_combined, dim=current_dim, n_samples=n_dynamic, key=subkey)
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

                 log_prior_alpha = -0.5 * 1.0 * (log_alpha**2)
                 
                 return jnp.sum(gs * ll_point) + log_prior_w + log_prior_alpha

             current_dim = D_gm+1
             n_dynamic = max(min_samples, current_dim * samples_per_dim)
             
             lz, var_lz = run_bq_tinygp(lp_gamma, center_combined, dim=current_dim, n_samples=n_dynamic, key=subkey)
             log_Z_glm += lz
             var_log_Z_glm += var_lz


    log_P_Y = compute_log_marginal_likelihood(params, data)
    
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

    alpha_prior = 1.0
    term1 = jnp.sum(jax.scipy.special.gammaln(alpha_prior + init_counts))
    term2 = jax.scipy.special.gammaln(jnp.sum(alpha_prior + init_counts))
    term3 = jnp.sum(jax.scipy.special.gammaln(alpha_prior))
    term4 = jax.scipy.special.gammaln(jnp.sum(jnp.array([alpha_prior] * K)))
    
    log_Z_init = (term1 - term2) - (term3 - term4)

    total_log_Z = log_Z_init + log_Z_trans + log_Z_glm + entropy
    total_var = var_log_Z_trans + var_log_Z_glm
    
    print(f"  [BQ Debug K={K}] log_Z_init: {float(log_Z_init):.2f}")
    print(f"  [BQ Debug K={K}] log_Z_trans: {float(log_Z_trans):.2f}")
    print(f"  [BQ Debug K={K}] log_Z_glm: {float(log_Z_glm):.2f}")
    print(f"  [BQ Debug K={K}] entropy: {float(entropy):.2f}")
    print(f"  [BQ Debug K={K}] total: {float(total_log_Z):.2f}")
    
    return total_log_Z, total_var



