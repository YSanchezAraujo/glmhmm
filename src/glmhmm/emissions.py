import jax.numpy as jnp
from jax import jit, nn
from jax.scipy.stats import multivariate_normal as jax_mvnormal
from jax.ops import segment_sum 
from jax.scipy.special import gammaln


@jit
def bernoulli_glmhmm_loglikelihood(
    X: jnp.ndarray,    
    y: jnp.ndarray,
    W: jnp.ndarray     
) -> jnp.ndarray:
    """
    Computes the log likelihood of the data under the parameters W

    Args:
        X: Design matrix, shape (n_trials, n_features)
        y: Response data (binary 0 or 1), shape (n_trials,)
        W: Weight matrix, shape (n_features, n_states)

    Returns:
        Log likelihood of the data, shape (n_trials, n_states)
    """
    
    if W.ndim == 1:
        _W = W[:, jnp.newaxis]
    elif W.ndim == 2:
        _W = W
    else:
        raise ValueError(f"W must be 1D or 2D, but got {W.ndim} dimensions.")

    logits_per_state = X @ _W
    log_prob_y_eq_1 = nn.log_sigmoid(logits_per_state)
    log_prob_y_eq_0 = nn.log_sigmoid(-logits_per_state)
    y_col = y.astype(jnp.float32)[:, jnp.newaxis]
    loglik_obs_bern= y_col * log_prob_y_eq_1 + (1 - y_col) * log_prob_y_eq_0
    return loglik_obs_bern

def multinomial_glmhmm_loglikelihood(
    X: jnp.ndarray,     
    y: jnp.ndarray,     
    W: jnp.ndarray      # (F, C) or (F, C, K) 
) -> jnp.ndarray:
    """
    Computes Log-Likelihood for Multinomial GLM. 
    Returns (N, K) if W is (F, C, K), or (N,) if W is (F, C).
    """
    if W.ndim == 3:
        # W shape: (F, C, K)
        # X shape: (N, F)
        # logits: (N, C, K)
        logits = jnp.einsum('nf,fck->nck', X, W)
        log_probs = nn.log_softmax(logits, axis=1) 
        
        # y shape: (N,) indices or (N, C) one-hot
        if y.ndim == 2:
            loglik = jnp.sum(y[:, :, jnp.newaxis] * log_probs, axis=1)
        else:
            y_expanded = jnp.broadcast_to(y[:, None, None], (y.shape[0], 1, logits.shape[2]))
            loglik = jnp.take_along_axis(log_probs, y_expanded, axis=1).squeeze(axis=1) # (N, K)
            
    else:
        logits = X @ W 
        log_probs = nn.log_softmax(logits, axis=-1)
        
        if y.ndim == 2:
            loglik = jnp.sum(y * log_probs, axis=-1)
        else:
            loglik = jnp.take_along_axis(log_probs, y[:, None], axis=-1).squeeze()
        
    return loglik


@jit
def negbin_glmhmm_loglikelihood(
    X: jnp.ndarray,     # (n_trials, n_features)
    y: jnp.ndarray,     # (n_trials,) -> Integer counts
    W: jnp.ndarray,     # (n_features, n_states)
    phi: jnp.ndarray    # (n_states,) -> Dispersion parameter (r)
) -> jnp.ndarray:
    """
    Computes Log-Likelihood for Negative Binomial GLM using stable log-space ops.
    """
    logits = X @ W  # (N, K)
    log_mu = jnp.clip(logits, a_max=20.0)
    
    y_col = y[:, jnp.newaxis]
    phi_row = phi[jnp.newaxis, :]
    log_phi = jnp.log(phi_row)

    log_p = -nn.softplus(log_mu - log_phi)
    log_1_minus_p = log_mu - log_phi - nn.softplus(log_mu - log_phi)

    term1 = gammaln(y_col + phi_row) - gammaln(y_col + 1.0) - gammaln(phi_row)
    ll = term1 + (phi_row * log_p) + (y_col * log_1_minus_p)
    return ll

@jit
def gamma_glmhmm_loglikelihood(
    X: jnp.ndarray,     # (n_trials, n_features)
    y: jnp.ndarray,     # (n_trials,) -> Continuous positive values (y > 0)
    W: jnp.ndarray,     # (n_features, n_states)
    alpha: jnp.ndarray  # (n_states,) -> Shape parameter (concentration)
) -> jnp.ndarray:
    """
    Computes Log-Likelihood for Gamma GLM.
    
    Model:
        y ~ Gamma(shape=alpha, rate=beta)
        Mean mu = alpha / beta = exp(X @ W)  [Log Link]
        Therefore, rate beta = alpha / exp(X @ W)

    Args:
        X: Design matrix
        y: Response data (must be strictly positive)
        W: Weights determining the mean
        alpha: Shape parameter per state (controls dispersion/variance)

    Returns:
        Log likelihood of the data, shape (n_trials, n_states)
    """
    logits = X @ W
    mu = jnp.exp(jnp.clip(logits, a_min=-10.0, a_max=20.0)) 
    y_col = y[:, jnp.newaxis]
    alpha_row = alpha[jnp.newaxis, :]
    beta = alpha_row / mu 
    term1 = alpha_row * jnp.log(beta)
    term2 = -gammaln(alpha_row)
    term3 = (alpha_row - 1.0) * jnp.log(y_col)
    term4 = -beta * y_col
    ll = term1 + term2 + term3 + term4
    return ll

@jit
def gaussian_glmhmm_loglikelihoods(
    X_gauss: jnp.ndarray, # matrix size (N x p)
    y_gauss: jnp.ndarray, # matrix size (N x m)
    W_gauss: jnp.ndarray, # matrix size (p x m)
    Sigma: jnp.ndarray    # matrix size (m x m)
) -> jnp.ndarray:
    """
    Computes the loglikelihoods of the data under the parameters

    Args:
        X_gauss: Design matrix for fluoresence data
        y_gauss: Neural data
        W_gauss: Time varying kernels for a single state
        Sigma: Full covariance matrix for a single state

    Returns:
        Vector of log-likelihoods for a single state
    """
    mu = X_gauss @ W_gauss  # Shape (N, m)
    z_mu = jnp.zeros(mu.shape[1], dtype=mu.dtype) 
    log_likelihoods = jax_mvnormal.logpdf(y_gauss - mu, mean=z_mu, cov=Sigma) # Shape (N,)
    return log_likelihoods

def normalized_gauss_loglikelihoods(
    all_loglikelihoods,
    first_idx_set,
    last_idx_set,
    n_regions,
    segment_ids=None,
    normalize=True
):
    """
    Computes normalized (or summed) loglikelihoods of gaussian segements
    """
    n_segments = last_idx_set.shape[0]
    segment_lens = last_idx_set - first_idx_set
    
    if segment_ids is None:
        segment_ids = jnp.repeat(jnp.arange(n_segments), segment_lens)
        
    segment_sums = segment_sum(all_loglikelihoods, segment_ids, num_segments = n_segments)
    
    if normalize:
        norm_ll = segment_sums / segment_lens
        return norm_ll / n_regions
    else:
        return segment_sums