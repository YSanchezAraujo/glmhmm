import jax.numpy as jnp
from jax import jit, nn
from jax.scipy.stats import multivariate_normal as jax_mvnormal
from jax.ops import segment_sum 

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
    n_regions
):
    """
    Computes normalized loglikelihoods of gaussian segements

    Args:
        all_likelihoods: 1-d array of loglikelihoods
        start_idx_set: 1-d array of integers for when segements start
        end_idx_set: 1-d array of integers for when segements end
        n_regions: number of regions

    Returns:
        Normalized loglikelihoods of gaussian segements, 1-d array
    """
    n_segments = len(last_idx_set)
    segment_lens = last_idx_set - first_idx_set
    segment_ids = jnp.repeat(jnp.arange(n_segments), segment_lens)
    segment_sums = segment_sum(all_loglikelihoods, segment_ids, num_segments = n_segments)
    norm_ll = segment_sums / segment_lens
    return norm_ll / n_regions