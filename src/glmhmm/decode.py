"""
Decoding algorithms for GLM-HMM.

This module provides algorithms for finding the most likely state sequences
given fitted parameters and observations.

Available functions:
- viterbi: Decode a single sequence
- viterbi_all_sequences: Decode all sequences in a dataset
- viterbi_core: Low-level JIT-compiled Viterbi algorithm
"""
import jax.numpy as jnp
from jax import jit, lax
from typing import List, Tuple

from .structs import GLMHMMParams, GLMHMMData
from .fit import compute_total_emissions
from .utils import compute_A_from_theta_and_inputs


@jit
def viterbi_core(
    log_lik_obs: jnp.ndarray,  # (T, K)
    log_pi0: jnp.ndarray,      # (K,)
    log_A: jnp.ndarray         # (T-1, K, K) or (K, K)
) -> Tuple[jnp.ndarray, float]:
    """
    JIT-compiled Viterbi algorithm using lax.scan.
    
    This is the low-level implementation that operates on pre-computed
    log-likelihoods and transition matrices.
    
    Args:
        log_lik_obs: Log-likelihoods of observations, shape (T, K)
        log_pi0: Initial state log-probabilities, shape (K,)
        log_A: Transition log-probs, shape (T-1, K, K) for input-driven 
               or (K, K) for static transitions
    
    Returns:
        best_path: Most likely state sequence, shape (T,)
        max_log_prob: Log-probability of the best path
    """
    T, K = log_lik_obs.shape
    
    if log_A.ndim == 2:
        log_A = jnp.broadcast_to(log_A, (T - 1, K, K))

    def forward_step(delta_prev, inputs):
        log_A_t, ll_t = inputs
        trans_scores = delta_prev[:, None] + log_A_t
        best_prev = jnp.argmax(trans_scores, axis=0)  # (K,)
        max_scores = jnp.max(trans_scores, axis=0)    # (K,)
        delta_curr = max_scores + ll_t
        return delta_curr, best_prev

    delta_0 = log_pi0 + log_lik_obs[0]
    scan_inputs = (log_A, log_lik_obs[1:])
    final_delta, backpointers = lax.scan(forward_step, delta_0, scan_inputs)
    best_last_state = jnp.argmax(final_delta)
    max_prob = jnp.max(final_delta)

    def backward_step(next_state, psi_t):
        curr_state = psi_t[next_state]
        return curr_state, curr_state

    _, path_rev = lax.scan(backward_step, best_last_state, backpointers, reverse=True)
    best_path = jnp.concatenate([path_rev, best_last_state[None]])
    return best_path, max_prob


def viterbi(
    params: GLMHMMParams, 
    data: GLMHMMData, 
    sequence_idx: int = 0
) -> jnp.ndarray:
    """
    Viterbi decoding: finds the most likely state sequence for a specific session.
    
    Handles all emission types (Bernoulli, Gaussian, Multinomial, NegBin, Gamma)
    automatically by using compute_total_emissions.
    
    Args:
        params: Fitted GLMHMMParams
        data: GLMHMMData containing observations
        sequence_idx: Which sequence to decode (default 0)
    
    Returns:
        best_path: Most likely state sequence, shape (T,) of integers
    
    Example:
        >>> params, _ = fit_glmhmm(data, config)
        >>> most_likely_states = viterbi(params, data, sequence_idx=0)
    """
    i = sequence_idx
    
    x_b  = data.x_bern[i] if data.x_bern is not None else None
    y_b  = data.y_bern[i] if data.y_bern is not None else None
    x_mn = data.x_multinom[i] if data.x_multinom is not None else None
    y_mn = data.y_multinom[i] if data.y_multinom is not None else None
    x_nb = data.x_negbin[i] if data.x_negbin is not None else None
    y_nb = data.y_negbin[i] if data.y_negbin is not None else None
    x_gm = data.x_gamma[i] if data.x_gamma is not None else None
    y_gm = data.y_gamma[i] if data.y_gamma is not None else None
    
    x_gs, y_gs = None, None
    gs_first, gs_last, gs_segment_ids = None, None, None
    
    if data.x_gauss is not None and isinstance(data.x_gauss, list):
        if isinstance(data.x_gauss[i], (list, tuple)):
            x_gs = jnp.concatenate(data.x_gauss[i], axis=0)
            y_gs = jnp.concatenate(data.y_gauss[i], axis=0)
        else:
            x_gs = data.x_gauss[i]
            y_gs = data.y_gauss[i]
        
        if data.gauss_first_idx is not None:
            gs_first = data.gauss_first_idx[i]
            gs_last = data.gauss_last_idx[i]
            n_segs = gs_first.shape[0]
            lengths = gs_last - gs_first
            gs_segment_ids = jnp.repeat(jnp.arange(n_segs), lengths)

    ll_t = compute_total_emissions(
        params, x_b, y_b, x_mn, y_mn, x_nb, y_nb, x_gm, y_gm, 
        x_gs, y_gs, gs_first, gs_last, gs_segment_ids
    )

    if params.theta is not None and data.x_trans is not None:
        x_trans = data.x_trans[i]
        A_t = compute_A_from_theta_and_inputs(x_trans, params.theta)
        log_A_t = jnp.log(A_t + 1e-45)  
    else:
        log_A_t = params.log_A

    best_path, _ = viterbi_core(ll_t, params.log_pi0, log_A_t)
    return best_path


def viterbi_all_sequences(params: GLMHMMParams, data: GLMHMMData) -> List[jnp.ndarray]:
    """
    Run Viterbi on all sequences in the data.
    
    Args:
        params: Fitted GLMHMMParams
        data: GLMHMMData with multiple sequences
    
    Returns:
        List of state sequences, one per sequence in data
    """
    seq_sources = [data.x_trans, data.x_bern, data.x_multinom, data.x_negbin, data.x_gamma]
    
    if data.x_gauss is not None and isinstance(data.x_gauss, list) and len(data.x_gauss) > 0:
        n_sequences = len(data.x_gauss)
    else:
        n_sequences = next((len(s) for s in seq_sources if s is not None), 0)
    
    return [viterbi(params, data, i) for i in range(n_sequences)]
