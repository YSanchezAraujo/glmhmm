import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import jit
from typing import Tuple
from jax import lax

@jit
def log_normalize(log_prob: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalizes log probabilities
    
    Args:
        log_prob: A vector of log probabilities
    
    Returns:
        A tuple containing the normalized log probabilities and the log of the normalization constant.
    """
    log_c = logsumexp(log_prob)
    return log_prob - log_c, log_c

@jit
def compute_log_forward_message(
    log_lik_obs: jnp.ndarray,
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Computes the forward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_pi0: Log initial state probabilities, shape (n_states,).
        log_A: Log transition matrix, shape (n_states, n_states).
    
    Returns:
        A tuple containing log_alpha, and the log normalizers.
    """
    n_steps, _ = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_alpha,  = carry
        log_alpha_step, log_c_step = log_normalize(
            log_lik_obs[step, :] + logsumexp(log_A + prev_log_alpha[:, jnp.newaxis], axis=0)
            )
        return (log_alpha_step, ), (log_alpha_step, log_c_step)

    initial_log_alpha, initial_log_c = log_normalize(log_lik_obs[0, :] + log_pi0)
    initial_carry = (initial_log_alpha, )

    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(1, n_steps))

    log_alpha, log_c = scan_output
    log_alpha = jnp.vstack([initial_log_alpha, log_alpha])
    log_c = jnp.hstack([initial_log_c, log_c])

    return log_alpha, log_c

@jit
def compute_log_backward_message(
    log_lik_obs: jnp.ndarray, 
    log_A: jnp.ndarray, 
    log_c: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the backward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_A: Log transition matrix, shape (n_states, n_states).
        log_c: Log normalization constants from forward messages, shape (n_steps,).
    
    Returns:
        Log beta messages.
    """
    n_steps, n_states = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_beta, = carry
        log_beta_sum = prev_log_beta + log_lik_obs[step+1, :]
        log_beta_step = logsumexp(log_A.T + log_beta_sum[:, jnp.newaxis], axis=0) - log_c[step+1]
        return (log_beta_step, ), log_beta_step

    initial_log_beta = jnp.zeros(n_states) 
    initial_carry = (initial_log_beta, )
    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(n_steps-2, -1, -1))
    log_beta = jnp.vstack([jnp.flip(scan_output, axis=0), initial_log_beta])

    return log_beta

@jit
def compute_expectations(
    log_alpha: jnp.ndarray,
    log_beta: jnp.ndarray,
    log_c: jnp.ndarray,
    log_lik_obs: jnp.ndarray,
    log_A: jnp.ndarray   
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the expectations (xi and gamma) for the Hidden Markov Model.

    Args:
        log_alpha: Log scaled forward messages from compute_log_forward_message.
        log_beta: Log scaled backward messages from compute_log_backward_message.
        log_c: Log normalization constants from forward messages.
        log_lik_obs: Log likelihoods of observations.
        log_A: Log transition matrix.

    Returns:
        A tuple containing:
            - 'xi_summed': Expected transitions sum_t xi_t(i,j),
                           shape (n_states, n_states). If not transposed, xi_summed[i,j]
                           is the expected number of transitions from state i to state j.
            - 'gamma': Expected states P(z_t=i | O), shape (n_steps, n_states).
    """
    n_steps, _ = log_lik_obs.shape

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)
    
    def compute_xi_step(step: int):
        log_b_lik = (log_lik_obs[step + 1, :] + log_beta[step + 1, :])[jnp.newaxis, :]
        log_xi_step_ij = log_alpha[step, :][:, jnp.newaxis] + log_A + log_b_lik - log_c[step + 1]
        return jnp.exp(log_xi_step_ij)

    xi_over_time = jax.vmap(compute_xi_step)(jnp.arange(n_steps - 1))
    xi_summed = jnp.sum(xi_over_time, axis=0)

    return xi_summed, gamma
