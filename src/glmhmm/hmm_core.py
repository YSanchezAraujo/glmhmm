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
    Computes the forward messages (alpha) for a Hidden Markov Model using the forward algorithm.
    
    The forward message alpha[t, k] represents the probability of being in state k at time t
    given all observations up to time t. This implementation uses log-space arithmetic and
    scaling for numerical stability.
    
    Args:
        log_lik_obs: Log-likelihoods of observations, shape (T, K) where T is the number
                     of time steps and K is the number of states.
        log_pi0: Log initial state probabilities, shape (K,).
        log_A: Log transition matrix, shape (T, K, K) for time-varying transitions.
               For static transitions, broadcast from (K, K) to (T, K, K).
               log_A[t, i, j] = log P(state_t+1 = j | state_t = i).
    
    Returns:
        log_alpha: Scaled log forward messages, shape (T, K).
        log_c: Log normalization constants, shape (T,). Sum of log_c gives the
               log marginal likelihood of the observations.
    """
    n_steps, _ = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_alpha,  = carry
        log_alpha_step, log_c_step = log_normalize(
            log_lik_obs[step, :] + logsumexp(log_A[step-1, :, :] + prev_log_alpha[:, jnp.newaxis], axis=0)
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
    Computes the backward messages (beta) for a Hidden Markov Model using the backward algorithm.
    
    The backward message beta[t, k] represents the probability of observing all future
    observations from time t+1 onwards, given that we are in state k at time t.
    This implementation uses the same scaling as the forward pass for numerical stability.
    
    Args:
        log_lik_obs: Log-likelihoods of observations, shape (T, K).
        log_A: Log transition matrix, shape (T, K, K) for time-varying transitions.
               log_A[t, i, j] = log P(state_t+1 = j | state_t = i).
        log_c: Log normalization constants from forward pass, shape (T,).
               Used to ensure consistent scaling between forward and backward passes.
    
    Returns:
        log_beta: Scaled log backward messages, shape (T, K).
    """
    n_steps, n_states = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_beta, = carry
        log_beta_sum = prev_log_beta + log_lik_obs[step+1, :]
        log_beta_step = logsumexp(log_A[step, :, :].T + log_beta_sum[:, jnp.newaxis], axis=0) - log_c[step+1]
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
    Computes the E-step expectations (xi and gamma) for the Hidden Markov Model.
    
    These expectations are used in the M-step to update model parameters:
    - gamma[t, k] = P(state_t = k | observations) - state responsibilities
    - xi[t, i, j] = P(state_t = i, state_t+1 = j | observations) - transition responsibilities

    Args:
        log_alpha: Scaled log forward messages, shape (T, K).
        log_beta: Scaled log backward messages, shape (T, K).
        log_c: Log normalization constants from forward pass, shape (T,).
        log_lik_obs: Log-likelihoods of observations, shape (T, K).
        log_A: Log transition matrix, shape (T, K, K) for time-varying transitions.

    Returns:
        xi: Expected transition counts, shape (T-1, K, K).
            xi[t, i, j] = P(state_t = i, state_t+1 = j | all observations).
            Sum over time gives total expected transitions from state i to state j.
        gamma: Expected state occupancies, shape (T, K).
               gamma[t, k] = P(state_t = k | all observations).
    """
    n_steps, _ = log_lik_obs.shape

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)
    
    def compute_xi_step(step: int):
        log_b_lik = (log_lik_obs[step + 1, :] + log_beta[step + 1, :])[jnp.newaxis, :]
        log_xi_step_ij = log_alpha[step, :][:, jnp.newaxis] + log_A[step, :, :] + log_b_lik - log_c[step + 1]
        return jnp.exp(log_xi_step_ij)

    xi_over_time = jax.vmap(compute_xi_step)(jnp.arange(n_steps - 1))
    return xi_over_time, gamma
