"""
M-step optimization functions for GLMHMM emission and transition parameters.

This module contains M-step functions for updating all GLMHMM parameters:
- Emission weights for Bernoulli, Multinomial, Negative Binomial, and Gamma distributions
- Dispersion parameters (phi for NegBin, alpha for Gamma)
- Transition parameters (static log_A or input-driven theta)

All M-steps use L-BFGS optimization via the generic infrastructure in m_step_utils.py.
Each distribution has:
1. A negative log-likelihood function (with L2 regularization)
2. A single-state optimizer wrapper (run_opt_*)
3. A multi-state M-step function (*_m_step_optax) that vmaps across states

Key functions:
- bern_m_step_optax: Bernoulli (binary) observations
- multinomial_obs_m_step_optax: Multinomial (categorical) observations  
- negbin_m_step_optax: Negative Binomial (count) observations
- gamma_m_step_optax: Gamma (positive continuous) observations
- transitions_m_step_optax_reduced: Input-driven transition parameters
"""
import jax
import jax.numpy as jnp
from typing import List
from functools import partial
from typing import Tuple
from jax import jit, lax, nn, vmap
import optax
import optax.tree_utils as otu
from jax.scipy.special import gammaln

from .m_step_utils import run_lbfgs_optimizer, tuple_param_error_fn

@jit
def bern_neg_loglik_with_prior(
    w_bern_state: jnp.ndarray, 
    X_bern: jnp.ndarray,  
    y_bern: jnp.ndarray, 
    gamma_state: jnp.ndarray,
    l2_reg: jnp.float64
) -> jnp.ndarray:
    """
    Computes the negative log-likelihood for a Bernoulli GLM for a single state,
    including an L2 prior on the weights.

    Args:
        w_bern_state: Weight vector for the current HMM state.
        X_bern: Design matrix for Bernoulli emissions.
        y_bern: Binary response vector (0 or 1).
        gamma_state: Responsibilities (gammas) for the current HMM state.

    Returns:
        Scalar negative log-likelihood value.
    """
    logits = X_bern @ w_bern_state
    y = y_bern.astype(jnp.float32)
    nll_data = -jnp.dot(gamma_state * y, logits) + jnp.sum(gamma_state * nn.softplus(logits))
    l2_prior = l2_reg * 0.5 * jnp.sum(w_bern_state[1:]**2)
    total_nll = nll_data + l2_prior
    return total_nll

# this is mostly from the optax documentation
# https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver
# small change to otu.tree_l2_norm
def run_opt_bern(
    init_params, 
    X_bern,
    y_bern,
    gamma_state,
    l2_reg,
    max_iter, 
    tol
):
    loss_fn = partial(bern_neg_loglik_with_prior, X_bern=X_bern, y_bern=y_bern, 
                     gamma_state=gamma_state, l2_reg=l2_reg)
    return run_lbfgs_optimizer(loss_fn, init_params, max_iter, tol)

@partial(jit, static_argnames=("tol", "num_opt_steps"))
def bern_init_opt(
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    l2_reg: jnp.float64 = 1.0,
    tol: float = 1e-2, 
    num_opt_steps: int = 200    
): 
    """
    Optimizes weights for a simple Bernoulli GLM (logistic regression)
    with L2 prior and scaled loss, starting from zero weights, using optax.adam.

    Args:
        X_bern: Design matrix for bernoulli observations.
        y_bern: Binary response vector (0 or 1).        
        learning_rate: Learning rate for the optimizer.
        num_opt_steps: Number of optimization steps.

    Returns:
    """
    gamma_state_ones = jnp.ones(X_bern.shape[0], dtype=X_bern.dtype)
    initial_w = jnp.zeros(X_bern.shape[1], dtype=X_bern.dtype)
    final_params, _ = run_opt_bern(initial_w, X_bern, y_bern, gamma_state_ones, l2_reg, num_opt_steps, tol)
    return final_params

@partial(jit, static_argnames=("tol", "num_opt_steps")) 
def bern_m_step_optax( 
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    gamma_all_states: jnp.ndarray, 
    initial_W_bern: jnp.ndarray,
    l2_reg: jnp.float64,
    tol: float = 1e-3,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    M-step for Bernoulli (binary) emission weights across all states.
    
    Optimizes logistic regression weights for each hidden state using L-BFGS.
    The weights are optimized independently for each state using the state
    responsibilities (gamma) as observation weights.
    
    Args:
        X_bern: Design matrix for Bernoulli observations, shape (N, D).
        y_bern: Binary response vector (0 or 1), shape (N,).
        gamma_all_states: State responsibilities, shape (N, K).
        initial_W_bern: Initial weight matrix, shape (D, K).
        l2_reg: L2 regularization strength (Ridge penalty).
        tol: Convergence tolerance for gradient norm.
        num_opt_steps: Maximum number of L-BFGS iterations.

    Returns:
        W_bern: Optimized weight matrix, shape (D, K).
        final_states: Optimizer states for all K states (for diagnostics).
    """
    optimizer = vmap(
        run_opt_bern, in_axes=(0, None, None, 0, None, None, None)
    )
    optimized_Ws_T, final_state = optimizer(
        initial_W_bern.T, X_bern, y_bern, gamma_all_states.T, l2_reg, num_opt_steps, tol
    )
    return optimized_Ws_T.T, final_state

@jit
def multinomial_obs_negloglik(
    w_reduced: jnp.ndarray, # (n_features, n_categories - 1)
    X: jnp.ndarray,         # (n_trials, n_features)
    y_integers: jnp.ndarray,# (n_trials,) -> Integer class indices
    gamma: jnp.ndarray,     # (n_trials,) -> State responsibilities
    l2_reg: float
):
    zeros = jnp.zeros((w_reduced.shape[0], 1), dtype=w_reduced.dtype)
    w_full = jnp.concatenate([w_reduced, zeros], axis=1)
    
    logits = X @ w_full
    per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels=y_integers)
    weighted_loss = jnp.dot(gamma, per_sample_loss)
    l2_penalty = l2_reg * 0.5 * jnp.sum(w_reduced**2)
    
    return weighted_loss + l2_penalty

def run_opt_multinomial_obs(
    init_w_reduced, 
    X,
    y_integers,
    gamma,
    l2_reg,
    max_iter, 
    tol
):
    loss_fn = partial(multinomial_obs_negloglik, X=X, y_integers=y_integers, 
                     gamma=gamma, l2_reg=l2_reg)
    return run_lbfgs_optimizer(loss_fn, init_w_reduced, max_iter, tol)


@partial(jit, static_argnames=("tol", "num_opt_steps"))
def multinomial_obs_m_step_optax(
    X: jnp.ndarray,                # (N, F)
    y: jnp.ndarray,                # (N,) Integers
    gamma_all_states: jnp.ndarray, # (N, K)
    initial_W: jnp.ndarray,        # (F, C, K) 
    l2_reg: float,
    tol: float = 1e-3,
    num_opt_steps: int = 2000
):
    """
    M-step for Multinomial (categorical) emission weights across all states.
    
    Optimizes softmax regression weights for each hidden state. Uses the
    reduced parameterization where the last category is the reference (weights = 0)
    for identifiability.
    
    Args:
        X: Design matrix, shape (N, D).
        y: Categorical response (integer labels 0 to C-1), shape (N,).
        gamma_all_states: State responsibilities, shape (N, K).
        initial_W: Initial weight tensor, shape (D, C, K).
                   Last category (C-1) should be zeros.
        l2_reg: L2 regularization strength.
        tol: Convergence tolerance.
        num_opt_steps: Maximum L-BFGS iterations.
    
    Returns:
        W: Optimized weight tensor, shape (D, C, K).
           Last category weights are constrained to zero.
        final_states: Optimizer states for diagnostics.
    """
    w_init_reduced_all = initial_W[:, :-1, :]
    
    optimizer = vmap(
        run_opt_multinomial_obs, 
        in_axes=(2, None, None, 1, None, None, None),
        out_axes=(2, 0) 
    )

    optimized_w_reduced, final_states = optimizer(
        w_init_reduced_all, 
        X, 
        y, 
        gamma_all_states, 
        l2_reg, 
        num_opt_steps, 
        tol
    )
    
    n_features = X.shape[1]
    n_states = gamma_all_states.shape[1]
    
    zeros = jnp.zeros((n_features, 1, n_states), dtype=optimized_w_reduced.dtype)
    final_W = jnp.concatenate([optimized_w_reduced, zeros], axis=1)
    
    return final_W, final_states

@jit
def negbin_neg_loglik_with_prior(
    params: Tuple[jnp.ndarray, jnp.ndarray], 
    X: jnp.ndarray,  
    y: jnp.ndarray, 
    gamma_state: jnp.ndarray,
    l2_reg: jnp.float64
) -> jnp.ndarray:
    
    w_state, log_phi_scalar = params
    phi_state = jnp.exp(log_phi_scalar) + 1e-6
    logits = X @ w_state
    mu = jnp.exp(jnp.clip(logits, a_max=20.0))
    y_float = y.astype(jnp.float32)
    term1 = gammaln(y_float + phi_state) - gammaln(y_float + 1.0) - gammaln(phi_state)
    log_p = jnp.log(phi_state) - jnp.log(phi_state + mu)
    log_1_minus_p = jnp.log(mu) - jnp.log(phi_state + mu)
    per_sample_ll = term1 + (phi_state * log_p) + (y_float * log_1_minus_p)
    nll_data = -jnp.dot(gamma_state, per_sample_ll)    
    l2_prior = l2_reg * 0.5 * jnp.sum(w_state[1:]**2)
    return nll_data + l2_prior

def run_opt_negbin(
    init_w,
    init_log_phi,
    X,
    y,
    gamma_state,
    l2_reg,
    max_iter, 
    tol
):
    init_params = (init_w, init_log_phi)
    loss_fn = partial(negbin_neg_loglik_with_prior, X=X, y=y, 
                     gamma_state=gamma_state, l2_reg=l2_reg)
    return run_lbfgs_optimizer(loss_fn, init_params, max_iter, tol, 
                               error_fn=tuple_param_error_fn)

@partial(jit, static_argnames=("tol", "num_opt_steps")) 
def negbin_m_step_optax( 
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma_all_states: jnp.ndarray, 
    initial_W: jnp.ndarray,
    initial_phi: jnp.ndarray,
    l2_reg: jnp.float64,
    tol: float = 1e-3,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
    
    optimizer = vmap(
        run_opt_negbin, in_axes=(0, 0, None, None, 0, None, None, None) 
    )
    
    safe_initial_phi = jnp.maximum(initial_phi, 1e-6)
    initial_log_phi = jnp.log(safe_initial_phi)
    
    (optimized_Ws_T, optimized_log_phis), final_state = optimizer(
        initial_W.T, 
        initial_log_phi, 
        X, 
        y, 
        gamma_all_states.T, 
        l2_reg, 
        num_opt_steps, 
        tol
    )
    
    return optimized_Ws_T.T, jnp.exp(optimized_log_phis), final_state


@jit
def gamma_neg_loglik_with_prior(
    params: Tuple[jnp.ndarray, jnp.ndarray], 
    X: jnp.ndarray,  
    y: jnp.ndarray, 
    gamma_state: jnp.ndarray,
    l2_reg: jnp.float64,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    Computes the negative log-likelihood for a Gamma GLM for a single state.
    Includes protections for y=0 and extreme logits.
    """
    w_state, log_alpha_scalar = params
    alpha_state = jnp.exp(log_alpha_scalar) + 1e-6
    
    logits = X @ w_state
    log_mu = jnp.clip(logits, a_min=-10.0, a_max=20.0)
    
    log_beta = log_alpha_scalar - log_mu
    beta_state = jnp.exp(log_beta)
    
    y_float = jnp.maximum(y.astype(jnp.float32), eps)
    
    term1 = alpha_state * log_beta
    term2 = -gammaln(alpha_state)
    term3 = (alpha_state - 1.0) * jnp.log(y_float)
    term4 = -beta_state * y_float
    
    per_sample_ll = term1 + term2 + term3 + term4
    nll_data = -jnp.dot(gamma_state, per_sample_ll)
    l2_prior = l2_reg * 0.5 * jnp.sum(w_state[1:]**2)
    
    return nll_data + l2_prior

def run_opt_gamma(
    init_w,
    init_log_alpha,
    X,
    y,
    gamma_state,
    l2_reg,
    max_iter, 
    tol
):
    init_params = (init_w, init_log_alpha)
    loss_fn = partial(gamma_neg_loglik_with_prior, X=X, y=y, 
                     gamma_state=gamma_state, l2_reg=l2_reg)
    return run_lbfgs_optimizer(loss_fn, init_params, max_iter, tol, 
                               error_fn=tuple_param_error_fn)

@partial(jit, static_argnames=("tol", "num_opt_steps")) 
def gamma_m_step_optax( 
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma_all_states: jnp.ndarray, 
    initial_W: jnp.ndarray,
    initial_alpha: jnp.ndarray,
    l2_reg: jnp.float64,
    tol: float = 1e-3,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
    
    optimizer = vmap(
        run_opt_gamma, in_axes=(0, 0, None, None, 0, None, None, None) 
    )
    
    safe_initial_alpha = jnp.maximum(initial_alpha, 1e-6)
    initial_log_alpha = jnp.log(safe_initial_alpha) 

    (optimized_Ws_T, optimized_log_alphas), final_state = optimizer(
        initial_W.T, 
        initial_log_alpha, 
        X, 
        y, 
        gamma_all_states.T, 
        l2_reg, 
        num_opt_steps, 
        tol
    )
    return optimized_Ws_T.T, jnp.exp(optimized_log_alphas), final_state


def multinomial_negloglik_reduced(
    theta_reduced,
    U,
    xi_state,
    alpha_row_minus_1, 
    l2_reg
):
    """
    Optimizes P-1 weights for transitions from a single source state.
    theta_reduced shape: (n_features, n_states - 1)
    xi_state shape: (T-1, n_states) - expected transitions from this source state
    """
    zeros = jnp.zeros((theta_reduced.shape[0], 1), dtype=theta_reduced.dtype)
    theta_full = jnp.concatenate([theta_reduced, zeros], axis=1)
    logits = U @ theta_full  # (T-1, K)
    xi_normalized = xi_state / (jnp.sum(xi_state, axis=1, keepdims=True) + 1e-9)
    
    weights = jnp.sum(xi_state, axis=1)  # (T-1,)
    per_step_loss = optax.softmax_cross_entropy(logits, labels=xi_normalized)
    data_loss = jnp.sum(weights * per_step_loss)
    
    l2_penalty = l2_reg * 0.5 * jnp.sum(theta_reduced[1:]**2)
    total_log_probs = jnp.sum(nn.log_softmax(logits, axis=-1), axis=0)
    dirichlet_penalty = -jnp.sum(alpha_row_minus_1 * total_log_probs)
    
    return data_loss + dirichlet_penalty + l2_penalty

def run_opt_multinomial_reduced(
    init_params, 
    U,
    xi,
    alpha_row_minus_1,
    l2_reg,
    max_iter, 
    tol
):
    loss_fn = partial(multinomial_negloglik_reduced, U=U, xi_state=xi, 
                     alpha_row_minus_1=alpha_row_minus_1, l2_reg=l2_reg)
    return run_lbfgs_optimizer(loss_fn, init_params, max_iter, tol)

@partial(jit, static_argnames=("tol", "num_opt_steps")) 
def transitions_m_step_optax_reduced( 
    U: jnp.ndarray,                
    xi: jnp.ndarray,               
    initial_theta: jnp.ndarray,    # Shape: (n_features, n_states, n_states)
    dir_diag: jnp.ndarray,
    transition_ridge: float = 0.1,
    tol: float = 1e-3,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    M-step for input-driven transition parameters (theta).
    
    Optimizes parameters that control how inputs affect state transitions.
    For each source state, learns a softmax regression from inputs to next states.
    Uses reduced parameterization (last target state has zero weights).
    
    IMPORTANT: For input-driven transitions, use dir_diag = jnp.ones(K) to
    disable the Dirichlet prior (alpha-1 = 0). Using dir_diag = jnp.zeros(K)
    will cause numerical instability!
    
    Args:
        U: Transition input features, shape (T-1, D) where T is sequence length.
        xi: Expected transition counts, shape (T-1, K, K).
            xi[t, i, j] = P(state_t=i, state_t+1=j | observations).
        initial_theta: Initial parameters, shape (D, K, K).
                       theta[d, i, j] = influence of feature d on i→j transition.
        dir_diag: Dirichlet prior diagonal. Use jnp.ones(K) for input-driven.
        transition_ridge: L2 regularization strength.
        tol: Convergence tolerance.
        num_opt_steps: Maximum L-BFGS iterations.
    
    Returns:
        theta: Optimized transition parameters, shape (D, K, K).
               Last target state (K-1) has zero weights for identifiability.
        final_states: Optimizer states for diagnostics.
    """ 
    
    n_states = initial_theta.shape[1]
    alpha_minus_1_matrix = (dir_diag - 1.0) * jnp.eye(n_states)    
    theta_reduced_init = initial_theta[:, :, :-1]

    optimizer = jax.vmap(
        run_opt_multinomial_reduced, 
        in_axes=(1, None, 1, 0, None, None, None) 
    )

    optimized_theta_reduced, final_state = optimizer(
        theta_reduced_init, U, xi, alpha_minus_1_matrix, transition_ridge, num_opt_steps, tol
    )
    
    optimized_theta_reduced = jnp.moveaxis(optimized_theta_reduced, 0, 1)
    zeros_col = jnp.zeros((optimized_theta_reduced.shape[0], optimized_theta_reduced.shape[1], 1))
    final_theta_full = jnp.concatenate([optimized_theta_reduced, zeros_col], axis=-1)
    return final_theta_full, final_state


@jit
def compute_A_from_theta_and_inputs(
    U: jnp.ndarray, # (n_steps, n_features)
    theta: jnp.ndarray # (n_features, n_states, n_states)
):
    logits = jnp.einsum('ij,jmn->imn', U, theta, optimize="optimal")
    A = jax.nn.softmax(logits, axis=-1) # use -1 ?
    return A


@jit
def compute_mean_map_contributions(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    gamma: jnp.ndarray
):
    """
    Computes Sz_XX_contrib and Sz_XY_contrib for a single (X, Y, g) item.
    
    Args:
        X: Input data of shape (N, p).
        Y: Target data of shape (N, m).
        gamma: Responsibilities of shape (N,).

    Returns:
        Sz_XX_contrib: Contribution to Sz_XX.
        Sz_XY_contrib: Contribution to Sz_XY.
    """

    sqrt_gamma = jnp.sqrt(gamma)
    Xsz = X * sqrt_gamma[:, jnp.newaxis]
    Ysz = Y * sqrt_gamma[:, jnp.newaxis]
    
    Sz_XX_contrib = Xsz.T @ Xsz
    Sz_XY_contrib = Xsz.T @ Ysz
    return Sz_XX_contrib, Sz_XY_contrib

def batch_ridge_regression(
    X_gauss_set: List[jnp.ndarray],
    Y_gauss_set: List[jnp.ndarray],
    gamma_set: List[jnp.ndarray],
    ridge_lambda: float,
    p: int,  # Number of features in X
    m: int   # Number of features in Y (or tasks)
):
    """
    Performs batch ridge regression with varying numbers of samples per batch item.

    Args:
        X_gauss_set: List of JAX arrays, each X_i of shape (N_i, p).
        Y_gauss_set: List of JAX arrays, each Y_i of shape (N_i, m).
        gamma_set: List of JAX arrays, each g_i of shape (N_i,).
        ridge_lambda: Regularization parameter.
        p: Dimensionality of features in X.
        m: Dimensionality of targets in Y.

    Returns:
        Wz: The solution weights of shape (p, m).
    """
    Sz_XX_accum = jnp.zeros((p, p), dtype=X_gauss_set[0].dtype) 
    Sz_XY_accum = jnp.zeros((p, m), dtype=Y_gauss_set[0].dtype)

    for X, Y, gamma in zip(X_gauss_set, Y_gauss_set, gamma_set):
        Sz_XX_contrib, Sz_XY_contrib = compute_mean_map_contributions(X, Y, gamma)
        Sz_XX_accum += Sz_XX_contrib
        Sz_XY_accum += Sz_XY_contrib

    ridge_penalty = ridge_lambda * jnp.eye(p, dtype=Sz_XX_accum.dtype)
    ridge_penalty = ridge_penalty.at[0, 0].set(0.0)
    Sz_XX_final = Sz_XX_accum + ridge_penalty
    Wz = jnp.linalg.solve(Sz_XX_final, Sz_XY_accum)
    return Wz

@jit
def compute_covar_map_contributions(
    W: jnp.ndarray, 
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    gamma: jnp.ndarray
):
    """
    Computes individual contributions for the MAP estimate of the covariance matrix. 

    Args:
        W: Weight matrix of shape (p, m).
        X: Input data of shape (N, p).
        Y: Target data of shape (N, m).
        gamma: Responsibilities of shape (N,).

    Returns:
        Covariance matrix component of shape (m, m).
    """
    sqrt_gamma = jnp.sqrt(gamma)
    M = X @ W
    residuals = sqrt_gamma[:, jnp.newaxis] * (Y - M)
    sum_of_squares_resid = residuals.T @ residuals
    sum_of_gamma = jnp.sum(gamma)
    return sum_of_squares_resid, sum_of_gamma

def batch_full_covariance(
    W: jnp.ndarray, 
    X_gauss_set: List[jnp.ndarray], 
    Y_gauss_set: List[jnp.ndarray], 
    gamma_set: List[jnp.ndarray]
):
    """
    Computes the MAP estimate of the covariance matrix.

    Args:
        W: Weight matrix of shape (p, m).
        X_gauss_set: List of JAX arrays, each X_i of shape (N_i, p).
        Y_gauss_set: List of JAX arrays, each Y_i of shape (N_i, m).
        gamma_set: List of JAX arrays, each g_i of shape (N_i,).

    Returns:
        Covariance matrix of shape (m, m).
    """
    sum_of_squares_resid_accum = jnp.zeros((W.shape[1], W.shape[1]), dtype=X_gauss_set[0].dtype)
    sum_of_gamma_accum = 0.0

    for X, Y, gamma in zip(X_gauss_set, Y_gauss_set, gamma_set):
        sum_of_squares_resid_contrib, sum_of_gamma_contrib = compute_covar_map_contributions(W, X, Y, gamma)
        sum_of_squares_resid_accum += sum_of_squares_resid_contrib
        sum_of_gamma_accum += sum_of_gamma_contrib

    Sigma = sum_of_squares_resid_accum / sum_of_gamma_accum
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


@partial(jax.jit, static_argnames=("p", "m", "n_states", "ridge_pen", "jitter"))
def batch_m_step_updates(
    X_gauss_set: List[jnp.ndarray],
    Y_gauss_set: List[jnp.ndarray],
    gamma_set_expanded: List[jnp.ndarray], # List[(Nk, n_states)]
    ridge_pen: float,
    p: int,
    m: int,
    n_states: int,
    jitter: float = 1e-3
):
    """
    Computes M-step updates for W and Sigma for all states simultaneously.

    Args:
        
    """
    dtype = X_gauss_set[0].dtype
    Sz_XX_accum = jnp.zeros((p, p, n_states), dtype=dtype)
    Sz_XY_accum = jnp.zeros((p, m, n_states), dtype=dtype)
    sum_of_gamma_accum = jnp.zeros((n_states,), dtype=dtype)

    for X, Y, G in zip(X_gauss_set, Y_gauss_set, gamma_set_expanded):
        # X:(Nk, p), Y:(Nk, m), G:(Nk, n_states)
        sqrt_G = jnp.sqrt(G) # (Nk, n_states)

        Xsz = X[:, :, None] * sqrt_G[:, None, :] # (Nk, p, n_states)
        Ysz = Y[:, :, None] * sqrt_G[:, None, :] # (Nk, m, n_states)

        Sz_XX_accum += jnp.einsum('ipk,iqk->pqk', Xsz, Xsz, optimize="optimal")
        Sz_XY_accum += jnp.einsum('ipk,imk->pmk', Xsz, Ysz, optimize="optimal")
        sum_of_gamma_accum += jnp.sum(G, axis=0)

    ridge_penalty = ridge_pen * jnp.eye(p, dtype=dtype)
    ridge_penalty = ridge_penalty.at[0, 0].set(0.0)
    Sz_XX_final = Sz_XX_accum + ridge_penalty[..., None] # Add ridge to each state
    
    Sz_XX_final_T = jnp.moveaxis(Sz_XX_final, -1, 0) # (n_states, p, p)
    Sz_XY_accum_T = jnp.moveaxis(Sz_XY_accum, -1, 0) # (n_states, p, m)
    
    W_all_T = jnp.linalg.solve(Sz_XX_final_T, Sz_XY_accum_T) # (n_states, p, m)
    W = jnp.moveaxis(W_all_T, 0, -1) # (p, m, n_states)

    sum_sq_resid_all = jnp.zeros((m, m, n_states), dtype=dtype)
    for X, Y, G in zip(X_gauss_set, Y_gauss_set, gamma_set_expanded):
        M = jnp.einsum('ip,pmk->imk', X, W) # (Nk, m, n_states)
        residuals = Y[:, :, None] - M # (Nk, m, n_states)
        weighted_residuals = residuals * jnp.sqrt(G)[:, None, :] # (Nk, m, n_states)
        sum_sq_resid_all += jnp.einsum('imk,ink->mnk', weighted_residuals, weighted_residuals, optimize="optimal")

    safe_sum_of_gamma = jnp.maximum(sum_of_gamma_accum, 1e-9)
    Sigma = sum_sq_resid_all / safe_sum_of_gamma[None, None, :]
    Sigma = 0.5 * (Sigma + jnp.moveaxis(Sigma, [0, 1], [1, 0]))
    Sigma = Sigma + jnp.eye(m, dtype=dtype)[..., None] * jitter 

    return W, Sigma

def initial_distribution_m_step(
    gammas: List[jnp.ndarray], 
    dirichlet_alpha: float = 1.0001
    ) -> jnp.ndarray:
    """
    Updates initial state probabilities (pi0) using the expected state at t=0.
    
    Args:
        gammas: List of state responsibilities, each shape (T_i, K).
        dirichlet_alpha: Concentration parameter for Dirichlet prior (default slightly > 1 for stability).
        
    Returns:
        log_pi0: Log of updated initial state probabilities.
    """
    K = gammas[0].shape[1]
    
    if isinstance(gammas, list):
        init_counts = sum([g[0] for g in gammas]) # (K,)
    else:
        if gammas.ndim == 3:
             init_counts = jnp.sum(gammas[:, 0, :], axis=0) # (K,)
        else:
             init_counts = gammas[0] # (K,)

    pseudo_counts = dirichlet_alpha - 1.0
    denom = jnp.sum(init_counts + pseudo_counts)
    pi0 = (init_counts + pseudo_counts) / denom
    
    return jnp.log(pi0 + 1e-10)