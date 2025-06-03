import jax
import jax.numpy as jnp
from typing import List
from functools import partial
from typing import Tuple
from jax import jit, lax, nn, vmap
import optax
import optax.tree_utils as otu
from collections import namedtuple
import jax.scipy.optimize as jso

@jit
def bern_neg_loglik_with_prior(
    w_bern_state: jnp.ndarray, 
    X_bern: jnp.ndarray,  
    y_bern: jnp.ndarray, 
    gamma_state: jnp.ndarray 
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
    l2_prior = 0.5 * jnp.sum(w_bern_state**2)
    total_nll = nll_data + l2_prior
    return total_nll / X_bern.shape[0] # for stability because of optax, just incase

@jit
def optimize_single_state_jso(
    w_initial_s: jnp.ndarray,
    gamma_s: jnp.ndarray,
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    maxiter: int = 500, 
    gtol: float = 1e-4 
) -> namedtuple:
    """
    Optimizes weights for a single state using jax.scipy.optimize.minimize (BFGS).
    """
    loss_fn_s = partial(bern_neg_loglik_with_prior, X_bern=X_bern, y_bern=y_bern, gamma_state=gamma_s)

    result = jso.minimize(
        fun=loss_fn_s,
        x0=w_initial_s,
        method='BFGS',
        options={
            'maxiter': maxiter,
            'gtol': gtol
        }
    )

    opt = namedtuple('opt', ['w', 'loss', 'success', "niter"])(
        w = result.x,
        loss = result.fun,
        success = result.success,
        niter = result.nit
    )

    return opt

def bern_m_step_jso(
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    gamma_all_states: jnp.ndarray, # (n_steps, n_states)
    initial_W_bern: jnp.ndarray,   # (n_features, n_states)
    num_opt_steps: int = 500,
    gtol: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimizes weights for all states using jso.minimize via vmap.
    """
    partial_optimizer = partial(optimize_single_state_jso,
                                 X_bern=X_bern,
                                 y_bern=y_bern,
                                 maxiter=num_opt_steps,
                                 gtol=gtol)


    opt_results = vmap(partial_optimizer, in_axes=(0, 0))(initial_W_bern.T, gamma_all_states.T)
    optimized_Ws_T = opt_results.w
    final_losses = opt_results.loss  

    return optimized_Ws_T.T, final_losses # (p, n_states), (n_states,)

# this is mostly from the optax documentation
# https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver
# small change to otu.tree_l2_norm
def run_opt_bern(
    init_params, 
    X_bern,
    y_bern,
    gamma_state,
    max_iter, 
    tol
):
    fun = partial(bern_neg_loglik_with_prior, X_bern=X_bern, y_bern=y_bern, gamma_state=gamma_state)
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    opt = optax.lbfgs()

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = lax.while_loop(
        continuing_criterion, step, init_carry
    )

    return final_params, final_state

@partial(jax.jit, static_argnames=("tol", "num_opt_steps"))
def bern_init_opt(
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
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
    final_params, _ = run_opt_bern(initial_w, X_bern, y_bern, gamma_state_ones, num_opt_steps, tol)
    return final_params

@partial(jax.jit, static_argnames=("tol", "num_opt_steps")) 
def bern_m_step_optax( 
    X_bern: jnp.ndarray,
    y_bern: jnp.ndarray,
    gamma_all_states: jnp.ndarray, 
    initial_W_bern: jnp.ndarray,
    tol: float = 1e-2,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Optimizes weights for a all states in the GLMHMM m-step

    Args:
        X_bern: Design matrix for bernoulli observations.
        y_bern: Binary response vector (0 or 1).
        gamma_all_states: Responsibilities (gammas) for all states.
        initial_W_bern: Initial weight vector for the current HMM state.
        learning_rate: Learning rate for the optimizer.
        num_opt_steps: Number of optimization steps.

    Returns:
        Estimated parameter vectors, and the final losses from the optimizer
    """
                                 
    optimizer = vmap(
        run_opt_bern, in_axes=(0, None, None, 0, None, None) 
    )

    optimized_Ws_T, final_state = optimizer(
        initial_W_bern.T, X_bern, y_bern, gamma_all_states.T, num_opt_steps, tol
    )

    return optimized_Ws_T.T, final_state


@jit
def multinomial_negloglik(theta_state, U, xi_state, dir_diag):
    """
    Computes a weighted negative log likelihood of the data

    Args:
        theta_state: Matrix of weights for state i
        U: Matrix of inputs for the transitions
        xi_state: Matrix of transitions over time
        dir_diag: Array of floats, dirichlet prior penalty

    returns:
        Scalar valued negative log likelihood
    """
    logits = U @ theta_state
    loss = optax.softmax_cross_entropy(logits, xi_state)
    log_probs = nn.log_softmax(logits, axis=1)
    penalty = -jnp.sum((dir_diag - 1.0)[jnp.newaxis, :] * log_probs)
    total_loss = jnp.sum(loss) + penalty
    return total_loss / U.shape[0]

def run_opt_multinomial(
    init_params, 
    U,
    xi,
    dir_diag,
    max_iter, 
    tol
):
    fun = partial(multinomial_negloglik, U=U, xi_state=xi, dir_diag=dir_diag)
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    opt = optax.lbfgs()

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = lax.while_loop(
        continuing_criterion, step, init_carry
    )

    return final_params, final_state

@partial(jax.jit, static_argnames=("tol", "num_opt_steps")) 
def transitions_m_step_optax( 
    U: jnp.ndarray,                # (n_trans_steps, n_features), inputs
    xi: jnp.ndarray,               # (n_trans_steps, n_states, n_states)
    initial_theta: jnp.ndarray,    # (n_features, n_states, n_states)
    dir_diag: jnp.ndarray,
    tol: float = 1e-2,
    num_opt_steps: int = 2000
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Optimizes weights for a all states in the GLMHMM m-step, input driven transitions

    Args:


    Returns:
        Estimated parameter matrices, and the final state from the optimizer
    """
                                 
    optimizer = vmap(
        run_opt_multinomial, in_axes=(1, None, 1, None, None, None) 
    )

    optimized_theta, final_state = optimizer(
        initial_theta, U, xi, dir_diag, num_opt_steps, tol
    )

    return optimized_theta, final_state

@jit
def compute_A_from_theta_and_inputs(
    U: jnp.ndarray, # (n_steps, n_features)
    theta: jnp.ndarray # (n_features, n_states, n_states)
):
    logits = jnp.einsum('ij,jmn->imn', U, theta, optimize="optimal")
    A = jax.nn.softmax(logits, axis=2)
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
    jitter: float = 1e-5
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