"""
Generic optimizer utilities for M-step optimization in GLMHMM.

This module provides reusable infrastructure for L-BFGS optimization
to eliminate code duplication across different emission distributions.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial
from jax import lax, vmap
import optax
import optax.tree_utils as otu


def default_error_fn(grad: Any) -> float:
    """
    Default error function for convergence checking.
    Uses tree L2 norm on gradient.
    
    Args:
        grad: Gradient (can be array or pytree)
        
    Returns:
        Scalar error value
    """
    return otu.tree_l2_norm(grad)


def tuple_param_error_fn(grad: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
    """
    Error function for tuple parameters (weights, dispersion).
    Used for distributions with auxiliary parameters like NegBin and Gamma.
    
    Args:
        grad: Tuple of (grad_weights, grad_dispersion)
        
    Returns:
        Scalar error value
    """
    grad_w, grad_disp = grad
    return jnp.linalg.norm(grad_w) + jnp.abs(grad_disp)


def run_lbfgs_optimizer(
    loss_fn: Callable,
    init_params: Any,
    max_iter: int,
    tol: float,
    error_fn: Optional[Callable] = None
) -> Tuple[Any, Any]:
    """
    Generic L-BFGS optimizer for any loss function.
    
    This function eliminates the need to write separate optimizer functions
    for each emission distribution. It handles the standard L-BFGS optimization
    loop with customizable convergence criteria.
    
    Args:
        loss_fn: Loss function to minimize. Should be a function of only the
                 parameters (other arguments should be bound via functools.partial)
        init_params: Initial parameters. Can be a single array or a tuple of arrays
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for gradient norm
        error_fn: Optional custom error function for convergence check.
                  Takes gradient and returns scalar error.
                  Default uses tree_l2_norm on gradient.
                  
    Returns:
        final_params: Optimized parameters
        final_state: Final optimizer state
        
    Example:
        >>> loss = partial(my_loss_fn, X=X, y=y, l2_reg=1.0)
        >>> params, state = run_lbfgs_optimizer(loss, init_w, 2000, 1e-3)
    """
    if error_fn is None:
        error_fn = default_error_fn
    
    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)
    opt = optax.lbfgs()

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=loss_fn
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = error_fn(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def simple_vmap_m_step(
    single_state_optimizer: Callable,
    init_params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma_all_states: jnp.ndarray,
    l2_reg: float,
    max_iter: int,
    tol: float,
    transpose_params: bool = True
) -> Tuple[jnp.ndarray, Any]:
    """
    Simple vmap wrapper for single-parameter M-steps (e.g., Bernoulli).
    
    This handles the common pattern of:
    1. Transpose parameters from (D, K) to (K, D) for vmap
    2. Vmap the optimizer across states
    3. Transpose results back to (D, K)
    
    Args:
        single_state_optimizer: Optimizer function for a single state
        init_params: Initial parameters, shape (D, K)
        X: Design matrix, shape (N, D)
        y: Response data, shape (N,)
        gamma_all_states: Responsibilities, shape (N, K)
        l2_reg: L2 regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance
        transpose_params: If True, transpose params before/after vmap
        
    Returns:
        optimized_params: Optimized parameters, shape (D, K)
        final_states: Final optimizer states for all states
    """
    optimizer = vmap(
        single_state_optimizer, 
        in_axes=(0, None, None, 0, None, None, None)
    )
    
    if transpose_params:
        init_params_T = init_params.T
        gamma_T = gamma_all_states.T
    else:
        init_params_T = init_params
        gamma_T = gamma_all_states
    
    optimized_params_T, final_states = optimizer(
        init_params_T, X, y, gamma_T, l2_reg, max_iter, tol
    )
    
    if transpose_params:
        optimized_params = optimized_params_T.T
    else:
        optimized_params = optimized_params_T
    
    return optimized_params, final_states


def dual_param_vmap_m_step(
    single_state_optimizer: Callable,
    init_params_1: jnp.ndarray,
    init_params_2: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma_all_states: jnp.ndarray,
    l2_reg: float,
    max_iter: int,
    tol: float
) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
    """
    Vmap wrapper for dual-parameter M-steps (e.g., NegBin, Gamma).
    
    Handles distributions with both weights and a dispersion parameter.
    
    Args:
        single_state_optimizer: Optimizer function for a single state
        init_params_1: Initial weights, shape (D, K)
        init_params_2: Initial dispersion params, shape (K,)
        X: Design matrix, shape (N, D)
        y: Response data, shape (N,)
        gamma_all_states: Responsibilities, shape (N, K)
        l2_reg: L2 regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        optimized_params_1: Optimized weights, shape (D, K)
        optimized_params_2: Optimized dispersion params, shape (K,)
        final_states: Final optimizer states for all states
    """
    optimizer = vmap(
        single_state_optimizer, 
        in_axes=(0, 0, None, None, 0, None, None, None)
    )
    
    init_params_1_T = init_params_1.T
    gamma_T = gamma_all_states.T
    
    (optimized_params_1_T, optimized_params_2), final_states = optimizer(
        init_params_1_T, init_params_2, X, y, gamma_T, l2_reg, max_iter, tol
    )
    
    optimized_params_1 = optimized_params_1_T.T
    
    return optimized_params_1, optimized_params_2, final_states
