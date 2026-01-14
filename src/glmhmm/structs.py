import jax.numpy as jnp
from typing import NamedTuple, Optional

class GLMHMMConfig(NamedTuple):
    num_states: int
    num_em_iters: int
    tol: float = 1e-3
    l2_reg_bern: float = 1.0
    l2_reg_multinom: float = 1.0
    l2_reg_negbin: float = 1.0
    l2_reg_gamma: float = 1.0
    l2_reg_trans: float = 1.0
    l2_reg_gauss: float = 1.0

class GLMHMMParams(NamedTuple):
    # transitions
    log_pi0: jnp.ndarray
    log_A: jnp.ndarray
    theta: Optional[jnp.ndarray]
    # emissions
    W_bern: Optional[jnp.ndarray]
    W_multinom: Optional[jnp.ndarray]
    W_negbin: Optional[jnp.ndarray]
    phi_negbin: Optional[jnp.ndarray]
    W_gamma: Optional[jnp.ndarray]
    alpha_gamma: Optional[jnp.ndarray]
    W_gauss: Optional[jnp.ndarray]
    Sigma_gauss: Optional[jnp.ndarray]

class GLMHMMData(NamedTuple):
    x_trans: Optional[list] # list of (T, K)
    x_bern: Optional[list]
    y_bern: Optional[list]
    x_multinom: Optional[list]
    y_multinom: Optional[list]
    x_negbin: Optional[list] 
    y_negbin: Optional[list]
    x_gamma: Optional[list]
    y_gamma: Optional[list]
    x_gauss: Optional[list] # list of arrays (total_samples, p)
    y_gauss: Optional[list] # list of arrays (total_samples, m)
    gauss_first_idx: Optional[list] # list of arrays (T_i,)
    gauss_last_idx: Optional[list] # list of arrays (T_i,)

class EMState(NamedTuple):
    params: GLMHMMParams
    iteration: int
    lml: float
    lml_history: jnp.ndarray