import jax.numpy as jnp
from typing import NamedTuple, Optional

class GLMHMMConfig(NamedTuple):
    num_states: int
    num_em_iters: int
    min_em_iters: int = 0
    tol: float = 1e-3
    l2_reg_bern: float = 1.0
    l2_reg_multinom: float = 1.0
    l2_reg_negbin: float = 1.0
    l2_reg_gamma: float = 1.0
    l2_reg_trans: float = 1.0
    l2_reg_gauss: float = 1.0
    seed: int = 0
    init_noise_scale: float = 0.1

class GLMHMMParams(NamedTuple):
    # transitions
    log_pi0: jnp.ndarray
    log_A: jnp.ndarray
    theta: Optional[jnp.ndarray] = None
    # emissions
    W_bern: Optional[jnp.ndarray] = None
    W_multinom: Optional[jnp.ndarray] = None
    W_negbin: Optional[jnp.ndarray] = None
    phi_negbin: Optional[jnp.ndarray] = None
    W_gamma: Optional[jnp.ndarray] = None
    alpha_gamma: Optional[jnp.ndarray] = None
    W_gauss: Optional[jnp.ndarray] = None
    Sigma_gauss: Optional[jnp.ndarray] = None

class GLMHMMData(NamedTuple):
    x_trans: Optional[list] = None # list of (T, K)
    x_bern: Optional[list] = None
    y_bern: Optional[list] = None
    x_multinom: Optional[list] = None
    y_multinom: Optional[list] = None
    x_negbin: Optional[list] = None
    y_negbin: Optional[list] = None
    x_gamma: Optional[list] = None
    y_gamma: Optional[list] = None
    x_gauss: Optional[list] = None # list of arrays (total_samples, p)
    y_gauss: Optional[list] = None # list of arrays (total_samples, m)
    gauss_first_idx: Optional[list] = None # list of arrays (T_i,)
    gauss_last_idx: Optional[list] = None # list of arrays (T_i,)

class EMState(NamedTuple):
    params: GLMHMMParams
    iteration: int
    lml: float
    lml_history: jnp.ndarray