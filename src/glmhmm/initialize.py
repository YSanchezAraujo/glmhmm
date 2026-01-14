import jax.numpy as jnp
from jax import random
from .structs import GLMHMMData, GLMHMMConfig, GLMHMMParams

def initialize_params(data: GLMHMMData, config: GLMHMMConfig, seed: int = 0) -> GLMHMMParams:
    key = random.PRNGKey(seed)
    K = config.num_states
    
    log_pi0 = jnp.log(jnp.ones(K) / K)
    log_A = jnp.log(jnp.ones((K, K)) / K)
    
    theta = None
    if data.x_trans is not None:
        key, subkey = random.split(key)
        D_trans = data.x_trans[0].shape[1]
        theta = 0.01 * random.normal(subkey, (D_trans, K, K))

    W_bern = None
    if data.x_bern is not None:
        key, subkey = random.split(key)
        D_bern = data.x_bern[0].shape[1]
        W_bern = 0.01 * random.normal(subkey, (D_bern, K))

    # Negative Binomial
    W_nb, phi_nb = None, None 
    if data.x_negbin is not None:
        key, subkey = random.split(key)
        D_nb = data.x_negbin[0].shape[1]
        W_nb = 0.01 * random.normal(subkey, (D_nb, K))
        phi_nb = jnp.ones(K) * 5.0 # Positive start

    # Gamma
    W_gamma, alpha_gamma = None, None
    if data.x_gamma is not None:
        key, subkey = random.split(key)
        D_gamma = data.x_gamma[0].shape[1]
        W_gamma = 0.01 * random.normal(subkey, (D_gamma, K))
        alpha_gamma = jnp.ones(K) * 2.0 # Positive start

    # Multinomial
    W_multinom = None
    if data.x_multinom is not None:
        key, subkey = random.split(key)
        D_mn = data.x_multinom[0].shape[1]
        C_mn = data.y_multinom[0].max() + 1 if data.y_multinom else 2
        W_multinom = 0.01 * random.normal(subkey, (D_mn, C_mn, K))

    # Gaussian
    W_gauss, Sigma_gauss = None, None
    if data.x_gauss is not None:
        key, subkey = random.split(key)
        if isinstance(data.x_gauss[0], list):
             D_gauss = data.x_gauss[0][0].shape[1]
             M_gauss = data.y_gauss[0][0].shape[1]
        else:
             D_gauss = data.x_gauss[0].shape[1]
             M_gauss = data.y_gauss[0].shape[1]
        W_gauss = 0.01 * random.normal(subkey, (D_gauss, M_gauss, K))
        Sigma_gauss = jnp.tile(jnp.eye(M_gauss)[:, :, None], (1, 1, K))

    return GLMHMMParams(log_pi0, log_A, theta, W_bern, W_multinom, W_nb, phi_nb, W_gamma, alpha_gamma, W_gauss, Sigma_gauss)