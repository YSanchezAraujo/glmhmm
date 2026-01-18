import jax
import jax.numpy as jnp
from jax import random
from .structs import GLMHMMData, GLMHMMConfig, GLMHMMParams

def kmeans_init(X, K, key, max_iters=10):
    """
    Simple K-means initialization.
    X: (N, D) or (N,) data points.
    K: number of clusters.
    Returns: centers (K, D)
    """
    N = X.shape[0]
    
    if X.ndim == 1:
        X = X[:, None]
    
    key, subkey = random.split(key)
    center_indices = random.choice(subkey, N, shape=(K,), replace=False)
    centers = X[center_indices] # (K, D)
    
    for _ in range(max_iters):
        dists = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        assignments = jnp.argmin(dists, axis=1)
        
        new_centers = []
        for k in range(K):
            mask = (assignments == k)
            count = jnp.sum(mask)
            center_k = jnp.sum(X * mask[:, None], axis=0) / (count + 1e-5)
            center_k = jnp.where(count < 1.0, centers[k], center_k)
            new_centers.append(center_k)
        centers = jnp.stack(new_centers)
        
    return centers

def initialize_params(data: GLMHMMData, config: GLMHMMConfig, seed: int = 0) -> GLMHMMParams:
    key = random.PRNGKey(seed)
    K = config.num_states
    
    log_pi0 = jnp.log(jnp.ones(K) / K)
    log_A = jnp.log(jnp.ones((K, K)) / K)
    
    theta = None
    if data.x_trans is not None:
        key, subkey = random.split(key)
        D_trans = data.x_trans[0].shape[1]
        theta = 0.1 * random.normal(subkey, (D_trans, K, K))

    W_bern = None
    if data.x_bern is not None:
        key, subkey = random.split(key)
        D_bern = data.x_bern[0].shape[1]
        W_bern = 0.1 * random.normal(subkey, (D_bern, K))

    W_nb, phi_nb = None, None 
    if data.x_negbin is not None:
        key, subkey = random.split(key)
        D_nb = data.x_negbin[0].shape[1]
        
        if data.y_negbin is not None:
             y_flat = jnp.concatenate([y for y in data.y_negbin]) # (N,)
             centers = kmeans_init(y_flat, K, subkey) # (K, 1)
             W_nb = jnp.zeros((D_nb, K))
             log_centers = jnp.log(centers.flatten() + 1e-3)
             W_nb = W_nb.at[0, :].set(log_centers)
             W_nb += 0.1 * random.normal(subkey, (D_nb, K))
        else:
             W_nb = 0.1 * random.normal(subkey, (D_nb, K))
             
        phi_nb = jnp.ones(K) * 5.0

    W_gamma, alpha_gamma = None, None
    if data.x_gamma is not None:
        key, subkey = random.split(key)
        D_gamma = data.x_gamma[0].shape[1]
        
        if data.y_gamma is not None:
             y_flat = jnp.concatenate([y for y in data.y_gamma])
             centers = kmeans_init(y_flat, K, subkey) # (K, 1)
             W_gamma = jnp.zeros((D_gamma, K))
             log_centers = jnp.log(centers.flatten() + 1e-3)
             W_gamma = W_gamma.at[0, :].set(log_centers)
             W_gamma += 0.1 * random.normal(subkey, (D_gamma, K))
        else:
             W_gamma = 0.1 * random.normal(subkey, (D_gamma, K))
             
        alpha_gamma = jnp.ones(K) * 2.0 

    W_multinom = None
    if data.x_multinom is not None:
        key, subkey = random.split(key)
        D_mn = data.x_multinom[0].shape[1]
        C_mn = data.y_multinom[0].max() + 1 if data.y_multinom else 2
        
        if data.y_multinom is not None:
             y_flat = jnp.concatenate([y for y in data.y_multinom]) # (N_total,)
             y_onehot = jax.nn.one_hot(y_flat, C_mn) # (N, C)
             centers = kmeans_init(y_onehot, K, subkey) # (K, C)
             W_multinom = jnp.zeros((D_mn, C_mn, K))
             log_probs = jnp.log(centers.T + 1e-3) # (C, K)
             W_multinom = W_multinom.at[0, :, :].set(log_probs)
             W_multinom += 0.1 * random.normal(subkey, (D_mn, C_mn, K))
        else:
            W_multinom = 0.1 * random.normal(subkey, (D_mn, C_mn, K))

    W_gauss, Sigma_gauss = None, None
    if data.x_gauss is not None:
        key, subkey = random.split(key)
        if isinstance(data.x_gauss[0], list):
             D_gauss = data.x_gauss[0][0].shape[1]
             M_gauss = data.y_gauss[0][0].shape[1]
             y_flat = jnp.concatenate([jnp.concatenate(seq) for seq in data.y_gauss])
        else:
             D_gauss = data.x_gauss[0].shape[1]
             M_gauss = data.y_gauss[0].shape[1]
             y_flat = jnp.concatenate([y for y in data.y_gauss])

        W_gauss = 0.1 * random.normal(subkey, (D_gauss, M_gauss, K))
        
        if data.y_gauss is not None:
             centers = kmeans_init(y_flat, K, subkey) # (K, M)
             for k in range(K):
                 W_gauss = W_gauss.at[0, :, k].set(centers[k])
        
        Sigma_gauss = jnp.tile(jnp.eye(M_gauss)[:, :, None], (1, 1, K))

    return GLMHMMParams(log_pi0, log_A, theta, W_bern, W_multinom, W_nb, phi_nb, W_gamma, alpha_gamma, W_gauss, Sigma_gauss)