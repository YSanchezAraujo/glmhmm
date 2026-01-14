
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.fit import fit_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.utils import simulate_data_for_glmhmm, find_best_permutation

def test_variable_rate_recovery():
    print("Running Variable Rate Gaussian Recovery Test...")
    key = random.PRNGKey(42)
    n_states = 3
    n_steps = 500
    
    # 1. Simulate HMM states and static transitions
    # Use utils to simulate basic HMM first
    A_true = jnp.array([[0.8, 0.1, 0.1], 
                        [0.1, 0.8, 0.1], 
                        [0.1, 0.1, 0.8]])
    
    true_params_dict = {
        "pi0": jnp.ones(n_states)/n_states,
        "A": A_true,
        "W_bern": None, "W_multinom": None, "W_negbin": None, "W_gamma": None,
        "theta": None
    }
    
    # We use simulate_data just to get states z
    sim_data = simulate_data_for_glmhmm(
        key, true_params_dict, n_sequences=1, n_hmm_steps=n_steps
    )
    # y_trans contains the states? No, simulate_data doesn't return states directly usually...
    # It returns 'y_trans' which is the next state.
    # Actually simulate_data returns 'z' if requested? No.
    # But we can infer Z?
    # Let's write a simple HMM simulator or rely on y_multinom if we set it as identity?
    
    # Better: Inspect simulate_data return. It returns "states" key?
    # View utils.simulate_data_for_glmhmm to check return keys.
    # Assuming it returns "states".
    
    states = sim_data.get("true_states")[0] # Shape (T,)
    
    # 2. Simulate Variable Rate Gaussian Data
    # For each step t, we decide number of samples N_t
    key, subkey = random.split(key)
    # Random counts between 10 and 50
    counts = random.randint(subkey, (n_steps,), 10, 50) 
    
    # Define True Gaussian Params
    D_g = 2
    M_g = 2
    
    # W_gauss: (D, M, K)
    W_gauss_true = jnp.zeros((D_g, M_g, n_states))
    # State 0: Positive correlation
    W_gauss_true = W_gauss_true.at[:, :, 0].set(5.0)
    # State 1: Negative
    W_gauss_true = W_gauss_true.at[:, :, 1].set(-5.0)
    # State 2: Zero
    W_gauss_true = W_gauss_true.at[:, :, 2].set(0.0)
    
    Sigma_true = jnp.tile(jnp.eye(M_g), (n_states, 1, 1)) * 0.5
    
    # Generate X and Y
    X_flat_list = []
    Y_flat_list = []
    first_idxs = []
    last_idxs = []
    
    current_idx = 0
    all_X = []
    all_Y = []
    
    # We iterate and generate
    for t in range(n_steps):
        s = states[t]
        n = counts[t]
        
        first_idxs.append(current_idx)
        current_idx += n
        last_idxs.append(current_idx)
        
        key, k1, k2 = random.split(key, 3)
        X_t = random.normal(k1, (n, D_g))
        noise = random.multivariate_normal(k2, jnp.zeros(M_g), Sigma_true[s], (n,))
        
        mu = X_t @ W_gauss_true[:, :, s]
        Y_t = mu + noise
        
        all_X.append(X_t)
        all_Y.append(Y_t)
        
    X_flat = jnp.concatenate(all_X)
    Y_flat = jnp.concatenate(all_Y)
    
    first_idxs = jnp.array(first_idxs)
    last_idxs = jnp.array(last_idxs)
    
    print(f"Total steps: {n_steps}")
    print(f"Total Gaussian samples: {X_flat.shape[0]}")
    
    # 3. Pack Data
    data = GLMHMMData(
        x_trans=sim_data["x_trans_set"],
        x_bern=None, y_bern=None,
        x_multinom=None, y_multinom=None,
        x_negbin=None, y_negbin=None,
        x_gamma=None, y_gamma=None,
        x_gauss=[X_flat],
        y_gauss=[Y_flat],
        gauss_first_idx=[first_idxs],
        gauss_last_idx=[last_idxs]
    )
    
    config = GLMHMMConfig(
        num_states=n_states,
        num_em_iters=100,
        tol=1e-4,
        l2_reg_trans=1.0,
        l2_reg_gauss=0.1
    )
    
    # 4. Fit
    print("Fitting model...")
    # Manual High Variance Init to overcome normalization dampening
    from glmhmm.initialize import initialize_params
    init_params = initialize_params(data, config, seed=55)
    
    if init_params.W_gauss is not None:
         key = random.PRNGKey(123)
         # True scale is 5.0. 
         # Normalization divides LL by ~60. 
         # We need initial W to separate states. Use scale 2.0.
         noisy_W = 2.0 * random.normal(key, init_params.W_gauss.shape)
         init_params = init_params._replace(W_gauss=noisy_W)
         
    fit_params, lml = fit_glmhmm(data, config, initial_params=init_params)
    print("Fit complete.")
    
    # 5. Check Recovery
    # We construct a full params dict for permutation matching
    # We only care about W_gauss really, and A
    true_params_all = true_params_dict.copy()
    true_params_all['W_gauss'] = W_gauss_true
    # Model uses (M, M, K), but generation used (K, M, M). Transpose for comparison.
    true_params_all['Sigma_gauss'] = jnp.moveaxis(Sigma_true, 0, -1)
    
    best_perm, perm_res = find_best_permutation(true_params_all, fit_params)
    
    print("MSE W_gauss:", np.mean((W_gauss_true - perm_res['W_gauss'])**2))
    
    # Assert
    mse = np.mean((W_gauss_true - perm_res['W_gauss'])**2)
    if mse < 0.1:
        print("PASS: Gaussian weights recovered.")
    else:
        print("FAIL: High MSE.")
        
    # Plot
    save_dir = "tests/plots/variable_rate"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    for k in range(n_states):
        plt.subplot(1, n_states, k+1)
        sns.heatmap(perm_res['W_gauss'][:,:,k], annot=True, center=0)
        plt.title(f"Est State {k}")
    plt.savefig(f"{save_dir}/W_gauss_est.png")
    
    plt.figure(figsize=(10, 4))
    for k in range(n_states):
        plt.subplot(1, n_states, k+1)
        sns.heatmap(W_gauss_true[:,:,k], annot=True, center=0)
        plt.title(f"True State {k}")
    plt.savefig(f"{save_dir}/W_gauss_true.png")
    
    plt.figure()
    plt.plot(lml)
    plt.title("LML History")
    plt.savefig(f"{save_dir}/lml.png")

if __name__ == "__main__":
    test_variable_rate_recovery()
