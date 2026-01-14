
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
from glmhmm.utils import simulate_data_for_glmhmm, find_best_permutation, compute_boundaries
from glmhmm.initialize import initialize_params

def test_joint_recovery_static():
    print("Running Joint Recovery Test (Bernoulli + Variable Rate Gaussian) - Static Transitions")
    key = random.PRNGKey(55)
    n_states = 3
    n_steps = 1000 # Increase steps for better joint recovery
    n_sequences = 1
    
    # 1. Define True Parameters
    # -------------------------
    # Static Transitions
    A_true = jnp.array([[0.8, 0.1, 0.1], 
                        [0.1, 0.8, 0.1], 
                        [0.1, 0.1, 0.8]])
    pi0 = jnp.ones(n_states)/n_states
    
    # Bernoulli (Bias + 1 Stimulus)
    D_b = 2
    W_bern_true = jnp.zeros((D_b, n_states))
    # State 0: Prefer 0 (Bias -2)
    W_bern_true = W_bern_true.at[0, 0].set(-2.0)
    # State 1: Prefer 1 (Bias +2)
    W_bern_true = W_bern_true.at[0, 1].set(2.0)
    # State 2: Stimulus driven (Bias 0, Stim +3)
    W_bern_true = W_bern_true.at[1, 2].set(3.0)
    
    # Gaussian (Mean only, Identity Sigma)
    # 2 Features (Bias + 1 Stimulus) -> 2 Neural outputs
    D_g = 2 
    M_g = 2
    W_gauss_true = jnp.zeros((D_g, M_g, n_states))
    
    # State 0: High activity (Bias +5)
    W_gauss_true = W_gauss_true.at[0, :, 0].set(5.0)
    # State 1: Low activity (Bias -5)
    W_gauss_true = W_gauss_true.at[0, :, 1].set(-5.0)
    # State 2: Stimulus driven (Stim +5)
    W_gauss_true = W_gauss_true.at[1, :, 2].set(5.0)
    
    Sigma_true = jnp.tile(jnp.eye(M_g)[:, :, None], (1, 1, n_states))
    
    true_params = {
        "pi0": pi0,
        "A": A_true,
        "W_bern": W_bern_true,
        "W_gauss": W_gauss_true,
        "Sigma_gauss": Sigma_true # Generation expects (M, M, K)
    }
    
    # 2. Simulate Data
    # ----------------
    # Variable rates for Gaussian: 10 to 30 samples per steps
    key, subkey = random.split(key)
    samples_per_step = random.randint(subkey, (n_steps,), 10, 30).tolist()
    
    print(f"Simulating {n_steps} steps with variable Gaussian rates...")
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences, n_steps,
        n_bern_features=D_b,
        n_gauss_features=D_g,
        n_regions=float(M_g), # Hack: pass float? check implementation. It's unused in sim, used in fit. 
        # Actually simulate_data_for_glmhmm signature line 205 says n_regions: int. 
        # But it doesn't seem to use it for generation logic, just passing checks?
        # Let's check logic: checks "True W_gauss".
        samples_per_gaussian_segment=samples_per_step
    )
    
    # 3. Pack Data
    # ------------
    # Bernoulli
    x_b = sim_data["x_set_bern"]
    y_b = sim_data["y_set_bern"]
    
    # Gaussian (Ragged)
    # sim_data["x_nested_set"] is list of lists of arrays? 
    # check utils.py: all_x_gauss_nested_set.append(x_gauss_nested_trials) -> List[List[Array]]
    # We have 1 sequence.
    x_g_nested = sim_data["x_nested_set"][0] # List of T arrays
    y_g_nested = sim_data["y_nested_set"][0] # List of T arrays
    
    # We need to flatten this for GLMHMMData x_gauss (List of arrays, one per seq)
    # Wait, GLMHMMData x_gauss is List[Array]. For one sequence, it should be [Big_Array]
    # And we need indices.
    
    X_flat = jnp.concatenate(x_g_nested, axis=0)
    Y_flat = jnp.concatenate(y_g_nested, axis=0)
    
    # Compute indices
    # We can reconstruct from samples_per_step
    counts = jnp.array(samples_per_step)
    last_idx = jnp.cumsum(counts)
    first_idx = last_idx - counts
    
    data = GLMHMMData(
        x_trans=None,
        x_bern=x_b, y_bern=y_b,
        x_multinom=None, y_multinom=None,
        x_negbin=None, y_negbin=None,
        x_gamma=None, y_gamma=None,
        x_gauss=[X_flat],
        y_gauss=[Y_flat],
        gauss_first_idx=[first_idx],
        gauss_last_idx=[last_idx]
    )
    
    config = GLMHMMConfig(
        num_states=n_states,
        num_em_iters=100,
        tol=1e-4,
        l2_reg_trans=1.0,
        l2_reg_bern=1.0,
        l2_reg_gauss=0.1 # Lower reg for Gaussian
    )
    
    # 4. Fit
    # ------
    print("Fitting model...")
    # Use robust initialization for Gaussian
    init_params = initialize_params(data, config, seed=42)
    if init_params.W_gauss is not None:
         k = random.PRNGKey(999)
         # High variance init
         noisy_W = 2.0 * random.normal(k, init_params.W_gauss.shape)
         init_params = init_params._replace(W_gauss=noisy_W)
         
    fit_params, lml = fit_glmhmm(data, config, initial_params=init_params)
    print(f"Fit complete. Final LML: {lml[-1]:.2f}")
    
    # 5. Evaluate & Plot
    # ------------------
    save_dir = "tests/plots/joint_static"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare True Params dict for Permutation
    # Transpose Sigma back to (M, M, K) for comparison
    true_params_cmp = true_params.copy()
    true_params_cmp['Sigma_gauss'] = Sigma_true
    
    best_perm, perm_res = find_best_permutation(true_params_cmp, fit_params)
    
    # Plot LML
    plt.figure()
    plt.plot(lml)
    plt.title("Joint LML History")
    plt.xlabel("Epoch")
    plt.ylabel("LML")
    plt.savefig(f"{save_dir}/lml.png")
    plt.close()
    
    # Plot A
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(A_true, ax=axes[0], annot=True, cmap="Blues")
    axes[0].set_title("True A")
    sns.heatmap(perm_res['A'], ax=axes[1], annot=True, cmap="Blues")
    axes[1].set_title("Est A")
    plt.savefig(f"{save_dir}/A.png")
    plt.close()
    
    # Plot W_bern
    # (D, K)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(W_bern_true, ax=axes[0], annot=True, center=0, cmap="coolwarm")
    axes[0].set_title("True W_bern")
    sns.heatmap(perm_res['W_bern'], ax=axes[1], annot=True, center=0, cmap="coolwarm")
    axes[1].set_title("Est W_bern")
    plt.savefig(f"{save_dir}/W_bern.png")
    plt.close()
    
    # Plot W_gauss
    # (D, M, K) -> Let's plot D x K for each M (output dim)
    # Or just flatten?
    # Let's do side-by-side for each state (K)
    for k in range(n_states):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(W_gauss_true[:, :, k], ax=axes[0], annot=True, center=0, cmap="coolwarm")
        axes[0].set_title(f"True W_gauss (State {k})")
        
        sns.heatmap(perm_res['W_gauss'][:, :, k], ax=axes[1], annot=True, center=0, cmap="coolwarm")
        axes[1].set_title(f"Est W_gauss (State {k})")
        
        plt.savefig(f"{save_dir}/W_gauss_k{k}.png")
        plt.close()
        
    # Plot Sigma_gauss
    # (M, M, K)
    for k in range(n_states):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(Sigma_true[:, :, k], ax=axes[0], annot=True, center=0, cmap="viridis")
        axes[0].set_title(f"True Sigma (State {k})")
        
        sns.heatmap(perm_res['Sigma_gauss'][:, :, k], ax=axes[1], annot=True, center=0, cmap="viridis")
        axes[1].set_title(f"Est Sigma (State {k})")
        
        plt.savefig(f"{save_dir}/Sigma_gauss_k{k}.png")
        plt.close()

    print("Plots saved to", save_dir)
    pass_bern = np.mean((W_bern_true - perm_res['W_bern'])**2) < 0.2
    pass_gauss = np.mean((W_gauss_true - perm_res['W_gauss'])**2) < 0.2
    pass_sigma = np.mean((Sigma_true - perm_res['Sigma_gauss'])**2) < 0.1
    
    if pass_bern and pass_gauss and pass_sigma:
        print("PASS: Recovered Bernoulli, Gaussian Weights, and Covariance.")
    else:
        print(f"FAIL: MSE Bern: {np.mean((W_bern_true - perm_res['W_bern'])**2):.4f}, MSE Gauss W: {np.mean((W_gauss_true - perm_res['W_gauss'])**2):.4f}, MSE Gauss Sigma: {np.mean((Sigma_true - perm_res['Sigma_gauss'])**2):.4f}")

if __name__ == "__main__":
    test_joint_recovery_static()
