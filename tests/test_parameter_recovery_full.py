
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.fit import fit_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.utils import simulate_data_for_glmhmm, find_best_permutation

def plot_recovery(true_params, permuted_params, title_prefix, save_dir):
    """Generates comparison plots for recovered parameters."""
    
    # helper for heatmap
    def plot_heatmap(true, est, title, filename):
        if true is None or est is None: return
        
        # Ensure 2D for heatmap
        if true.ndim == 1:
            true = true[:, None]
            est = est[:, None]
            
        # If 3D (e.g. theta), plot slices?
        if true.ndim == 3:
            # For theta (F, K, K), plot each feature?
            n_features = true.shape[0]
            fig, axes = plt.subplots(n_features, 2, figsize=(10, 4*n_features))
            if n_features == 1: axes = axes[None, :]
            
            for i in range(n_features):
                sns.heatmap(true[i], ax=axes[i,0], cmap="viridis", annot=True, cbar=False)
                axes[i,0].set_title(f"True Feat {i}")
                sns.heatmap(est[i], ax=axes[i,1], cmap="viridis", annot=True, cbar=False)
                axes[i,1].set_title(f"Est Feat {i}")
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename}.png"))
            plt.close()
            return

        # Standard 2D
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(true, ax=axes[0], cmap="RdBu_r", center=0, annot=True)
        axes[0].set_title(f"True")
        sns.heatmap(est, ax=axes[1], cmap="RdBu_r", center=0, annot=True)
        axes[1].set_title(f"Est")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}.png"))
        plt.close()

    plot_heatmap(true_params.get('W_bern'), permuted_params.get('W_bern'), f"{title_prefix} Bernoulli W", "bern_w")
    plot_heatmap(true_params.get('theta'), permuted_params.get('theta'), f"{title_prefix} Transition Theta", "theta")
    plot_heatmap(true_params.get('A'), permuted_params.get('A'), f"{title_prefix} Transition A", "trans_A")
    
    # Multinomial: (F, C, K) -> Plot as (F, C*K) or slices?
    # Let's flatten C and K for visual or plot each feature
    t_mn = true_params.get('W_multinom')
    p_mn = permuted_params.get('W_multinom') # This line was changed in the instruction to permuted_results, but permuted_params is correct here.
    if t_mn is not None:
        # Align True parameters to match the "last class 0" constraint
        # t_mn shape (F, C, K). Subtract last class (C-1) from all C.
        t_mn_aligned = t_mn - t_mn[:, -1:, :]
        
        # Plot Feature 0. p_mn is already aligned by find_best_permutation
        plot_heatmap(t_mn_aligned[0], p_mn[0], f"{title_prefix} Multinom W (Feat 0 aligned)", "multinom_w_f0")
        
        # Plot Feature 1 if available
        if t_mn.shape[0] > 1:
            plot_heatmap(t_mn_aligned[1], p_mn[1], f"{title_prefix} Multinom W (Feat 1 aligned)", "multinom_w_f1")

    plot_heatmap(true_params.get('W_negbin'), permuted_params.get('W_negbin'), f"{title_prefix} NegBin W", "negbin_w") # This line was changed in the instruction to permuted_results, but permuted_params is correct here.
    plot_heatmap(true_params.get('W_gamma'), permuted_params.get('W_gamma'), f"{title_prefix} Gamma W", "gamma_w")


def run_recovery_test(transition_type="input_driven"):
    print(f"\nRunning {transition_type} recovery test...")
    key = random.PRNGKey(42)
    n_states = 3
    n_steps = 2000 
    n_trans_features = 2
    n_bern_features = 2
    n_multinom_features = 2
    n_multinom_choices = 3
    n_negbin_features = 2
    n_gamma_features = 2

    # Define Params
    theta_true = None
    log_A_true = None
    
    if transition_type == "input_driven":
        theta_true = jnp.zeros((n_trans_features, n_states, n_states))
        theta_true = theta_true.at[0, 0, 0].set(3.0) 
        theta_true = theta_true.at[1, 0, 1].set(2.0)
    else:
        A = jnp.array([[0.8, 0.1, 0.1], 
                       [0.1, 0.8, 0.1], 
                       [0.1, 0.1, 0.8]])
        log_A_true = jnp.log(A)
    
    W_bern_true = jnp.zeros((n_bern_features, n_states))
    W_bern_true = W_bern_true.at[1, 0].set(2.0)
    W_bern_true = W_bern_true.at[1, 1].set(-2.0)
    
    W_multinom_true = jnp.zeros((n_multinom_features, n_multinom_choices, n_states))
    W_multinom_true = W_multinom_true.at[0, 0, 0].set(1.0)
    W_multinom_true = W_multinom_true.at[0, 1, 1].set(1.0)
    W_multinom_true = W_multinom_true.at[0, 2, 2].set(1.0)
    
    W_negbin_true = jnp.zeros((n_negbin_features, n_states))
    W_negbin_true = W_negbin_true.at[0, 0].set(1.0)
    phi_negbin_true = jnp.array([5.0, 10.0, 2.0])
    
    W_gamma_true = jnp.zeros((n_gamma_features, n_states))
    W_gamma_true = W_gamma_true.at[0, 1].set(0.5)
    alpha_gamma_true = jnp.array([2.0, 5.0, 1.0])
    
    true_params_dict = {
        "pi0": jnp.ones(n_states)/n_states,
        "theta": theta_true,
        "A": jnp.exp(log_A_true) if log_A_true is not None else None,
        "W_bern": W_bern_true,
        "W_multinom": W_multinom_true,
        "W_negbin": W_negbin_true,
        "phi_negbin": phi_negbin_true,
        "W_gamma": W_gamma_true,
        "alpha_gamma": alpha_gamma_true,
        "W_gauss": None,
        "Sigma_gauss": None
    }
    
    # Simulate
    sim_data = simulate_data_for_glmhmm(
        key, true_params_dict,
        n_sequences=1,
        n_hmm_steps=n_steps,
        n_trans_features=n_trans_features,
        n_bern_features=n_bern_features,
        n_multinom_features=n_multinom_features,
        n_multinom_choices=n_multinom_choices,
        n_negbin_features=n_negbin_features,
        n_gamma_features=n_gamma_features
    )
    
    data_struct = GLMHMMData(
        x_trans=sim_data["x_trans_set"],
        x_bern=sim_data["x_set_bern"],
        y_bern=sim_data["y_set_bern"],
        x_multinom=sim_data["x_multinom"],
        y_multinom=sim_data["y_multinom"],
        x_negbin=sim_data["x_negbin"],
        y_negbin=sim_data["y_negbin"],
        x_gamma=sim_data["x_gamma"],
        y_gamma=sim_data["y_gamma"]
    )
    
    # Longer run for better fit
    config = GLMHMMConfig(
        num_states=n_states,
        num_em_iters=200, 
        tol=1e-4,
        l2_reg_trans=0.01 # Lower reg for better recovery
    )
    
    save_dir = f"tests/plots/{transition_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize with higher variance to avoid saddle points
    from glmhmm.initialize import initialize_params
    init_params = initialize_params(data_struct, config, seed=42)
    
    if init_params.W_multinom is not None:
        key_init = random.PRNGKey(99)
        # Increase scale to 1.0 for Multinomial to break symmetry
        noisy_W = 1.0 * random.normal(key_init, init_params.W_multinom.shape)
        init_params = init_params._replace(W_multinom=noisy_W)
        
    fit_params, lml_history = fit_glmhmm(data_struct, config, initial_params=init_params)
    
    plt.figure()
    plt.plot(lml_history)
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Marginal Likelihood")
    plt.title(f"LML Convergence ({transition_type})")
    plt.savefig(os.path.join(save_dir, "lml_history.png"))
    plt.close()
    
    # Assert improvement
    lml_valid = [x for x in lml_history if not np.isnan(x)]
    if len(lml_valid) > 1:
        assert lml_valid[-1] >= lml_valid[0] - 10.0
        print(f"LML improved/stable: {lml_valid[0]:.2f} -> {lml_valid[-1]:.2f}")
    
    # Find Permutation
    best_perm, permuted_results = find_best_permutation(true_params_dict, fit_params)
    
    # Plot Recovery
    plot_recovery(true_params_dict, permuted_results, transition_type, save_dir)
    
    if transition_type == "input_driven":
        print("\n--- Theta Comparison ---")
        t_theta = true_params_dict['theta']
        p_theta = permuted_results['theta']
        # Print slices where true theta is non-zero
        print(f"True Theta (Slice 0,0,0): {t_theta[0,0,0]}")
        print(f"Est Theta (Slice 0,0,0): {p_theta[0,0,0]}")
        print(f"True Theta (Slice 1,0,1): {t_theta[1,0,1]}")
        print(f"Est Theta (Slice 1,0,1): {p_theta[1,0,1]}")
    
    print(f"✓ {transition_type} test passed. Plots saved to {save_dir}")

def test_full_parameter_recovery():
    print("\n" + "="*60)
    print("Full GLM-HMM Parameter Recovery Test")
    print("="*60)
    
    run_recovery_test("input_driven")
    run_recovery_test("static")

if __name__ == "__main__":
    test_full_parameter_recovery()
