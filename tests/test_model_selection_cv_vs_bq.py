
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Create path to src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_glmhmm, fit_with_restarts, compute_log_marginal_likelihood
from glmhmm.utils import simulate_train_test_data
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
from glmhmm.initialize import initialize_params
from glmhmm.fit import e_step

def test_cv_vs_bq():
    print("\n=== Comparison: Cross-Validation vs BQ Evidence ===")
    
    # Setup
    key = random.PRNGKey(142)
    K_true = 3
    D = 2
    n_train = 25 # Optimized size
    n_test = 5 
    n_steps = 250 # Reduced for speed
    
    # Generate Data
    print(f"Generating data with K_true={K_true}...")
    key, subkey = random.split(key)
    train_data, test_data, full_data, true_params = simulate_train_test_data(
        subkey, K=K_true, D=D, n_train=n_train, n_test=n_test, n_steps=n_steps
    )
    
    results_cv = {}
    results_bq = {}
    bq_errs = {}
    results_bic = {}
    results_hist_cv = {}
    results_hist_bq = {}
    
    K_candidates = [2, 3, 4]
    
    from glmhmm import compute_bic
    
    for K in K_candidates:
        print(f"\n--- Testing K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=150, # Reduced for speed
            tol=1e-3,
            l2_reg_bern=1.0,
            l2_reg_trans=1.0, # Regularization helps prevention of overfitting
            init_noise_scale=1.0,
            seed=42+K # Base seed for this K
        )
        
        # 1. Cross-Validation (Train -> Test)
        print("  Running Cross-Validation (with restarts)...")
        # init_params_train = initialize_params(train_data, config, seed=2+K*10)
        # params_train, lml_hist_train = fit_glmhmm(train_data, config, initial_params=init_params_train)
        params_train, lml_hist_train = fit_with_restarts(train_data, config, n_fits=2)
        
        # Evaluate on Test Data
        # P(Y_test | params_train)
        test_ll = compute_log_marginal_likelihood(params_train, test_data)
        results_cv[K] = test_ll
        print(f"    Train LML: {lml_hist_train[-1]:.2f} | Test LL: {test_ll:.2f}")
        
        # 2. BQ & BIC (Full Data)
        print("  Running BQ/BIC on Full Data (with restarts)...")
        # init_params_full = initialize_params(full_data, config, seed=K*20)
        # params_full, lml_hist_full = fit_glmhmm(full_data, config, initial_params=init_params_full)
        params_full, lml_hist_full = fit_with_restarts(full_data, config, n_fits=2)
        
        # Compute BQ
        gammas, xis, _ = e_step(params_full, full_data)
        key, subkey = random.split(key)
        log_Z, log_Z_var = compute_factorized_evidence(full_data, params_full, config, gammas, xis, key=subkey)
        
        # Compute BIC
        bic = compute_bic(full_data, params_full, lml_hist_full[-1])
        
        results_bq[K] = log_Z
        bq_errs[K] = jnp.sqrt(log_Z_var)
        results_bic[K] = bic
        
        print(f"    Full LML: {lml_hist_full[-1]:.2f} | BQ Evidence: {log_Z:.2f} | BIC: {bic:.2f}")

        # Store Histories
        results_hist_cv[K] = lml_hist_train
        results_hist_bq[K] = lml_hist_full

    # Results Analysis
    best_cv = max(results_cv, key=results_cv.get)
    best_bq = max(results_bq, key=results_bq.get)
    best_bic = min(results_bic, key=results_bic.get)
    
    print("\n=== Final Results ===")
    print(f"True K: {K_true}")
    print(f"Selected K (CV):  {best_cv}")
    print(f"Selected K (BQ):  {best_bq}")
    print(f"Selected K (BIC): {best_bic}")
    
    # Plotting
    ks = K_candidates
    cv_scores = [results_cv[k] for k in ks]
    bq_scores = [results_bq[k] for k in ks]
    bic_scores = [-results_bic[k]/2 for k in ks] # Plotting -BIC/2 (approx log evidence)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot CV
    color = 'tab:blue'
    ax1.set_xlabel('Number of States (K)')
    ax1.set_ylabel('Test Log-Likelihood (CV)', color=color)
    ax1.plot(ks, cv_scores, '-o', color=color, label='CV Test LL')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot BQ
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Log Model Evidence (BQ)', color=color)
    ax2.errorbar(ks, bq_scores, yerr=[bq_errs[k] for k in ks], fmt='-s', color=color, label='BQ Evidence')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # BIC as dashed
    # Normalize BIC to align roughly with BQ for visualization (or just plot raw)
    # BQ is roughly -BIC/2 + constants. 
    # Let's plot -BIC/2 on the BQ axis
    ax2.plot(ks, bic_scores, '--', color='green', alpha=0.5, label='Implied Log Z (BIC)')

    plt.title(f'Model Selection Comparison (True K={K_true})')
    fig.tight_layout()
    plt.savefig('comparison_cv_vs_bq.png')
    print("Plot saved to comparison_cv_vs_bq.png")

    # Plot Convergence Histories
    fig_conv, axs = plt.subplots(2, len(K_candidates), figsize=(15, 8))
    # Handle single K case mostly for debugging/robustness
    if len(K_candidates) == 1:
        axs = np.array([axs]).T 
        
    for i, K in enumerate(K_candidates):
        # CV (Train)
        axs[0, i].plot(results_hist_cv[K])
        axs[0, i].set_title(f'CV Train LML (K={K})')
        axs[0, i].set_xlabel('EM Iteration')
        axs[0, i].grid(True)
        
        # BQ (Full)
        axs[1, i].plot(results_hist_bq[K], color='tab:orange')
        axs[1, i].set_title(f'Full Data LML (K={K})')
        axs[1, i].set_xlabel('EM Iteration')
        axs[1, i].grid(True)
        
    plt.tight_layout()
    plt.savefig('convergence_history.png')
    print("Convergence plot saved to convergence_history.png")

    
if __name__ == "__main__":
    test_cv_vs_bq()
