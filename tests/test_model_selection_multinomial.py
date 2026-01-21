
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Ensure src is in path if running directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_glmhmm
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
from glmhmm.initialize import initialize_params
from glmhmm.fit import e_step
from glmhmm import compute_bic

def test_model_selection_multinomial():
    key = random.PRNGKey(42)
    
    # Simulate Multinomial Data
    K_true = 3
    D_mn = 2
    C_mn = 3 # 3 choices
    n_sequences = 10
    n_steps = 200
    
    A_true = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    pi0 = jnp.ones(3)/3
    
    # W shape: (D, C, K)
    W_mn_true = jnp.zeros((D_mn, C_mn, K_true))
    # Bias column (index 0) determines preference
    W_mn_true = W_mn_true.at[0, 0, 0].set(5.0)
    W_mn_true = W_mn_true.at[0, 1, 1].set(5.0)
    W_mn_true = W_mn_true.at[0, 2, 2].set(5.0)
    
    true_params = {
        "pi0": pi0,
        "A": A_true,
        "W_multinom": W_mn_true,
        "y_multinom": jnp.zeros((1, C_mn)) # Dummy
    }
    
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences, n_steps, 
        n_multinom_features=D_mn, n_multinom_choices=C_mn
    )
    
    data = GLMHMMData(
        x_trans=None,
        x_bern=None, y_bern=None,
        x_multinom=sim_data["x_multinom"], y_multinom=sim_data["y_multinom"],
        x_negbin=None, y_negbin=None,
        x_gamma=None, y_gamma=None,
        x_gauss=None, y_gauss=None,
        gauss_first_idx=None, gauss_last_idx=None
    )
    
    print("\nRunning Model Selection Test (Multinomial)...")
    
    results = {}
    variances = {}
    
    # Test K=2, 3, 4
    for K in [2, 3, 4]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            tol=1e-5,
            l2_reg_multinom=1.0,
            l2_reg_trans=1.0
        )
        
        init_params = initialize_params(data, config, seed=K*99)
        
        print("Starting EM...")
        params, lml_hist = fit_glmhmm(data, config, initial_params=init_params)
        print(f"  Fit Final LML (EM): {lml_hist[-1]:.2f} (Epochs: {len(lml_hist)})")
        
        # Evidence
        gammas, xis, _ = e_step(params, data)
        
        key, subkey = random.split(key)
        # Assuming model_selection.py has been updated for Multinomial
        log_Z, log_Z_var = compute_factorized_evidence(data, params, config, gammas, xis, key=subkey)
        
        # BIC & Implied Log Z
        from glmhmm import compute_bic, compute_implied_log_z
        bic = compute_bic(data, params, lml_hist[-1])
        implied_lz = compute_implied_log_z(bic)
        
        results[K] = log_Z
        variances[K] = log_Z_var
        
        print(f"K={K} | Evidence: {log_Z:.2f} +/- {jnp.sqrt(log_Z_var):.2f} | BIC: {bic:.2f} | Implied logZ: {implied_lz:.2f}")

    print("\nFinal Results:")
    ks = []
    lmls = []
    errs = []
    implied_lzs = [] # Store for plotting
    bics = {}
    implied_zs = {}
    
    for K in [2, 3, 4]:
        # ... (same setup) ...
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            tol=1e-5,
            l2_reg_multinom=1.0,
            l2_reg_trans=1.0
        )
        
        init_params = initialize_params(data, config, seed=K*99)
        
        print("Starting EM...")
        params, lml_hist = fit_glmhmm(data, config, initial_params=init_params)
        print(f"  Fit Final LML (EM): {lml_hist[-1]:.2f} (Epochs: {len(lml_hist)})")
        
        # Evidence
        gammas, xis, _ = e_step(params, data)
        
        key, subkey = random.split(key)
        log_Z, log_Z_var = compute_factorized_evidence(data, params, config, gammas, xis, key=subkey)
        
        # BIC
        from glmhmm import compute_bic, compute_implied_log_z
        bic = compute_bic(data, params, lml_hist[-1])
        ilz = compute_implied_log_z(bic)
        
        results[K] = log_Z
        variances[K] = log_Z_var
        bics[K] = bic
        implied_zs[K] = ilz
        
        print(f"K={K} | Evidence: {log_Z:.2f} +/- {jnp.sqrt(log_Z_var):.2f} | BIC: {bic:.2f} | Implied logZ: {ilz:.2f}")

    print("\nFinal Results:")
    ks = []
    lmls = []
    errs = []
    ilzs = []
    
    for k, v in results.items():
        print(f"K={k}: BQ={v:.2f}, BIC={bics[k]:.2f}, Implied={implied_zs[k]:.2f}")
        ks.append(k)
        lmls.append(v)
        errs.append(jnp.sqrt(variances[k]))
        ilzs.append(implied_zs[k])

    best_K = max(results, key=results.get)
    print(f"Selected K (BQ): {best_K}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(ks, lmls, yerr=errs, fmt='-o', capsize=5, label='BQ Log Evidence')
    plt.plot(ks, ilzs, '-s', label='Implied Log Z (BIC)')
    plt.xlabel('Number of States (K)')
    plt.ylabel('Log Evidence')
    plt.title(f'Model Selection (Multinomial, L2={config.l2_reg_multinom})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('multinomial_model_selection_results.png')
    print("Plot saved to multinomial_model_selection_results.png")
    
    assert best_K == 3, "Should recover true K=3 with strong regularization"

if __name__ == "__main__":
    test_model_selection_multinomial()
