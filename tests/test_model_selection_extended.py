
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest
import matplotlib.pyplot as plt
import sys
import os

# Create path to src
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_glmhmm
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection import compute_factorized_evidence, compute_bic
from glmhmm.initialize import initialize_params
from glmhmm.fit import e_step

def test_model_selection_negbin():
    print("\n=== Testing Negative Binomial Model Selection ===")
    key = random.PRNGKey(101)
    
    # 1. Simulate NegBin Data (K=3)
    K_true = 3
    D_nb = 2
    n_sequences = 10
    n_steps = 200
    
    A_true = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    pi0 = jnp.ones(3)/3
    
    # W_nb: (D, K). Controls log mean (?) or related.
    # W: (2, 3).
    W_nb_true = jnp.array([
        [2.0, 0.5, -1.0], # Feature 1 diff
        [0.0, 0.0, 0.0]   # Feature 2
    ])
    # phi: (K,). Dispersion.
    phi_nb_true = jnp.array([5.0, 5.0, 5.0])
    
    true_params = {
        "pi0": pi0,
        "A": A_true,
        "W_negbin": W_nb_true,
        "phi_negbin": phi_nb_true
    }
    
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences, n_steps, 
        n_negbin_features=D_nb
    )
    
    data = GLMHMMData(
        x_trans=None,
        x_bern=None, y_bern=None,
        x_multinom=None, y_multinom=None,
        x_negbin=sim_data["x_negbin"], y_negbin=sim_data["y_negbin"],
        x_gamma=None, y_gamma=None,
        x_gauss=None, y_gauss=None,
        gauss_first_idx=None, gauss_last_idx=None
    )
    
    results = {}
    bics = {}
    variances = {}
    
    for K in [2, 3, 4]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            tol=1e-5,
            l2_reg_negbin=300.0,
            l2_reg_trans=30.0
        )
        
        init_params = initialize_params(data, config, seed=K*55)
        params, lml_hist = fit_glmhmm(data, config, initial_params=init_params)
        print(f"  Fit Final LML: {lml_hist[-1]:.2f}")
        
        # BQ
        gammas, xis, _ = e_step(params, data)
        key, subkey = random.split(key)
        log_Z, log_Z_var = compute_factorized_evidence(data, params, config, gammas, xis, key=subkey)
        
        # BIC
        bic = compute_bic(data, params, lml_hist[-1])
        
        results[K] = log_Z
        variances[K] = log_Z_var
        bics[K] = bic
        
        print(f"K={K} | Evidence: {log_Z:.2f} +/- {np.sqrt(log_Z_var):.2f} | BIC: {bic:.2f}")

    best_bq = max(results, key=results.get)
    best_bic = min(bics, key=bics.get)
    
    print("\nNegBin Results:")
    print(f"Selected K (BQ): {best_bq}")
    print(f"Selected K (BIC): {best_bic}")
    
    # Simple Plot
    ks = list(results.keys())
    evs = [results[k] for k in ks]
    errs = [np.sqrt(variances[k]) for k in ks]
    
    plt.figure()
    plt.errorbar(ks, evs, yerr=errs, fmt='-o', label='Evidence')
    plt.title(f'NegBin Model Selection (l2=300.0)')
    plt.savefig('negbin_model_selection.png')

def test_model_selection_gamma():
    print("\n=== Testing Gamma Model Selection ===")
    key = random.PRNGKey(202)
    
    # 2. Simulate Gamma Data (K=3)
    K_true = 3
    D_gm = 2
    n_sequences = 10
    n_steps = 200
    
    A_true = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    pi0 = jnp.ones(3)/3
    
    # W_gamma: (D, K).
    # Mean = exp(X @ W). 
    W_gm_true = jnp.array([
        [0.5, 1.5, 2.5], # Feature 1 drives different means
        [0.0, 0.0, 0.0]
    ])
    # Alpha (Shape): Controls variance. Var = Mean^2 / Alpha
    alpha_gm_true = jnp.array([10.0, 10.0, 10.0]) # Sharp distributions
    
    true_params = {
        "pi0": pi0,
        "A": A_true,
        "W_gamma": W_gm_true,
        "alpha_gamma": alpha_gm_true
    }
    
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences, n_steps, 
        n_gamma_features=D_gm
    )
    
    data = GLMHMMData(
        x_trans=None,
        x_bern=None, y_bern=None,
        x_multinom=None, y_multinom=None,
        x_negbin=None, y_negbin=None,
        x_gamma=sim_data["x_gamma"], y_gamma=sim_data["y_gamma"],
        x_gauss=None, y_gauss=None,
        gauss_first_idx=None, gauss_last_idx=None
    )
    
    results = {}
    bics = {}
    variances = {}
    
    for K in [2, 3, 4]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            tol=1e-5,
            l2_reg_gamma=30.0,
            l2_reg_trans=30.0
        )
        
        init_params = initialize_params(data, config, seed=K*66)
        params, lml_hist = fit_glmhmm(data, config, initial_params=init_params)
        print(f"  Fit Final LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params, data)
        key, subkey = random.split(key)
        log_Z, log_Z_var = compute_factorized_evidence(data, params, config, gammas, xis, key=subkey)
        bic = compute_bic(data, params, lml_hist[-1])
        
        results[K] = log_Z
        variances[K] = log_Z_var
        bics[K] = bic
        
        print(f"K={K} | Evidence: {log_Z:.2f} +/- {np.sqrt(log_Z_var):.2f} | BIC: {bic:.2f}")

    best_bq = max(results, key=results.get)
    best_bic = min(bics, key=bics.get)
    
    print("\nGamma Results:")
    print(f"Selected K (BQ): {best_bq}")
    print(f"Selected K (BIC): {best_bic}")
    
    # Plot
    ks = list(results.keys())
    evs = [results[k] for k in ks]
    errs = [np.sqrt(variances[k]) for k in ks]
    
    plt.figure()
    plt.errorbar(ks, evs, yerr=errs, fmt='-o', label='Evidence')
    plt.title(f'Gamma Model Selection (l2=30.0)')
    plt.savefig('gamma_model_selection.png')

if __name__ == "__main__":
    test_model_selection_negbin()
    test_model_selection_gamma()
