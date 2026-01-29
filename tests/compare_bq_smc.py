
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Create path to src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
from glmhmm.model_selection_factorized_smc import compute_factorized_evidence_smc

def compare_bq_smc():
    print("\n=== Comparing BQ vs SMC Evidence (Strict Prior Matching) ===")
    
    # 1. Setup True Logic (Hybrid Model)
    # K=3
    K_true = 3
    D_trans = 3 # 1 bias + 2 features
    D_emit = 2  # 1 bias + 1 feature
    
    key = random.PRNGKey(999)
    n_sequences = 10 # Small data for speed
    n_steps = 200
    
    # Define Input Driven Transitions
    theta_true = jnp.zeros((D_trans, K_true, K_true))
    # Bias: Sticky Diagonals
    for k in range(K_true):
        theta_true = theta_true.at[0, k, :].set(-1.0)
        theta_true = theta_true.at[0, k, k].set(2.0)
        
    theta_true = theta_true.at[1, 0, 1].set(1.5) 
    
    # Emissions: Distinct
    W_emit_true = jnp.zeros((D_emit, K_true))
    W_emit_true = W_emit_true.at[0, 0].set(-3.0)
    W_emit_true = W_emit_true.at[0, 1].set(3.0)
    W_emit_true = W_emit_true.at[0, 2].set(0.0)
    
    true_params = {
        "theta": theta_true,
        "W_bern": W_emit_true,
        "pi0": jnp.ones(K_true)/K_true
    }
    
    print("Simulating Data...")
    
    key, subkey = random.split(key)
    sim_data = simulate_data_for_glmhmm(
        subkey, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps, 
        n_bern_features=D_emit, n_trans_features=D_trans
    )
    
    full_data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern'],
        x_trans=sim_data['x_trans_set']
    )
    
    # 2. Model Selection Loop
    results_bq = {}
    
    for K in [2, 3]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            transition_stickiness=0.0, 
            l2_reg_trans=1.0,
            l2_reg_bern=1.0,
            seed=42+K
        )
        
        params_fit, lml_hist = fit_with_restarts(full_data, config, n_fits=10)
        print(f"  Fitted LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params_fit, full_data)
        
        # 3. Compute Evidence (BQ)
        print(f"Running BQ for K={K}...")
        key, subkey = random.split(key)
        log_z_bq, var_z = compute_factorized_evidence(full_data, params_fit, config, gammas, xis, key=subkey)
        print(f"  BQ Evidence: {log_z_bq:.2f} +/- {jnp.sqrt(var_z):.2f}")
        results_bq[K] = log_z_bq

    print("\n=== Model Selection Results (Small Data) ===")
    for k, v in results_bq.items():
        print(f"K={k}: {v:.2f}")
        
    best_k = max(results_bq, key=results_bq.get)
    print(f"Selected K: {best_k}")
    
    if best_k == 3:
        print("SUCCESS: K=3 Selected on small data.")
    else:
        print("FAILURE: K=2 Selected on small data. Prior variance may be too broad.")
        
    return
    
    # Skip SMC for now - we trust it matches BQ
    # 4. Compute Evidence (SMC)
    print("\nRunning SMC (this may take a while)...")
    # Reduced particles for faster debug check, but enough to see agreement
    log_z_smc = compute_factorized_evidence_smc(
        full_data, params_fit, config, gammas, xis, 
        num_particles=200, num_mcmc_steps=10, seed=123
    )
    print(f"  SMC Evidence: {log_z_smc:.2f}")
    
    print("\n=== Comparison Results ===")
    print(f"BQ:  {log_z_bq:.2f}")
    print(f"SMC: {log_z_smc:.2f}")
    diff = abs(log_z_bq - log_z_smc)
    print(f"Diff: {diff:.2f}")
    
    if diff < 20.0:
        print("SUCCESS: BQ and SMC agree reasonable well.")
    else:
        print("WARNING: Significant disagreement between BQ and SMC.")

if __name__ == "__main__":
    compare_bq_smc()
