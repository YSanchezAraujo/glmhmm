
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# Create path to src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence

def test_final_verification_simple():
    print("\n=== Final Verification: Simple (3 States, Distinct Emissions) ===")
    
    K_true = 3
    D_emit = 1 # Just bias
    
    key = random.PRNGKey(42)
    n_sequences = 5
    n_steps = 1000
    
    # 1. Setup Static Transition (Simpler than input driven for now to isolate BQ)
    #    If this works, BQ core is fine.
    
    pi0_true = jnp.array([1.0, 0.0, 0.0])
    A_true = jnp.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    
    # Emissions: Very distinct Bernoulli
    # State 0: p=0.1
    # State 1: p=0.9
    # State 2: p=0.5
    W_emit_true = jnp.zeros((1, K_true))
    W_emit_true = W_emit_true.at[0, 0].set(-2.2) # p ~ 0.1
    W_emit_true = W_emit_true.at[0, 1].set(2.2)  # p ~ 0.9
    W_emit_true = W_emit_true.at[0, 2].set(0.0)  # p ~ 0.5
    
    true_params = {
        "pi0": pi0_true,
        "A": A_true,
        "W_bern": W_emit_true
    }
    
    print("Simulating Data...")
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps, 
        n_bern_features=D_emit
    )
    
    full_data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern']
    )
    
    # 2. Model Selection Loop
    K_candidates = [2, 3]
    results_bq = {}
    
    for K in K_candidates:
        print(f"\n--- Testing K={K} ---")
        
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=200,
            transition_stickiness=10.0, # Use static stickiness
            l2_reg_bern=1.0,
            seed=100+K
        )
        
        params_fit, lml_hist = fit_with_restarts(full_data, config, n_fits=3)
        print(f"  Fitted LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params_fit, full_data)
        
        key, subkey = random.split(key)
        # Assuming compute_factorized_evidence handles static stickiness now
        log_z_bq, var_z = compute_factorized_evidence(full_data, params_fit, config, gammas, xis, key=subkey)
        results_bq[K] = log_z_bq
        print(f"  BQ Evidence: {log_z_bq:.2f} +/- {jnp.sqrt(var_z):.2f}")

    best_k = max(results_bq, key=results_bq.get)
    print("\n=== Results ===")
    print(f"True K: {K_true}")
    print(f"Selected K: {best_k}")
    for k in K_candidates:
        print(f"K={k}: {results_bq[k]:.2f}")
        
    assert best_k == K_true, f"Model Selection Failed. Got {best_k}"

if __name__ == "__main__":
    test_final_verification_simple()
