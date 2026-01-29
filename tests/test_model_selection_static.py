
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from glmhmm.structs import GLMHMMConfig, GLMHMMData
from glmhmm.initialize import initialize_params
from glmhmm.fit import fit_glmhmm
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
from glmhmm.model_selection_factorized_smc import compute_factorized_evidence_smc

def test_static_transition_recovery():
    """
    Verifies that we can recover the true K=3 model when transitions are static,
    using both BQ and SMC.
    """
    key = jax.random.PRNGKey(42)
    
    # 1. Generate Data (K=3, Static Transitions)
    N, T = 80, 200  
    K_true = 3
    
    # Static transitions: A matrix
    A_true = jnp.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    
    # Emissions (Gaussian)
    # State 0: Mean -2, State 1: Mean 0, State 2: Mean 2
    W_gauss_true = jnp.array([
        [[-2.0, 0.0, 2.0]] # Biases (D=1, M=1, K=3)
    ]) 
    # Create true params dict
    true_params = {
        "pi0": jnp.array([1/3, 1/3, 1/3]),
        "A": A_true,
        "W_gauss": W_gauss_true,
        "Sigma_gauss": jnp.tile(jnp.eye(1)[:, :, None] * 0.25, (1, 1, 3)) # Sigma=0.25 (std=0.5)
    }

    from glmhmm.utils import simulate_data_for_glmhmm
    
    print("Simulating Data...")
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences=N, n_hmm_steps=T, 
        n_gauss_features=1, # 1 bias
        n_trans_features=0, # Static
        samples_per_gaussian_segment=1
    )

    # sim_data['x_nested_set'] is list of lists of arrays
    # We need to flatten the inner list maybe? 
    # simulate_data_for_glmhmm returns x_nested_set as list of lists of arrays (samples_per_segment x D)
    # But here we have 1 sample per segment (T segments of length 1)? 
    # Or just T samples?
    # Actually simulate_data_for_glmhmm loops T times.
    # checking code: x_gauss_nested_trials.append(x_gauss_trial)
    # So it returns list (sequences) of list (time steps) of arrays.
    
    # We need to convert this to what fit_glmhmm expects.
    # For Gaussian, fit_glmhmm expects list of arrays (if no segments).
    # Or list of lists.
    # Let's concatenate the time steps for each sequence.
    
    X_train = []
    Y_train = []
    for i in range(N):
        # Concatenate all T steps
        x_seq = jnp.concatenate(sim_data['x_nested_set'][i], axis=0) # (T, 1)
        y_seq = jnp.concatenate(sim_data['y_nested_set'][i], axis=0) # (T, 1)
        X_train.append(x_seq)
        Y_train.append(y_seq)
         
    data = GLMHMMData(
        x_gauss=X_train,
        y_gauss=Y_train,
        x_trans=None # Static
    )
    
    print(f"Generated data: N={N}, T={T}, K={K_true}")
    
    Y_flat = np.concatenate([np.array(y) for y in Y_train])
    print(f"Data Mean: {np.mean(Y_flat):.2f}")
    print(f"Data Std: {np.std(Y_flat):.2f}")
    print(f"Data [Min, Max]: [{np.min(Y_flat):.2f}, {np.max(Y_flat):.2f}]")

    # 2. Fit Models (K=2, 3, 4)
    results = {}
    
    for K in [2, 3, 4]:
        print(f"\nTraining K={K}...")
        
        # Config
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=200, 
            transition_stickiness=0.0, 
            prior_var_bias_trans=2.0,
            prior_var_bias_emit=2.0
        )
        
        # Init
        key, subk = jax.random.split(key)
        params = initialize_params(data, config, seed=42)
        
        if params.W_gauss is not None:
             print(f"Initialized Means K={K}:\n{params.W_gauss.squeeze()}")
        
        # Fit
        final_params, final_lls = fit_glmhmm(
            data, config, initial_params=params
        )

        # Debug: Check if means and sigma are learned correctly
        if final_params.W_gauss is not None:
             print(f"Learned Means K={K}:\n{final_params.W_gauss.squeeze()}")
             print(f"Learned Sigma K={K}:\n{final_params.Sigma_gauss.squeeze()}")

        
        # Recover posteriors
        from glmhmm.fit import e_step
        gammas, xis, _ = e_step(final_params, data)
        
        # 3. Compute Evidence
        print(f"Computing BQ Evidence K={K}...")
        bq_evidence, _ = compute_factorized_evidence(data, final_params, config, gammas, xis)
        
        print(f"Computing SMC Evidence K={K}...")
        smc_evidence = compute_factorized_evidence_smc(data, final_params, config, gammas, xis, num_particles=100, num_mcmc_steps=10) # Low particles for speed in test
        
        results[K] = {
            'bq': bq_evidence,
            'smc': smc_evidence
        }
        print(f"K={K}: BQ={bq_evidence:.2f}, SMC={smc_evidence:.2f}")

    # 4. Verification
    bq_best_k = max(results, key=lambda k: results[k]['bq'])
    smc_best_k = max(results, key=lambda k: results[k]['smc'])
    
    print(f"\nSelected K (BQ): {bq_best_k}")
    print(f"Selected K (SMC): {smc_best_k}")
    
    assert bq_best_k == K_true, f"BQ failed to select K={K_true}"
    assert smc_best_k == K_true, f"SMC failed to select K={K_true}"
    
    # 5. Check Consistency of Static Transition Evidence
    # We can't easily isolate just the transition evidence from the return values of the wrapper functions
    # without modifying them or copying code. 
    # BUT, the task implies ensuring they treat it similarly. 
    # The BQ function prints "log_Z_trans". We can capture that or just rely on the implementation change.
    # For now, let's just make sure the total evidence is reasonably close, 
    # but the primary goal of this test before the fix is to SHOW the discrepancy 
    # (or if they are close enough, just that they work).
    # After the fix, they should be VERY close.
    
    diff = abs(results[3]['bq'] - results[3]['smc'])
    print(f"K=3 Evidence Diff: {diff:.2f}")
    
    # Consistency Assertion
    assert diff < 2.0, f"BQ and SMC evidence differed by {diff:.2f}, expected < 2.0"
    
    # If the fix works, BQ and SMC should effectively be identical for the Transition part.
    # The Emissions part is still different (Analytic/BQ vs SMC). 
    # So we can't expect EXACT equality, but we expect better alignment.
    
if __name__ == "__main__":
    test_static_transition_recovery()
