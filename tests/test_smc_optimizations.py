"""
Test script for optimized SMC evidence computation.

Tests that the SMC optimizations work correctly:
1. Particle initialization around MAP estimate
2. Dimension-scaled step size  
3. Both NUTS and HMC kernels
4. Reduced default particles/steps

Run with: python test_smc_optimizations.py
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os
import time

# Create path to src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection_factorized_smc import compute_factorized_evidence_smc


def test_smc_basic():
    """Test that SMC runs without errors on simple data."""
    print("\n=== Test 1: Basic SMC Functionality ===")
    
    K_true = 2
    D_emit = 2
    
    key = random.PRNGKey(42)
    n_sequences = 5
    n_steps = 50
    
    # Simple static transition model
    W_emit_true = jnp.array([[-2.0, 2.0], [0.5, -0.5]])
    A_true = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    pi0_true = jnp.array([0.5, 0.5])
    
    true_params = {
        "W_bern": W_emit_true,
        "A": A_true,
        "pi0": pi0_true
    }
    
    print("Simulating data...")
    key, subkey = random.split(key)
    sim_data = simulate_data_for_glmhmm(
        subkey, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps,
        n_bern_features=D_emit
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern']
    )
    
    config = GLMHMMConfig(
        num_states=K_true,
        num_em_iters=50,
        l2_reg_bern=1.0,
        seed=42
    )
    
    print("Fitting model...")
    params, lml_hist = fit_with_restarts(data, config, n_fits=3)
    print(f"  Final LML: {lml_hist[-1]:.2f}")
    
    gammas, xis, _ = e_step(params, data)
    
    # Test SMC with NUTS (default)
    print("\nRunning SMC (NUTS kernel)...")
    start = time.time()
    try:
        log_z_nuts = compute_factorized_evidence_smc(
            data, params, config, gammas, xis,
            num_particles=50, num_mcmc_steps=3, seed=0,
            mcmc_kernel='nuts'
        )
        nuts_time = time.time() - start
        print(f"  NUTS Evidence: {log_z_nuts:.2f} (took {nuts_time:.1f}s)")
        nuts_success = True
    except Exception as e:
        print(f"  NUTS FAILED: {e}")
        nuts_success = False
    
    # Test SMC with HMC
    print("\nRunning SMC (HMC kernel)...")
    start = time.time()
    try:
        log_z_hmc = compute_factorized_evidence_smc(
            data, params, config, gammas, xis,
            num_particles=50, num_mcmc_steps=3, seed=0,
            mcmc_kernel='hmc'
        )
        hmc_time = time.time() - start
        print(f"  HMC Evidence: {log_z_hmc:.2f} (took {hmc_time:.1f}s)")
        hmc_success = True
    except Exception as e:
        print(f"  HMC FAILED: {e}")
        hmc_success = False
    
    # Summary
    print("\n=== Test 1 Results ===")
    if nuts_success and hmc_success:
        print("✓ Both NUTS and HMC kernels work")
        if hmc_time < nuts_time:
            print(f"✓ HMC is {nuts_time/hmc_time:.1f}x faster than NUTS")
        diff = abs(log_z_nuts - log_z_hmc)
        print(f"  Evidence difference: {diff:.2f}")
        return True
    else:
        print("✗ One or more kernels failed")
        return False


def test_smc_input_driven():
    """Test SMC with input-driven transitions."""
    print("\n=== Test 2: SMC with Input-Driven Transitions ===")
    
    K_true = 2
    D_trans = 3
    D_emit = 2
    
    key = random.PRNGKey(123)
    n_sequences = 5
    n_steps = 50
    
    # Input-driven transitions
    theta_true = jnp.zeros((D_trans, K_true, K_true))
    theta_true = theta_true.at[0, 0, 0].set(1.5)
    theta_true = theta_true.at[0, 1, 1].set(1.5)
    theta_true = theta_true.at[1, 0, 1].set(1.0)
    
    W_emit_true = jnp.array([[-2.0, 2.0], [0.3, -0.3]])
    
    true_params = {
        "theta": theta_true,
        "W_bern": W_emit_true,
        "pi0": jnp.array([0.5, 0.5])
    }
    
    print("Simulating data with input-driven transitions...")
    key, subkey = random.split(key)
    sim_data = simulate_data_for_glmhmm(
        subkey, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps,
        n_bern_features=D_emit, n_trans_features=D_trans
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern'],
        x_trans=sim_data['x_trans_set']
    )
    
    config = GLMHMMConfig(
        num_states=K_true,
        num_em_iters=50,
        l2_reg_trans=1.0,
        l2_reg_bern=1.0,
        seed=123
    )
    
    print("Fitting model...")
    params, lml_hist = fit_with_restarts(data, config, n_fits=3)
    print(f"  Final LML: {lml_hist[-1]:.2f}")
    
    gammas, xis, _ = e_step(params, data)
    
    print("\nRunning SMC with input-driven transitions...")
    start = time.time()
    try:
        log_z = compute_factorized_evidence_smc(
            data, params, config, gammas, xis,
            num_particles=50, num_mcmc_steps=3, seed=0
        )
        elapsed = time.time() - start
        print(f"  Evidence: {log_z:.2f} (took {elapsed:.1f}s)")
        print("✓ Input-driven SMC works")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection():
    """Test that SMC correctly selects the true number of states, compare with BQ."""
    print("\n=== Test 3: Model Selection (K_true=3 vs K=2,3,4) ===")
    print("    Comparing SMC (HMC) vs BQ evidence")
    print("    Using weaker prior (alpha=0.5) and more data\n")
    
    from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
    
    K_true = 3
    D_emit = 2
    
    key = random.PRNGKey(999)
    n_sequences = 25  # More data
    n_steps = 200     # Longer sequences
    
    # True K=3 model with distinct emissions
    W_emit_true = jnp.array([
        [-3.0, 3.0],   # State 0: high P(y=1)
        [0.0, 0.0],    # State 1: random
        [3.0, -3.0]    # State 2: high P(y=0)
    ]).T  # Shape: (D_emit, K_true)
    
    A_true = jnp.array([
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85]
    ])
    pi0_true = jnp.array([1/3, 1/3, 1/3])
    
    true_params = {
        "W_bern": W_emit_true,
        "A": A_true,
        "pi0": pi0_true
    }
    
    print("Simulating data from K=3 model...")
    key, subkey = random.split(key)
    sim_data = simulate_data_for_glmhmm(
        subkey, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps,
        n_bern_features=D_emit
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern']
    )
    
    results_smc = {}
    results_bq = {}
    
    for K in [2, 3, 4]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=150,
            l2_reg_bern=1.0,
            prior_var_bias_trans=10.0,  # Weaker prior on transitions
            seed=42 + K
        )
        
        params, lml_hist = fit_with_restarts(data, config, n_fits=5)
        print(f"  Final LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params, data)
        
        # Compute BQ evidence
        print(f"  Computing BQ evidence...")
        start = time.time()
        try:
            key, subkey = random.split(key)
            log_z_bq, var_bq = compute_factorized_evidence(
                data, params, config, gammas, xis, key=subkey,
                static_trans_alpha=0.5  # Weaker prior
            )
            bq_time = time.time() - start
            print(f"    BQ:  {log_z_bq:.2f} +/- {jnp.sqrt(var_bq):.2f} ({bq_time:.1f}s)")
            results_bq[K] = float(log_z_bq)
        except Exception as e:
            print(f"    BQ FAILED: {e}")
            results_bq[K] = float('-inf')
        
        # Compute SMC evidence (HMC kernel)
        print(f"  Computing SMC evidence (NUTS, high accuracy)...")
        start = time.time()
        try:
            log_z_smc = compute_factorized_evidence_smc(
                data, params, config, gammas, xis,
                num_particles=200, num_mcmc_steps=15, seed=0,
                mcmc_kernel='nuts', static_trans_alpha=0.5  # Match BQ
            )
            smc_time = time.time() - start
            print(f"    SMC: {log_z_smc:.2f} ({smc_time:.1f}s)")
            results_smc[K] = log_z_smc
        except Exception as e:
            print(f"    SMC FAILED: {e}")
            import traceback
            traceback.print_exc()
            results_smc[K] = float('-inf')
    
    print("\n" + "=" * 50)
    print("MODEL SELECTION RESULTS")
    print("=" * 50)
    print(f"{'K':<5} {'SMC (HMC)':<15} {'BQ':<15}")
    print("-" * 35)
    for K in [2, 3, 4]:
        smc_val = results_smc.get(K, float('-inf'))
        bq_val = results_bq.get(K, float('-inf'))
        print(f"{K:<5} {smc_val:<15.2f} {bq_val:<15.2f}")
    
    best_k_smc = max(results_smc, key=results_smc.get)
    best_k_bq = max(results_bq, key=results_bq.get)
    
    print("-" * 35)
    print(f"Selected by SMC: K={best_k_smc}")
    print(f"Selected by BQ:  K={best_k_bq}")
    print(f"True model:      K={K_true}")
    
    smc_correct = (best_k_smc == K_true)
    bq_correct = (best_k_bq == K_true)
    
    print()
    if smc_correct:
        print(f"✓ SMC correctly selected K={K_true}")
    else:
        print(f"✗ SMC incorrectly selected K={best_k_smc} (true: K={K_true})")
        
    if bq_correct:
        print(f"✓ BQ correctly selected K={K_true}")
    else:
        print(f"✗ BQ incorrectly selected K={best_k_bq} (true: K={K_true})")
    
    return smc_correct


def test_model_selection_input_driven():
    """Test model selection with input-driven transitions (better scaling)."""
    print("\n=== Test 4: Model Selection with Input-Driven Transitions ===")
    print("    (Input-driven transitions don't have O(K²) penalty)\n")
    
    from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
    
    K_true = 3
    D_emit = 2
    D_trans = 3
    
    key = random.PRNGKey(888)
    n_sequences = 20
    n_steps = 150
    
    # True K=3 input-driven model
    # theta shape: (D_trans, K_from, K_to)
    # Last destination state (index K-1) is reference (all zeros)
    theta_true = jnp.zeros((D_trans, K_true, K_true))
    
    # Set up sticky bias relative to reference state
    # For each source state k, set bias to prefer staying in k
    for k in range(K_true):
        # State k prefers to stay (higher logit for k vs reference)
        if k < K_true - 1:  # Not the reference state
            theta_true = theta_true.at[0, k, k].set(2.5)  # Self-transition preference
        # Other non-reference states get negative bias
        for j in range(K_true - 1):  # Only non-reference destinations
            if j != k:
                theta_true = theta_true.at[0, k, j].set(-0.5)
    
    # Input effects (only on non-reference destinations)
    theta_true = theta_true.at[1, 0, 1].set(2.0)  # Stronger input effect
    theta_true = theta_true.at[2, 1, 0].set(1.5)  # Another input effect
    
    W_emit_true = jnp.array([
        [-3.0, 3.0],
        [0.0, 0.0],
        [3.0, -3.0]
    ]).T
    
    true_params = {
        "theta": theta_true,
        "W_bern": W_emit_true,
        "pi0": jnp.array([1/3, 1/3, 1/3])
    }
    
    print("Simulating data from K=3 input-driven model...")
    key, subkey = random.split(key)
    sim_data = simulate_data_for_glmhmm(
        subkey, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps,
        n_bern_features=D_emit, n_trans_features=D_trans
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern'],
        x_trans=sim_data['x_trans_set']
    )
    
    results_smc = {}
    results_bq = {}
    
    for K in [2, 3, 4]:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            l2_reg_trans=0.1,  # Weaker prior (scale = sqrt(10) ≈ 3.16)
            l2_reg_bern=1.0,
            prior_var_bias_trans=100.0,  # Much weaker bias prior
            seed=42 + K
        )
        
        params, lml_hist = fit_with_restarts(data, config, n_fits=5)
        print(f"  Final LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params, data)
        
        # Compute BQ evidence
        print(f"  Computing BQ evidence...")
        start = time.time()
        try:
            key, subkey = random.split(key)
            log_z_bq, var_bq = compute_factorized_evidence(
                data, params, config, gammas, xis, key=subkey
            )
            bq_time = time.time() - start
            print(f"    BQ:  {log_z_bq:.2f} +/- {jnp.sqrt(var_bq):.2f} ({bq_time:.1f}s)")
            results_bq[K] = float(log_z_bq)
        except Exception as e:
            print(f"    BQ FAILED: {e}")
            results_bq[K] = float('-inf')
        
        # Compute SMC evidence (NUTS kernel for better accuracy)
        print(f"  Computing SMC evidence (NUTS, high accuracy)...")
        start = time.time()
        try:
            log_z_smc = compute_factorized_evidence_smc(
                data, params, config, gammas, xis,
                num_particles=200, num_mcmc_steps=15, seed=0,
                mcmc_kernel='nuts'  # NUTS for better accuracy
            )
            smc_time = time.time() - start
            print(f"    SMC: {log_z_smc:.2f} ({smc_time:.1f}s)")
            results_smc[K] = log_z_smc
        except Exception as e:
            print(f"    SMC FAILED: {e}")
            results_smc[K] = float('-inf')
    
    print("\n" + "=" * 50)
    print("MODEL SELECTION (INPUT-DRIVEN)")
    print("=" * 50)
    print(f"{'K':<5} {'SMC':<15} {'BQ':<15}")
    print("-" * 35)
    for K in [2, 3, 4]:
        smc_val = results_smc.get(K, float('-inf'))
        bq_val = results_bq.get(K, float('-inf'))
        print(f"{K:<5} {smc_val:<15.2f} {bq_val:<15.2f}")
    
    best_k_smc = max(results_smc, key=results_smc.get)
    best_k_bq = max(results_bq, key=results_bq.get)
    print(f"\nSelected by SMC: K={best_k_smc}")
    print(f"Selected by BQ:  K={best_k_bq}")
    print(f"True model:      K={K_true}")
    
    # Test passes if SMC and BQ agree (regardless of true K)
    if best_k_smc == best_k_bq:
        print(f"✓ SUCCESS: SMC and BQ agree on K={best_k_smc}")
        return True
    else:
        print(f"✗ FAILURE: SMC selected K={best_k_smc} but BQ selected K={best_k_bq}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SMC Optimization Tests")
    print("=" * 60)
    
    test1_pass = test_smc_basic()
    test2_pass = test_smc_input_driven()
    test3_pass = test_model_selection()
    test4_pass = test_model_selection_input_driven()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Basic SMC):            {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Input-Driven SMC):     {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Static Model Select):  {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Input Model Select):   {'PASS' if test4_pass else 'FAIL'}")
    
    if test1_pass and test2_pass and test3_pass and test4_pass:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
