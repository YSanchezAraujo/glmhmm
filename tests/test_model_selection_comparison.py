"""
Comprehensive Model Selection Comparison Test.

Compares all model selection methods:
- CV: Cross-validation test log-likelihood
- BQ: Bayesian Quadrature evidence
- SMC: Sequential Monte Carlo evidence  
- BIC: -BIC/2 (Laplace approximation to evidence)
- LML: Log marginal likelihood (train fit, no complexity penalty)
"""

import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

import jax.numpy as jnp
from jax import random
import time

from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.model_selection import compute_bic, compute_implied_log_z
from glmhmm.model_selection_cv import select_k_by_cv, cross_validate_k
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence
from glmhmm.model_selection_factorized_smc import compute_factorized_evidence_smc


def run_comparison_test():
    """Compare all model selection methods on K=3 static transition data."""
    print("=" * 70)
    print("COMPREHENSIVE MODEL SELECTION COMPARISON")
    print("=" * 70)
    
    K_true = 3
    D_emit = 2
    
    key = random.PRNGKey(999)
    n_sequences = 25
    n_steps = 100
    
    # True K=3 static transition model
    A_true = jnp.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.85, 0.10],
        [0.10, 0.05, 0.85]
    ])
    
    W_emit_true = jnp.array([
        [-3.0, 0.0, 3.0],
        [2.0, 0.0, -2.0]
    ])
    
    true_params = {
        "A": A_true,
        "W_bern": W_emit_true,
        "pi0": jnp.array([1/3, 1/3, 1/3])
    }
    
    print(f"\nSimulating data: K_true={K_true}, {n_sequences} sequences, {n_steps} steps each")
    sim_data = simulate_data_for_glmhmm(
        key=key,
        true_params=true_params,
        n_sequences=n_sequences,
        n_hmm_steps=n_steps,
        n_bern_features=D_emit,
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern'],
    )
    
    k_range = [2, 3, 4]
    results = {k: {} for k in k_range}
    
    # ========== Fit models for each K ==========
    print("\n" + "=" * 70)
    print("FITTING MODELS")
    print("=" * 70)
    
    fitted_params = {}
    fitted_lmls = {}
    
    for K in k_range:
        print(f"\n--- Fitting K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            l2_reg_bern=1.0,
        )
        
        params, lml_hist = fit_with_restarts(data, config, n_fits=3)
        fitted_params[K] = params
        fitted_lmls[K] = float(lml_hist[-1])
        
        results[K]['LML'] = fitted_lmls[K]
        print(f"  LML: {fitted_lmls[K]:.2f}")
    
    # ========== BIC ==========
    print("\n" + "=" * 70)
    print("COMPUTING BIC")
    print("=" * 70)
    
    for K in k_range:
        bic = compute_bic(data, fitted_params[K], fitted_lmls[K])
        implied_log_z = compute_implied_log_z(bic)
        results[K]['BIC'] = implied_log_z
        print(f"K={K}: BIC={bic:.2f}, -BIC/2={implied_log_z:.2f}")
    
    # ========== BQ ==========
    print("\n" + "=" * 70)
    print("COMPUTING BQ EVIDENCE")
    print("=" * 70)
    
    for K in k_range:
        print(f"K={K}...", end=" ", flush=True)
        config = GLMHMMConfig(num_states=K, num_em_iters=100, l2_reg_bern=1.0)
        gammas, xis, _ = e_step(fitted_params[K], data)
        
        key, subkey = random.split(key)
        t0 = time.time()
        log_z, var = compute_factorized_evidence(
            data, fitted_params[K], config, gammas, xis, subkey,
            static_trans_alpha=0.5
        )
        t1 = time.time()
        
        results[K]['BQ'] = float(log_z)
        print(f"BQ={log_z:.2f} ({t1-t0:.1f}s)")
    
    # ========== SMC ==========
    print("\n" + "=" * 70)
    print("COMPUTING SMC EVIDENCE (NUTS kernel)")
    print("=" * 70)
    
    for K in k_range:
        print(f"K={K}...", end=" ", flush=True)
        config = GLMHMMConfig(num_states=K, num_em_iters=100, l2_reg_bern=1.0)
        gammas, xis, _ = e_step(fitted_params[K], data)
        
        t0 = time.time()
        log_z = compute_factorized_evidence_smc(
            data, fitted_params[K], config, gammas, xis,
            num_particles=100,
            num_mcmc_steps=10,
            seed=42 + K,
            mcmc_kernel='nuts',
            static_trans_alpha=0.5
        )
        t1 = time.time()
        
        results[K]['SMC'] = float(log_z)
        print(f"SMC={log_z:.2f} ({t1-t0:.1f}s)")
    
    # ========== CV ==========
    print("\n" + "=" * 70)
    print("COMPUTING CV LOG-LIKELIHOOD")
    print("=" * 70)
    
    base_config = GLMHMMConfig(
        num_states=2,
        num_em_iters=100,
        l2_reg_bern=1.0,
    )
    
    for K in k_range:
        print(f"\n--- K={K} ---")
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=100,
            l2_reg_bern=1.0,
        )
        mean_ll, std_ll, _ = cross_validate_k(
            data, config, n_folds=5, n_fits=2, seed=42, verbose=True
        )
        results[K]['CV'] = mean_ll
        print(f"  CV Mean: {mean_ll:.2f} ± {std_ll:.2f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY: MODEL SELECTION SCORES")
    print("=" * 70)
    
    methods = ['LML', 'BIC', 'BQ', 'SMC', 'CV']
    
    # Header
    header = f"{'K':>3}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print("-" * (4 + 14 * len(methods)))
    
    # Rows
    for K in k_range:
        row = f"{K:>3}"
        for m in methods:
            row += f"  {results[K][m]:>12.2f}"
        print(row)
    
    print("-" * (4 + 14 * len(methods)))
    
    # Find best K for each method
    print("\nBest K selected:")
    selections = {}
    for m in methods:
        best_k = max(k_range, key=lambda k: results[k][m])
        selections[m] = best_k
        marker = "✓" if best_k == K_true else "✗"
        print(f"  {m:>4}: K={best_k} {marker}")
    
    # Count correct
    n_correct = sum(1 for m in methods if selections[m] == K_true)
    print(f"\nCorrect: {n_correct}/{len(methods)} methods selected K={K_true}")
    
    return selections, results


if __name__ == "__main__":
    run_comparison_test()
