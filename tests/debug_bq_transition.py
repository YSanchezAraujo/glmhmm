"""
Diagnostic script to understand why transition evidence scales poorly with K.
"""
import jax
import jax.numpy as jnp
from jax import random, hessian
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMConfig, GLMHMMParams
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection_factorized_bq import run_bq_tinygp

def analyze_transition_evidence():
    print("\n=== Diagnosing Transition Evidence Calculation ===\n")
    
    key = random.PRNGKey(42)
    n_sequences = 10
    n_steps = 1000
    K_true = 3
    D_emit = 1
    
    # Setup true params (same as test)
    pi0_true = jnp.array([1.0, 0.0, 0.0])
    A_true = jnp.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    
    W_emit_true = jnp.zeros((1, K_true))
    W_emit_true = W_emit_true.at[0, 0].set(-2.2)
    W_emit_true = W_emit_true.at[0, 1].set(2.2)
    W_emit_true = W_emit_true.at[0, 2].set(0.0)
    
    true_params = {"pi0": pi0_true, "A": A_true, "W_bern": W_emit_true}
    
    # Simulate data
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences=n_sequences, n_hmm_steps=n_steps, 
        n_bern_features=D_emit
    )
    
    full_data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern']
    )
    
    for K in [2, 3]:
        print(f"\n{'='*60}")
        print(f"Analyzing K = {K}")
        print(f"{'='*60}")
        
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=200,
            transition_stickiness=10.0,
            l2_reg_bern=1.0,
            l2_reg_trans=1.0,
            seed=100+K
        )
        
        params_fit, lml_hist = fit_with_restarts(full_data, config, n_fits=3)
        print(f"\nFitted LML: {lml_hist[-1]:.2f}")
        
        gammas, xis, _ = e_step(params_fit, full_data)
        
        # Analyze transition counts - handle list type
        if isinstance(xis, list):
            counts_trans = sum([jnp.sum(x, axis=0) for x in xis])  # (K, K)
        else:
            counts_trans = jnp.sum(xis, axis=(0, 1))  # (K, K)
        print(f"\nTransition Counts (xis summed):")
        print(counts_trans)
        print(f"Total transitions: {jnp.sum(counts_trans):.2f}")
        
        # Analyze per-row BQ
        print(f"\n--- Per-Row BQ Analysis ---")
        
        logit_A_full = params_fit.log_A
        print(f"\nlog_A (fitted):")
        print(logit_A_full)
        
        total_log_Z_trans = 0.0
        
        for j in range(K):
            center_j_full = logit_A_full[j]
            center_j = center_j_full[:-1] - center_j_full[-1]  # Reduce to K-1
            
            row_counts = counts_trans[j]
            
            def make_lp(row_counts, state_idx, stickiness, l2_reg):
                def _lp(logit_row):
                    probs = jax.nn.softmax(logit_row)
                    log_probs = jax.nn.log_softmax(logit_row)
                    ll = jnp.dot(row_counts, log_probs)
                    
                    D_prior = logit_row.shape[0]
                    log_prior_const = 0.5 * D_prior * (jnp.log(l2_reg) - jnp.log(2 * jnp.pi))
                    prior_stickiness = stickiness * log_probs[state_idx]
                    prior_l2 = -0.5 * l2_reg * jnp.sum(logit_row**2)
                    
                    return ll + prior_l2 + prior_stickiness + log_prior_const
                return _lp
            
            base_lp = make_lp(row_counts, state_idx=j, stickiness=config.transition_stickiness, l2_reg=config.l2_reg_trans)
            
            def lp_reduced(logit_reduced):
                logit_full = jnp.concatenate([logit_reduced, jnp.array([0.0])])
                return base_lp(logit_full)
            
            current_dim = K - 1
            
            # Compute Laplace approximation components manually
            val_at_center = lp_reduced(center_j)
            H = hessian(lp_reduced)(center_j)
            H_reg = H - 1e-4 * jnp.eye(current_dim)
            
            sign, logdet_neg_H = jnp.linalg.slogdet(-H_reg)
            log_z_laplace = val_at_center + 0.5 * current_dim * jnp.log(2 * jnp.pi) - 0.5 * logdet_neg_H
            
            # Full BQ
            key, subkey = random.split(key)
            log_z_bq, var_bq = run_bq_tinygp(lp_reduced, center_j, dim=current_dim, n_samples=2000, key=subkey)
            
            print(f"\n  Row {j}:")
            print(f"    center (reduced): {center_j}")
            print(f"    row_counts: {row_counts}")
            print(f"    val_at_center (log p(theta*)): {val_at_center:.4f}")
            print(f"    logdet(-H): {logdet_neg_H:.4f}")
            print(f"    Laplace log Z: {log_z_laplace:.4f}")
            print(f"    BQ log Z: {log_z_bq:.4f}")
            
            total_log_Z_trans += log_z_bq
        
        print(f"\n  Total log_Z_trans: {total_log_Z_trans:.4f}")

if __name__ == "__main__":
    analyze_transition_evidence()
