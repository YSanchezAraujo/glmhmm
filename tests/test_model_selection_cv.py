"""
Test Cross-Validation Model Selection.
"""

import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

import jax.numpy as jnp
from jax import random

from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.model_selection_cv import select_k_by_cv


def test_cv_model_selection_input_driven():
    """Test CV model selection on input-driven transitions (where BQ/SMC failed)."""
    print("=" * 60)
    print("CV Model Selection Test (Input-Driven Transitions)")
    print("=" * 60)
    
    K_true = 3
    D_emit = 2
    D_trans = 3
    
    key = random.PRNGKey(888)
    n_sequences = 30  # More data for CV
    n_steps = 150
    
    # True K=3 input-driven model
    theta_true = jnp.zeros((D_trans, K_true, K_true))
    
    # Set up sticky bias relative to reference state
    for k in range(K_true):
        if k < K_true - 1:
            theta_true = theta_true.at[0, k, k].set(2.5)
        for j in range(K_true - 1):
            if j != k:
                theta_true = theta_true.at[0, k, j].set(-0.5)
    
    # Input effects
    theta_true = theta_true.at[1, 0, 1].set(2.0)
    theta_true = theta_true.at[2, 1, 0].set(1.5)
    
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
    
    print("\nSimulating data from K=3 input-driven model...")
    sim_data = simulate_data_for_glmhmm(
        key=key,
        true_params=true_params,
        n_sequences=n_sequences,
        n_hmm_steps=n_steps,
        n_bern_features=D_emit,
        n_trans_features=D_trans,
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern'],
        x_trans=sim_data['x_trans_set']
    )
    
    # Base config
    base_config = GLMHMMConfig(
        num_states=2,  # Will be overridden
        num_em_iters=100,
        l2_reg_trans=0.1,
        l2_reg_bern=1.0,
    )
    
    # Run CV model selection
    best_k, cv_results = select_k_by_cv(
        data,
        k_range=[2, 3, 4],
        base_config=base_config,
        n_folds=5,
        n_fits=3,
        seed=42,
        verbose=True
    )
    
    print(f"\n{'=' * 60}")
    if best_k == K_true:
        print(f"✓ SUCCESS: CV correctly selected K={K_true}")
        return True
    else:
        print(f"✗ FAILURE: CV selected K={best_k} (true: K={K_true})")
        return False


if __name__ == "__main__":
    test_cv_model_selection_input_driven()
