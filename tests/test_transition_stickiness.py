
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from glmhmm.structs import GLMHMMConfig, GLMHMMData
from glmhmm.fit import fit_glmhmm
from glmhmm.utils import simulate_data_for_glmhmm

def test_stickiness():
    print("=== Testing Transition Stickiness ===")
    key = jax.random.PRNGKey(0)
    
    # 1. Setup Data
    K = 3
    true_params = {
        "pi0": jnp.ones(K)/K,
        "A": jnp.ones((K, K))/K, # Balanced transitions
        "W_bern": jnp.eye(K) # Identity emission logic (D=K)
    }
    
    sim_data = simulate_data_for_glmhmm(
        key, true_params, n_sequences=10, n_hmm_steps=100, n_bern_features=K
    )
    
    data = GLMHMMData(
        x_bern=sim_data['x_set_bern'],
        y_bern=sim_data['y_set_bern']
    )
    
    # 2. Fit with Stickiness = 0
    print("\nFitting with stickiness = 0.0...")
    config_0 = GLMHMMConfig(
        num_states=K, num_em_iters=20, transition_stickiness=0.0, seed=42
    )
    params_0, _ = fit_glmhmm(data, config_0)
    A_0 = jnp.exp(params_0.log_A)
    print("A (Stickiness 0):")
    print(np.round(A_0, 3))
    diag_mean_0 = jnp.mean(jnp.diag(A_0))
    print(f"Mean Diagonal: {diag_mean_0:.3f}")
    
    # 3. Fit with Stickiness = 50 (Strong)
    print("\nFitting with stickiness = 50.0...")
    config_sticky = GLMHMMConfig(
        num_states=K, num_em_iters=20, transition_stickiness=50.0, seed=42
    )
    params_sticky, _ = fit_glmhmm(data, config_sticky)
    A_sticky = jnp.exp(params_sticky.log_A)
    print("A (Stickiness 50):")
    print(np.round(A_sticky, 3))
    diag_mean_sticky = jnp.mean(jnp.diag(A_sticky))
    print(f"Mean Diagonal: {diag_mean_sticky:.3f}")
    
    # 4. Check
    assert diag_mean_sticky > diag_mean_0 + 0.1, "Stickiness did not significantly increase diagonal!"
    print("\nSUCCESS: Stickiness parameter increased diagonal weight.")

if __name__ == "__main__":
    test_stickiness()
