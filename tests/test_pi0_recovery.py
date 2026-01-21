
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from glmhmm.structs import GLMHMMConfig
from glmhmm.fit import fit_glmhmm
from glmhmm.utils import simulate_data_for_glmhmm
import glmhmm.fit
print(f"DEBUG: glmhmm.fit is loading from: {glmhmm.fit.__file__}")

def test_pi0_recovery():
    print("=== Testing Recovery of Initial State Probabilities (pi0) ===")
    
    key = jax.random.PRNGKey(42)
    
    # 1. Define True Parameters
    # K=2
    # Distinct pi0: [0.2, 0.8]
    K_true = 2
    true_pi0 = jnp.array([0.2, 0.8])
    
    # Static transitions (strong diagonal to persist state)
    true_A = jnp.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])
    
    # Distinct Emissions (Bernoulli)
    # State 0: Low Prob (logit -2)
    # State 1: High Prob (logit +2)
    D = 1 # Just bias
    true_W_bern = jnp.array([[-2.0], [2.0]]) # (D, K) -> (1, 2)
    # Wait, W shape is (D, K).
    true_W_bern = jnp.array([[-2.0, 2.0]]) # (1, 2)

    true_params = {
        "pi0": true_pi0,
        "A": true_A,
        "W_bern": true_W_bern
    }
    
    # 2. Simulate Data
    # IMPORTANT: To recover pi0 well, we need MANY sequences.
    # pi0 is only observed at t=0.
    n_sequences = 500
    n_steps = 20 # Short sequences are fine
    
    print(f"Simulating {n_sequences} sequences of length {n_steps}...")
    sim_data_dict = simulate_data_for_glmhmm(
        key, 
        true_params, 
        n_sequences=n_sequences, 
        n_hmm_steps=n_steps, 
        n_bern_features=D
    )
    
    from glmhmm.structs import GLMHMMData
    data = GLMHMMData(
        x_bern=sim_data_dict['x_set_bern'],
        y_bern=sim_data_dict['y_set_bern']
    )
    
    # 3. Fit Model
    print("Fitting GLM-HMM...")
    config = GLMHMMConfig(
        num_states=K_true,
        num_em_iters=300, # Should converge fast
        tol=1e-4,
        seed=123
    )
    
    # We initialize randomly, but we hope to recover the true states.
    # Given the distinct emissions, it should be easy.
    fitted_params, lml_hist = fit_glmhmm(data, config)
    
    # 4. Compare pi0
    recovered_log_pi0 = fitted_params.log_pi0
    recovered_pi0 = jnp.exp(recovered_log_pi0)
    
    print("\n--- Results ---")
    print(f"True pi0:      {true_pi0}")
    print(f"Recovered pi0: {recovered_pi0}")
    
    # Handle permutation (though with distinct emissions and random init, it might flip)
    # We can check both permutations
    
    diff_direct = jnp.abs(recovered_pi0 - true_pi0)
    diff_flipped = jnp.abs(recovered_pi0 - true_pi0[::-1])
    
    if jnp.mean(diff_direct) < jnp.mean(diff_flipped):
        print("Permutation: Direct match")
        final_pi0 = recovered_pi0
        final_diff = diff_direct
    else:
        print("Permutation: Flipped match")
        final_pi0 = recovered_pi0[::-1]
        final_diff = diff_flipped
        
    print(f"Aligned pi0:   {final_pi0}")
    print(f"Error:         {final_diff}")
    
    # Assert reasonable recovery (e.g., within 0.05)
    # Standard error for proportion p=0.2 with N=1000 is sqrt(0.2*0.8/1000) ~= 0.012.
    # So 0.05 is ~4 sigma. Safe.
    
    assert jnp.all(final_diff < 0.1), f"Failed to recover pi0. Error: {final_diff}"
    print("SUCCESS: pi0 recovered within tolerance.")

if __name__ == "__main__":
    test_pi0_recovery()
