"""
Parameter recovery test for Multinomial observations M-step.

Tests:
1. Sets known "true" W parameters for multinomial observations
2. Simulates multinomial choice data from those parameters
3. Fits the model to recover W
4. Checks if recovered parameters are close to true parameters
"""

import jax.numpy as jnp
from jax import random, nn
import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.m_step import multinomial_obs_m_step_optax


def test_multinomial_parameter_recovery():
    """
    Test Multinomial M-step by simulating choice data and recovering parameters.
    """
    print("=" * 60)
    print("Multinomial Observations Parameter Recovery Test")
    print("=" * 60)
    
    key = random.PRNGKey(46)
    N, D, C, K = 500, 4, 4, 3  # N trials, D features, C choices, K states
    
    true_W = jnp.array([
        # Feature 0 (intercept)
        [[0.5, -0.3, 0.2],   # Choice 0
         [-0.4, 0.6, -0.3],  # Choice 1
         [0.3, -0.2, 0.5],   # Choice 2
         [0.0, 0.0, 0.0]],   # Choice 3 (reference)
        
        # Feature 1 (e.g., reward magnitude)
        [[0.8, -0.5, 0.3],
         [-0.6, 0.7, -0.4],
         [0.4, -0.3, 0.6],
         [0.0, 0.0, 0.0]],
        
        # Feature 2 (e.g., previous choice)
        [[-0.3, 0.4, -0.2],
         [0.5, -0.4, 0.3],
         [-0.2, 0.3, -0.4],
         [0.0, 0.0, 0.0]],
        
        # Feature 3 (e.g., context)
        [[0.2, -0.3, 0.4],
         [-0.3, 0.5, -0.2],
         [0.4, -0.2, 0.3],
         [0.0, 0.0, 0.0]]
    ])
    
    print(f"\nTrue W shape: {true_W.shape}")
    print(f"True W (feature 0):\n{true_W[0]}")
    print(f"Reference category (last column of each feature) is zero: {jnp.allclose(true_W[:, -1, :], 0.0)}")
    
    X = random.normal(key, (N, D))
    X = X.at[:, 0].set(1.0)  # First feature is intercept
    
    state_assignments = random.categorical(random.fold_in(key, 1), 
                                          jnp.log(jnp.ones(K)/K), 
                                          shape=(N,))
    
    print(f"\nSimulated {N} trials")
    print(f"State distribution: {jnp.bincount(state_assignments, length=K)}")
    
    y = jnp.zeros(N, dtype=jnp.int32)
    for i in range(N):
        state = state_assignments[i]
        logits = X[i] @ true_W[:, :, state]  # (C,)
        probs = nn.softmax(logits)
        y = y.at[i].set(random.categorical(random.fold_in(key, i+100), jnp.log(probs)))
    
    print(f"Choice distribution: {jnp.bincount(y, length=C)}")
    
    gamma = jnp.zeros((N, K))
    for i in range(N):
        gamma = gamma.at[i, state_assignments[i]].set(1.0)
    
    init_W = 0.01 * random.normal(random.fold_in(key, 200), (D, C, K))
    init_W = init_W.at[:, -1, :].set(0.0) 
    
    print("\nFitting model...")
    W_recovered, _ = multinomial_obs_m_step_optax(
        X, y, gamma, init_W,
        l2_reg=0.01,
        tol=1e-4,
        num_opt_steps=1000
    )
    
    print(f"\nRecovered W shape: {W_recovered.shape}")
    print(f"Recovered W (feature 0):\n{W_recovered[0]}")
    print(f"Reference category is zero: {jnp.allclose(W_recovered[:, -1, :], 0.0, atol=1e-6)}")
    
    W_reduced_true = true_W[:, :-1, :]
    W_reduced_recovered = W_recovered[:, :-1, :]
    
    error = jnp.mean((W_reduced_recovered - W_reduced_true)**2)
    max_error = jnp.max(jnp.abs(W_reduced_recovered - W_reduced_true))
    
    print(f"\nMean Squared Error (non-reference categories): {error:.6f}")
    print(f"Max Absolute Error (non-reference categories): {max_error:.6f}")
    
    ref_is_zero = jnp.allclose(W_recovered[:, -1, :], 0.0, atol=1e-6)
    print(f"Reference category constraint maintained: {ref_is_zero}")
    
    success = (error < 0.2) and ref_is_zero
    
    if success:
        print("\n✓ PASSED: Multinomial parameters recovered successfully!")
    else:
        print("\n✗ FAILED: Parameter recovery error too large")
    
    return success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTINOMIAL M-STEP PARAMETER RECOVERY TEST")
    print("=" * 60)
    
    success = test_multinomial_parameter_recovery()
    
    print("\n" + "=" * 60)
    if success:
        print("✓✓✓ TEST PASSED! Multinomial M-step works correctly! ✓✓✓")
    else:
        print("✗ TEST FAILED. Review errors above.")
    print("=" * 60 + "\n")
