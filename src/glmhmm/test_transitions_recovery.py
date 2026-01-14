"""
Parameter recovery test for input-driven transitions M-step.

Tests:
1. Sets known "true" theta parameters
2. Simulates transition data from those parameters
3. Fits the model to recover theta
4. Checks if recovered parameters are close to true parameters
"""

import jax.numpy as jnp
from jax import random, nn
import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.m_step import transitions_m_step_optax_reduced
from glmhmm.utils import compute_A_from_theta_and_inputs


def test_transitions_parameter_recovery():
    """
    Test input-driven transitions M-step by simulating data and recovering parameters.
    """
    print("=" * 60)
    print("Input-Driven Transitions Parameter Recovery Test")
    print("=" * 60)
    
    key = random.PRNGKey(45)
    T, D, K = 200, 3, 3  # T time steps, D features, K states
    
    # True theta parameters (D, K, K)
    # theta[d, i, j] = influence of feature d on transition from state i to state j
    true_theta = jnp.array([
        [[0.0, -1.0, -1.5],   # From state 0
         [-1.5, 0.0, -1.0],   # From state 1
         [-1.0, -1.5, 0.0]],  # From state 2 (last column will be set to 0)
        
        [[0.0, 0.5, -0.3],
         [-0.3, 0.0, 0.5],
         [0.5, -0.3, 0.0]],
        
        [[0.0, -0.4, 0.6],
         [0.6, 0.0, -0.4],
         [-0.4, 0.6, 0.0]]
    ])
    
    true_theta = true_theta.at[:, :, -1].set(0.0)
    
    print(f"\nTrue theta shape: {true_theta.shape}")
    print(f"True theta (feature 0):\n{true_theta[0]}")
    print(f"True theta (feature 1):\n{true_theta[1]}")
    print(f"True theta (feature 2):\n{true_theta[2]}")
    
    U = random.normal(key, (T, D))
    U = U.at[:, 0].set(1.0)  # First feature is intercept
    
    print(f"\nGenerated {T} time steps of transition inputs")
    
    A_t = compute_A_from_theta_and_inputs(U, true_theta)  # (T, K, K)
    print(f"Transition matrix shape: {A_t.shape}")
    print(f"Example A_t[0]:\n{A_t[0]}")
    
    states = jnp.zeros(T, dtype=jnp.int32)
    states = states.at[0].set(random.categorical(random.fold_in(key, 0), 
                                                  jnp.log(jnp.ones(K)/K)))
    
    for t in range(1, T):
        prev_state = states[t-1]
        transition_probs = A_t[t-1, prev_state, :]
        states = states.at[t].set(random.categorical(random.fold_in(key, t), 
                                                      jnp.log(transition_probs)))
    
    print(f"\nSimulated state sequence")
    print(f"State distribution: {jnp.bincount(states, length=K)}")
    
    xi = jnp.zeros((T-1, K, K))
    for t in range(T-1):
        i, j = states[t], states[t+1]
        xi = xi.at[t, i, j].set(1.0)
    
    print(f"\nXi shape: {xi.shape}")
    print(f"Total transitions per state pair:")
    xi_sum = jnp.sum(xi, axis=0)
    print(xi_sum)
    
    init_theta = 0.01 * random.normal(random.fold_in(key, 100), (D, K, K))
    init_theta = init_theta.at[:, :, -1].set(0.0)
    
    print("\nFitting model...")
    theta_recovered, _ = transitions_m_step_optax_reduced(
        U[:-1],
        xi,
        init_theta,
        dir_diag=jnp.ones(K),
        transition_ridge=0.01,
        tol=1e-4,
        num_opt_steps=1000
    )
    
    print(f"\nRecovered theta shape: {theta_recovered.shape}")
    print(f"Recovered theta (feature 0):\n{theta_recovered[0]}")
    print(f"Recovered theta (feature 1):\n{theta_recovered[1]}")
    print(f"Recovered theta (feature 2):\n{theta_recovered[2]}")
    
    theta_reduced_true = true_theta[:, :, :-1]
    theta_reduced_recovered = theta_recovered[:, :, :-1]
    
    error = jnp.mean((theta_reduced_recovered - theta_reduced_true)**2)
    max_error = jnp.max(jnp.abs(theta_reduced_recovered - theta_reduced_true))
    
    print(f"\nMean Squared Error (reduced theta): {error:.6f}")
    print(f"Max Absolute Error (reduced theta): {max_error:.6f}")
    
    last_col_is_zero = jnp.allclose(theta_recovered[:, :, -1], 0.0, atol=1e-6)
    print(f"Last column is zero (identifiability constraint): {last_col_is_zero}")
    
    success = (error < 0.5) and last_col_is_zero
    
    if success:
        print("\n✓ PASSED: Transition parameters recovered successfully!")
    else:
        print("\n✗ FAILED: Parameter recovery error too large")
    
    return success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRANSITIONS M-STEP PARAMETER RECOVERY TEST")
    print("=" * 60)
    
    success = test_transitions_parameter_recovery()
    
    print("\n" + "=" * 60)
    if success:
        print("✓✓✓ TEST PASSED! Transitions M-step works correctly! ✓✓✓")
    else:
        print("✗ TEST FAILED. Review errors above.")
    print("=" * 60 + "\n")
