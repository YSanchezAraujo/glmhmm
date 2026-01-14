"""
Parameter recovery tests for refactored M-steps.

Each test:
1. Sets known "true" parameters
2. Simulates data from those parameters
3. Fits the model to recover parameters
4. Checks if recovered parameters are close to true parameters
"""

import jax.numpy as jnp
from jax import random, nn
import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.m_step import (
    bern_m_step_optax,
    negbin_m_step_optax,
    gamma_m_step_optax
)


def test_bernoulli_parameter_recovery():
    """
    Test Bernoulli M-step by simulating data and recovering parameters.
    """
    print("=" * 60)
    print("Test 1: Bernoulli Parameter Recovery")
    print("=" * 60)
    
    key = random.PRNGKey(42)
    N, D, K = 500, 4, 3
    
    true_W = jnp.array([
        [1.5, -0.8, 0.3],
        [-1.2, 1.0, -0.5],
        [0.5, -0.3, 1.2],
        [0.0, 0.0, 0.0]
    ])
    
    print(f"\nTrue W:\n{true_W}")
    
    X = random.normal(key, (N, D))
    X = X.at[:, -1].set(1.0)  # Intercept column
    
    state_assignments = random.categorical(random.fold_in(key, 1), 
                                          jnp.log(jnp.ones(K)/K), 
                                          shape=(N,))
    
    logits = X @ true_W  # (N, K)
    probs = nn.sigmoid(logits)
    
    y = jnp.zeros(N)
    for i in range(N):
        state = state_assignments[i]
        y = y.at[i].set(random.bernoulli(random.fold_in(key, i+100), 
                                         probs[i, state]).astype(jnp.float32))
    
    gamma = jnp.zeros((N, K))
    for i in range(N):
        gamma = gamma.at[i, state_assignments[i]].set(1.0)
    
    print(f"\nSimulated {N} observations")
    print(f"State distribution: {jnp.bincount(state_assignments, length=K)}")
    
    init_W = 0.01 * random.normal(random.fold_in(key, 200), (D, K))
    
    print("\nFitting model...")
    W_recovered, _ = bern_m_step_optax(
        X, y, gamma, init_W, 
        l2_reg=0.01,  
        tol=1e-4, 
        num_opt_steps=1000
    )
    
    print(f"\nRecovered W:\n{W_recovered}")
    
    error = jnp.mean((W_recovered - true_W)**2)
    max_error = jnp.max(jnp.abs(W_recovered - true_W))
    
    print(f"\nMean Squared Error: {error:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    
    success = error < 0.1
    
    if success:
        print("✓ PASSED: Parameters recovered successfully!")
    else:
        print("✗ FAILED: Parameter recovery error too large")
    
    return success


def test_negbin_parameter_recovery():
    """
    Test Negative Binomial M-step by simulating data and recovering parameters.
    """
    print("\n" + "=" * 60)
    print("Test 2: Negative Binomial Parameter Recovery")
    print("=" * 60)
    
    key = random.PRNGKey(43)
    N, D, K = 500, 4, 3
    
    # NOTE: Using Poisson data (phi → ∞), so set true_phi very large
    true_W = jnp.array([
        [0.5, -0.3, 0.2],
        [-0.4, 0.6, -0.3],
        [0.3, -0.2, 0.5],
        [1.0, 1.5, 2.0]  # Intercepts (log mean counts)
    ])
    true_phi = jnp.array([10000.0, 10000.0, 10000.0])  # Very large = Poisson limit
    
    print(f"\nTrue W:\n{true_W}")
    print(f"True phi: {true_phi}")
    
    X = random.normal(key, (N, D))
    X = X.at[:, -1].set(1.0)  # Intercept
    
    state_assignments = random.categorical(random.fold_in(key, 1), 
                                          jnp.log(jnp.ones(K)/K), 
                                          shape=(N,))
    
    logits = X @ true_W
    mu = jnp.exp(jnp.clip(logits, a_max=5.0))
    
    y = jnp.zeros(N)
    for i in range(N):
        state = state_assignments[i]
        y_sample = random.poisson(random.fold_in(key, i+100), mu[i, state])
        y = y.at[i].set(y_sample.astype(jnp.float32))
    
    gamma = jnp.zeros((N, K))
    for i in range(N):
        gamma = gamma.at[i, state_assignments[i]].set(1.0)
    
    print(f"\nSimulated {N} observations")
    print(f"Mean counts per state: {[jnp.mean(y[state_assignments == k]) for k in range(K)]}")
    
    init_W = 0.01 * random.normal(random.fold_in(key, 200), (D, K))
    init_W = init_W.at[-1, :].set(jnp.log(jnp.mean(y)))  # Better intercept init
    init_phi = jnp.ones(K) * 3.0
    
    print("\nFitting model...")
    W_recovered, phi_recovered, _ = negbin_m_step_optax(
        X, y, gamma, init_W, init_phi,
        l2_reg=0.01,
        tol=1e-4,
        num_opt_steps=1000
    )
    
    print(f"\nRecovered W:\n{W_recovered}")
    print(f"Recovered phi: {phi_recovered}")
    
    w_error = jnp.mean((W_recovered - true_W)**2)
    phi_is_large = jnp.all(phi_recovered > 50.0)
    
    print(f"\nW Mean Squared Error: {w_error:.6f}")
    print(f"Phi values are large (>50): {phi_is_large}")
    print(f"  Recovered phi: {phi_recovered}")
    print(f"  (This is correct for Poisson data, which is NegBin with phi → ∞)")
    
    success = (w_error < 0.2) and phi_is_large
    
    if success:
        print("✓ PASSED: Parameters recovered successfully!")
    else:
        print("✗ FAILED: Parameter recovery error too large")
    
    return success


def test_gamma_parameter_recovery():
    """
    Test Gamma M-step by simulating data and recovering parameters.
    """
    print("\n" + "=" * 60)
    print("Test 3: Gamma Parameter Recovery")
    print("=" * 60)
    
    key = random.PRNGKey(44)
    N, D, K = 500, 4, 3
    
    true_W = jnp.array([
        [0.3, -0.2, 0.1],
        [-0.2, 0.4, -0.1],
        [0.1, -0.1, 0.3],
        [0.5, 1.0, 1.5]  # Intercepts (log mean)
    ])
    true_alpha = jnp.array([2.0, 3.0, 2.5])  # Shape parameters
    
    print(f"\nTrue W:\n{true_W}")
    print(f"True alpha: {true_alpha}")
    
    X = random.normal(key, (N, D))
    X = X.at[:, -1].set(1.0)  # Intercept
    
    state_assignments = random.categorical(random.fold_in(key, 1), 
                                          jnp.log(jnp.ones(K)/K), 
                                          shape=(N,))
    
    logits = X @ true_W
    mu = jnp.exp(jnp.clip(logits, a_min=-2.0, a_max=3.0))
    
    y = jnp.zeros(N)
    for i in range(N):
        state = state_assignments[i]
        y_sample = random.gamma(random.fold_in(key, i+100), true_alpha[state]) * (mu[i, state] / true_alpha[state])
        y = y.at[i].set(y_sample)
    
    gamma = jnp.zeros((N, K))
    for i in range(N):
        gamma = gamma.at[i, state_assignments[i]].set(1.0)
    
    print(f"\nSimulated {N} observations")
    print(f"Mean values per state: {[jnp.mean(y[state_assignments == k]) for k in range(K)]}")
    
    init_W = 0.01 * random.normal(random.fold_in(key, 200), (D, K))
    init_W = init_W.at[-1, :].set(jnp.log(jnp.mean(y)))  
    init_alpha = jnp.ones(K) * 2.0
    
    print("\nFitting model...")
    W_recovered, alpha_recovered, _ = gamma_m_step_optax(
        X, y, gamma, init_W, init_alpha,
        l2_reg=0.01,
        tol=1e-4,
        num_opt_steps=1000
    )
    
    print(f"\nRecovered W:\n{W_recovered}")
    print(f"Recovered alpha: {alpha_recovered}")
    
    w_error = jnp.mean((W_recovered - true_W)**2)
    alpha_error = jnp.mean((alpha_recovered - true_alpha)**2)
    
    print(f"\nW Mean Squared Error: {w_error:.6f}")
    print(f"Alpha Mean Squared Error: {alpha_error:.6f}")
    
    success = (w_error < 0.2) and (alpha_error < 1.0)
    
    if success:
        print("✓ PASSED: Parameters recovered successfully!")
    else:
        print("✗ FAILED: Parameter recovery error too large")
    
    return success


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PARAMETER RECOVERY TEST SUITE")
    print("Testing refactored M-steps with known ground truth")
    print("=" * 60)
    
    results = []
    results.append(test_bernoulli_parameter_recovery())
    results.append(test_negbin_parameter_recovery())
    results.append(test_gamma_parameter_recovery())
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED! Refactored M-steps work correctly! ✓✓✓\n")
    else:
        print("\n✗ Some tests failed. Review errors above.\n")
