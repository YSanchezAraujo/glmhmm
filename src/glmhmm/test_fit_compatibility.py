"""
Test script to verify fit.py compatibility with other GLMHMM modules.
Tests both static and input-driven transitions.
"""

import jax.numpy as jnp
from jax import random
import sys
sys.path.insert(0, '/Users/ysa/Desktop/glmhmm/src')

from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.fit import fit_glmhmm

def test_static_transitions():
    """Test with static transitions (no theta, no x_trans)"""
    print("Testing static transitions...")
    
    key = random.PRNGKey(42)
    n_sequences = 3
    T = 50
    n_features = 5
    K = 3
    
    x_bern = [random.normal(random.fold_in(key, i), (T, n_features)) for i in range(n_sequences)]
    y_bern = [random.bernoulli(random.fold_in(key, i+100), p=0.5, shape=(T,)).astype(jnp.float32) 
              for i in range(n_sequences)]
    
    data = GLMHMMData(
        x_trans=None, 
        x_bern=x_bern,
        y_bern=y_bern,
        x_multinom=None,
        y_multinom=None,
        x_negbin=None,
        y_negbin=None,
        x_gamma=None,
        y_gamma=None
    )
    
    config = GLMHMMConfig(
        num_states=K,
        num_em_iters=5,
        tol=1e-3,
        l2_reg_bern=1.0
    )
    
    try:
        params, lml_history = fit_glmhmm(data, config)
        print(f"✓ Static transitions test passed!")
        print(f"  Final log-likelihood: {lml_history[-1]:.2f}")
        print(f"  LML increased: {lml_history[-1] > lml_history[0]}")
        return True
    except Exception as e:
        print(f"✗ Static transitions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_driven_transitions():
    """Test with input-driven transitions (with theta and x_trans)"""
    print("\nTesting input-driven transitions...")
    
    key = random.PRNGKey(43)
    n_sequences = 3
    T = 50
    n_bern_features = 5
    n_trans_features = 3
    K = 3
    
    x_bern = [random.normal(random.fold_in(key, i), (T, n_bern_features)) for i in range(n_sequences)]
    y_bern = [random.bernoulli(random.fold_in(key, i+100), p=0.5, shape=(T,)).astype(jnp.float32) 
              for i in range(n_sequences)]
    
    x_trans = [random.normal(random.fold_in(key, i+200), (T, n_trans_features)) for i in range(n_sequences)]
    
    data = GLMHMMData(
        x_trans=x_trans,
        x_bern=x_bern,
        y_bern=y_bern,
        x_multinom=None,
        y_multinom=None,
        x_negbin=None,
        y_negbin=None,
        x_gamma=None,
        y_gamma=None
    )
    
    config = GLMHMMConfig(
        num_states=K,
        num_em_iters=5,
        tol=1e-3,
        l2_reg_bern=1.0,
        l2_reg_trans=1.0
    )
    
    try:
        params, lml_history = fit_glmhmm(data, config)
        print(f"✓ Input-driven transitions test passed!")
        print(f"  Final log-likelihood: {lml_history[-1]:.2f}")
        print(f"  LML increased: {lml_history[-1] > lml_history[0]}")
        return True
    except Exception as e:
        print(f"✗ Input-driven transitions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multinomial_observations():
    """Test with multinomial observations to verify l2_reg_multinom field"""
    print("\nTesting multinomial observations...")
    
    key = random.PRNGKey(44)
    n_sequences = 3
    T = 50
    n_features = 5
    n_categories = 4
    K = 3
    
    x_multinom = [random.normal(random.fold_in(key, i), (T, n_features)) for i in range(n_sequences)]
    y_multinom = [random.randint(random.fold_in(key, i+100), (T,), 0, n_categories) 
                  for i in range(n_sequences)]
    
    data = GLMHMMData(
        x_trans=None,
        x_bern=None,
        y_bern=None,
        x_multinom=x_multinom,
        y_multinom=y_multinom,
        x_negbin=None,
        y_negbin=None,
        x_gamma=None,
        y_gamma=None
    )
    
    config = GLMHMMConfig(
        num_states=K,
        num_em_iters=5,
        tol=1e-3,
        l2_reg_multinom=1.0 
    )
    
    try:
        params, lml_history = fit_glmhmm(data, config)
        print(f"✓ Multinomial observations test passed!")
        print(f"  Final log-likelihood: {lml_history[-1]:.2f}")
        return True
    except Exception as e:
        print(f"✗ Multinomial observations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("fit.py Compatibility Test Suite")
    print("=" * 60)
    
    results = []
    results.append(test_static_transitions())
    results.append(test_input_driven_transitions())
    results.append(test_multinomial_observations())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed! fit.py is compatible with other modules.")
    else:
        print("✗ Some tests failed. Please review the errors above.")
