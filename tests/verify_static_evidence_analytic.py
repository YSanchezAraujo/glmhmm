
import jax
import jax.numpy as jnp
from jax import random
import scipy.special
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from glmhmm.structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence

def verify_static_evidence():
    print("Verifying Analytic Static Transition Evidence...")
    
    # Setup
    K = 2
    stickiness = 1.0
    config = GLMHMMConfig(
        num_states=K,
        num_em_iters=1,
        transition_stickiness=stickiness,
        prior_var_bias_trans=10.0, # Not used for analytic but required by config
        prior_var_bias_emit=10.0
    )
    
    # Params (Dummy)
    params = GLMHMMParams(
        log_pi0=jnp.zeros(K),
        log_A=jnp.zeros((K, K)), # Not used
        theta=None,
        W_bern=None,
        W_multinom=None,
        W_negbin=None,
        phi_negbin=None,
        W_gamma=None,
        alpha_gamma=None,
        W_gauss=jnp.zeros((1, 1, K)), # Dummy
        Sigma_gauss=jnp.ones((1, 1, K)) # Dummy
    )
    
    # Data (Dummy)
    # Need x_gauss to satisfy shapes, or just ensure correct handling
    # We'll use x_gauss as the driver.
    T = 100
    data = GLMHMMData(
        x_gauss=[jnp.zeros((T, 1))],
        y_gauss=[jnp.zeros((T, 1))],
        x_trans=None,
        gauss_first_idx=None,
        gauss_last_idx=None
    )
    
    # Transitions Counts (xis)
    # Create xis such that sum is exactly what we want.
    # Row 0: [10, 5]
    # Row 1: [2, 8]
    
    xis_sum = jnp.array([[10, 5], 
                         [2, 8]], dtype=jnp.float32)
    
    # We need xis to be (N, T-1, K, K) or list.
    # We can just pass the sum directly if we were calling the inner logic, 
    # but the function expects xis.
    # Let's create a single frame xi with the counts.
    # xis shape: (1, 1, K, K)
    xis = [xis_sum[jnp.newaxis, :, :]] # List of (T-1, K, K)
    
    # Gammas (Needed for other parts, but we can ignore them if we only care about trans evidence)
    # But compute_factorized_evidence computes everything. 
    # We need to isolate log_Z_trans.
    # The function returns total log_Z. 
    # However, it prints "  [BQ Debug K=2] log_Z_trans: ..."
    # So we can capture stdout or just trust the components if we make others 0?
    # No, others won't be 0.
    
    # Better: Inspect the printed output or modify the function to return components?
    # I can't modify the function easily just for test.
    # The function returns (total_log_Z, total_var).
    # Since I set W_bern etc to None, and W_gauss is present, log_Z_glm will be computed.
    # If I set gammas to zeros, log_Z_glm might be close to 0 or prior value.
    
    gammas = jnp.zeros((1, T, K))
    
    # To isolate log_Z_trans, we could check the printed output.
    # Or, we can manually compute the expected Log Z total, assuming we can compute the others.
    # Actually, simpler: Set config.num_states=K
    
    # Let's calculate the expected value for transitions
    def log_beta(alpha):
        return jnp.sum(jax.scipy.special.gammaln(alpha)) - jax.scipy.special.gammaln(jnp.sum(alpha))
        
    alpha_row0 = jnp.array([1.0, 1.0])
    alpha_row0 = alpha_row0.at[0].add(stickiness) # [2, 1]
    
    alpha_row1 = jnp.array([1.0, 1.0])
    alpha_row1 = alpha_row1.at[1].add(stickiness) # [1, 2]
    
    counts0 = jnp.array([10.0, 5.0])
    counts1 = jnp.array([2.0, 8.0])
    
    expected_row0 = log_beta(alpha_row0 + counts0) - log_beta(alpha_row0)
    expected_row1 = log_beta(alpha_row1 + counts1) - log_beta(alpha_row1)
    
    expected_log_Z_trans = expected_row0 + expected_row1
    
    print(f"Calculated Expected log_Z_trans: {expected_log_Z_trans:.4f}")
    
    # Run Function
    # We need to capture stdout to see the specific component, 
    # or we can trust that if the total matches (expected_trans + expected_others), we are good.
    # But calculating expected_others is annoying.
    
    # Let's rely on the capture of stdout which I can see in the tool output.
    
    log_Z, _ = compute_factorized_evidence(data, params, config, gammas, xis)
    
    # We will rely on the printed logs from the function to verify log_Z_trans matches.
    
if __name__ == "__main__":
    verify_static_evidence()
