import jax.numpy as jnp
from collections import namedtuple
from jax import jit

def compute_boundaries(list_of_array):
    """Computes trial/segment info using jax.numpy."""
    n_trials = len(list_of_array)
    samples_per_trial = jnp.array([x.shape[0] for x in list_of_array])
    last_idx = jnp.cumsum(samples_per_trial)
    first_idx = last_idx - samples_per_trial

    ind_info = namedtuple(
        'info', 
        ['n_trials', 'samples_per_trial', 'last_idx', 'first_idx']
        )(n_trials=n_trials, samples_per_trial=samples_per_trial, last_idx=last_idx, first_idx=first_idx)
    
    return ind_info

# will prob remove this, computation is already simple
@jit
def transform_theta(theta, ref_state_ind):
    theta_prime = theta - theta[:, None, ref_state_ind]
    return theta_prime

@jit
def transform_theta(theta):
    theta_prime = theta - theta[:, None, 0]
    return theta_prime
