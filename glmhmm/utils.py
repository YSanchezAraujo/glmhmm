import jax.numpy as jnp
from collections import namedtuple
from jax import jit

def session_info_neural(x_gauss_in_list):
    """Computes trial/segment info using jax.numpy."""
    n_trials = len(x_gauss_in_list)
    # Use jnp for array and cumsum
    samples_per_trial = jnp.array([x.shape[0] for x in x_gauss_in_list])
    last_idx = jnp.cumsum(samples_per_trial)
    first_idx = last_idx - samples_per_trial

    info = namedtuple(
        'info', 
        ['n_trials', 'samples_per_trial', 'last_idx', 'first_idx']
        )(n_trials=n_trials, samples_per_trial=samples_per_trial, last_idx=last_idx, first_idx=first_idx)
    return info

def compute_boundaries(list_of_array):
    samples_per_set = jnp.array([x.shape[0] for x in list_of_array])
    last_idx = jnp.cumsum(samples_per_set)
    first_idx = last_idx - samples_per_set

    inds = namedtuples('inds', ['n_trials', 'first', 'last'])(
        n_trials = samples_per_set, first = first_idx, last = last_idx
    )

    return inds

@jit
def transform_theta(theta):
    theta_prime = theta - theta[:, None, 0]
    return theta_prime