import jax.numpy as jnp
from collections import namedtuple

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
