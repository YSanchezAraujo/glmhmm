import jax.numpy as jnp
from jax import random, vmap
from typing import List, NamedTuple, Optional, Union
from collections import namedtuple
from tqdm import tqdm
from jax.experimental import sparse

from .hmm_core import (compute_log_forward_message, 
                       compute_log_backward_message, 
                       compute_expectations)

from .emissions import (gaussian_glmhmm_loglikelihoods, 
                        normalized_gauss_loglikelihoods,
                        bernoulli_glmhmm_loglikelihood)

from .m_step import *
from .utils import compute_boundaries


def fit_glmhmm(
    n_states: int,
    dir_diag: jnp.ndarray,
    seed: int,
    # input driven transition data (required)
    x_trans_set: List[jnp.ndarray],
    # Gaussian data (optional)
    x_nested_set: Optional[Union[List[List[jnp.ndarray]], List[jnp.ndarray]]] = None,
    y_nested_set: Optional[Union[List[List[jnp.ndarray]], List[jnp.ndarray]]] = None,
    segment_gaussian: bool = True, 
    ridge_pen: float = 1.0,
    jitter: float = 1e-5,
    # Bernoulli data (optional)
    x_set_bern: Optional[List[jnp.ndarray]] = None,
    y_set_bern: Optional[List[jnp.ndarray]] = None,
    bern_m_step_maxiter: int = 100,
    tol_optax: float = 1e-3,
    # Common EM params
    max_iter: int = 500,
    em_tol: float = 1e-3
) -> NamedTuple:
    """
    Fits a Generalized Linear Model HMM (GLM-HMM) using Expectation-Maximization.

    Args:
        n_states: Number of hidden states.
        dir_diag: Diagonal enhancement for Dirichlet prior on transitions.
        seed: Random seed.
        x_trans_set: Input driven transition features (list per sequence).
        x_nested_set: Gaussian model input features (nested list per session/trial OR list per sequence depending on segment_gaussian).
        y_nested_set: Gaussian model observations (nested list per session/trial OR list per sequence depending on segment_gaussian).
        segment_gaussian: If True, sum Gaussian likelihoods over trials/segments.
                          If False, treat each Gaussian time point individually.
        ridge_pen: Ridge penalty for Gaussian M-step.
        jitter: Jitter added to covariance in Gaussian M-step.
        x_set_bern: Bernoulli model input features (list per sequence).
        y_set_bern: Bernoulli model observations (list per sequence).
        bern_m_step_maxiter: Max iterations for Bernoulli M-step optimizer.
        bern_lr_optax: Learning rate for Bernoulli M-step optimizer.
        max_iter: Maximum EM iterations.
        em_tol: Log-marginal likelihood tolerance for convergence.

    Returns:
        A NamedTuple containing the fitted parameters and results.
    """
    has_bern = x_set_bern is not None and y_set_bern is not None
    has_gauss = x_nested_set is not None and y_nested_set is not None

    if not has_bern and not has_gauss:
        raise ValueError("Must provide data for at least one model (Bernoulli or Gaussian).")

    x_set_gauss, y_set_gauss = None, None
    if has_gauss and segment_gaussian:
        x_set_gauss = [jnp.concatenate(x, axis=0) for x in x_nested_set]
        y_set_gauss = [jnp.concatenate(y, axis=0) for y in y_nested_set]
        n_feat_gauss, n_reg_gauss = x_set_gauss[0].shape[1], y_set_gauss[0].shape[1]
    else:
        x_set_gauss = x_nested_set 
        y_set_gauss = y_nested_set
        n_feat_gauss, n_reg_gauss = x_set_gauss[0].shape[1], y_set_gauss[0].shape[1]

    if has_bern and has_gauss:
        if len(x_nested_set) != len(x_set_bern):
            raise ValueError("Bernoulli and Gaussian data must have the same batch size.")
        batch_size = len(x_nested_set)
        for i in range(batch_size):
            if segment_gaussian:
                gauss_len = len(x_nested_set[i]) 
            else:
                gauss_len = x_set_gauss[i].shape[0] 

            if x_set_bern[i].shape[0] != gauss_len:
                 raise ValueError(f"Sequence {i}: Bernoulli length ({x_set_bern[i].shape[0]}) "
                                  f"must match Gaussian length ({gauss_len}) "
                                  f"for segment_gaussian={segment_gaussian}.")

    batch_size = len(x_trans_set)

    lml = []
    lml_prev = -jnp.inf
    key = random.PRNGKey(seed)
    key, key_theta = random.split(key, 2)
    A_inds = compute_boundaries(x_trans_set)

    x_trans_concat = jnp.concatenate(x_trans_set, axis=0)
    n_trans_steps, n_feat_trans = x_trans_concat.shape
    xi_over_time = jnp.zeros((n_trans_steps, n_states, n_states))
    theta = random.normal(key_theta, (n_feat_trans, n_states, n_states))
    A = compute_A_from_theta_and_inputs(x_trans_concat, theta)
    log_A = jnp.log(A)
    pi0 = jnp.ones(n_states) / n_states
    log_pi0 = jnp.log(pi0)

    W_bern, x_concat_bern, y_concat_bern = None, None, None
    num_rows_bern, first_ii_bern, last_ii_bern = None, None, None
    if has_bern:
        key, W_bern_key = random.split(key)
        x_concat_bern = jnp.concatenate(x_set_bern, axis=0)
        y_concat_bern = jnp.concatenate(y_set_bern, axis=0)
        num_rows_bern = jnp.array([x.shape[0] for x in x_set_bern])
        last_ii_bern = jnp.cumsum(num_rows_bern)
        first_ii_bern = last_ii_bern - num_rows_bern
        n_feat_bern = x_concat_bern.shape[1]
        W_bern_init = bern_init_opt(x_concat_bern, y_concat_bern, num_opt_steps=bern_m_step_maxiter, tol=tol_optax)
        W_bern = W_bern_init[:, jnp.newaxis] + (0.1 * random.normal(W_bern_key, (n_feat_bern, n_states)))

    W_gauss, Sigma = None, None
    neural_trial_info, first_idx_set, last_idx_set = None, None, None
    if has_gauss:
        key, W_gauss_key, Sigma_key = random.split(key, 3)
        gamma_set_init = [jnp.ones(y.shape[0]) for y in y_set_gauss]
        W_gauss_init = batch_ridge_regression(x_set_gauss, y_set_gauss, gamma_set_init, ridge_pen, n_feat_gauss, n_reg_gauss)
        Sigma_init = batch_full_covariance(W_gauss_init, x_set_gauss, y_set_gauss, gamma_set_init)
        W_gauss = W_gauss_init[:, :, jnp.newaxis] + (0.1 * random.normal(W_gauss_key, (n_feat_gauss, n_reg_gauss, n_states)))
        Sigma = Sigma_init[:, :, jnp.newaxis] + (jitter * random.normal(Sigma_key, (n_reg_gauss, n_reg_gauss, n_states)))
        if segment_gaussian:
            neural_trial_info = [compute_boundaries(x) for x in x_nested_set]
            first_idx_set = [jnp.asarray(nti.first_idx) for nti in neural_trial_info]
            last_idx_set = [jnp.asarray(nti.last_idx) for nti in neural_trial_info]

    print("Starting EM iterations...")
    for k in tqdm(range(max_iter), desc="EM Iteration"):
        gamma_set_all = []
        pi0_total = jnp.zeros(n_states)
        lml_total = 0.0
        gamma_set_gauss_m_step = []

        # E-step
        for i in range(batch_size):
            ll_bern = 0.0
            if has_bern:
                ll_bern = bernoulli_glmhmm_loglikelihood(x_set_bern[i], y_set_bern[i], W_bern)

            ll_gauss = 0.0
            segment_lens_i = None
            if has_gauss:
                raw_ll_gauss = vmap(gaussian_glmhmm_loglikelihoods, in_axes=(None, None, 2, 2))(
                    x_set_gauss[i], y_set_gauss[i], W_gauss, Sigma
                )
                if segment_gaussian:
                    segment_lens_i = last_idx_set[i] - first_idx_set[i]
                    ll_gauss = vmap(normalized_gauss_loglikelihoods, in_axes=(0, None, None, None))(
                        raw_ll_gauss, first_idx_set[i], last_idx_set[i], n_reg_gauss
                    ).T
                else:
                    ll_gauss = raw_ll_gauss.T

            ll = ll_bern + ll_gauss 
            log_alpha, log_c = compute_log_forward_message(ll, log_pi0, log_A[A_inds.first_idx:A_inds.last_idx, :, :])
            log_beta = compute_log_backward_message(ll, log_A[A_inds.first_idx:A_inds.last_idx, :, :], log_c)
            xi_i, gamma_i = compute_expectations(log_alpha, log_beta, log_c, ll, log_A[A_inds.first_idx:A_inds.last_idx, :, :])

            xi_over_time.at[A_inds.first_idx:A_inds.last_idx, :, :].set(xi_i)
            gamma_set_all.append(gamma_i)
            pi0_total += gamma_i[0, :]
            lml_total += jnp.sum(log_c)

            if has_gauss:
                if segment_gaussian:
                    expanded_i = jnp.repeat(gamma_i, segment_lens_i, axis=0)
                    gamma_set_gauss_m_step.append(expanded_i)
                else:
                    gamma_set_gauss_m_step.append(gamma_i)

        # M-step
        theta, _ = transitions_m_step_optax(x_trans_concat, xi_over_time, theta, dir_diag)
        A = compute_A_from_theta_and_inputs(x_trans_concat, theta)
        log_A = jnp.log(A)
        pi0 = pi0_total / jnp.sum(pi0_total)
        log_pi0 = jnp.log(pi0)
        gamma = jnp.concatenate(gamma_set_all, axis=0)

        if has_bern:
            W_bern, _ = bern_m_step_optax(
                x_concat_bern, y_concat_bern, gamma, W_bern,
                num_opt_steps=bern_m_step_maxiter, tol=tol_optax
            )

        if has_gauss:
            W_gauss, Sigma = batch_m_step_updates(
                x_set_gauss, y_set_gauss, gamma_set_gauss_m_step,
                ridge_pen, n_feat_gauss, n_reg_gauss, n_states, jitter=jitter
            )

        lml.append(lml_total)
        if jnp.abs(lml_prev - lml_total) < em_tol and k > 0:
            print(f"Converged in {k} iterations.")
            break
        lml_prev = lml_total

    output_fields = ["transition_matrix", "initial_transition", "log_marginal_likelihood"]
    output_values = [A, pi0, lml]

    output_fields.append("state_posterior_list")
    output_values.append(gamma_set_all)
    output_fields.append("state_posterior_concat")
    output_values.append(gamma) 

    if has_bern:
        output_fields.append("W_bern")
        output_values.append(W_bern)
        output_fields.extend(["bern_first_idx", "bern_last_idx"])
        output_values.extend([first_ii_bern, last_ii_bern])

    if has_gauss:
        output_fields.extend(["W_gauss", "covariance"])
        output_values.extend([W_gauss, Sigma])
        if segment_gaussian:
            output_fields.append("trial_info")
            output_values.append(neural_trial_info)

    output = namedtuple('glmhmm_fit', output_fields)(*output_values)

    return output
