import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from tqdm import tqdm
from typing import NamedTuple, List, Tuple, Optional

from .structs import GLMHMMParams, GLMHMMData, GLMHMMConfig

from .emissions import (bernoulli_glmhmm_loglikelihood, 
                        multinomial_glmhmm_loglikelihood,
                        negbin_glmhmm_loglikelihood, 
                        gamma_glmhmm_loglikelihood,
                        gaussian_glmhmm_loglikelihoods,
                        normalized_gauss_loglikelihoods)

from .hmm_core import (compute_log_forward_message, 
                       compute_log_backward_message, 
                       compute_expectations)

from .m_step import *
from .utils import compute_A_from_theta_and_inputs
from .initialize import initialize_params

def gauss_normalization_default(data: GLMHMMData) -> bool:
    """
    Whether the Gaussian log-likelihood is rescaled onto the same footing as the
    other emission families.

    When Gaussian data is fit alongside a behavioural family, each segment's summed
    Gaussian log-likelihood is divided by the segment length and by the number of
    regions, so that many neural samples do not swamp a single behavioural
    observation. Override with GLMHMMConfig.normalize_gauss_loglik.

    NOTE: this rescaling is a heuristic reweighting, not a likelihood. When it is
    active, compute_log_marginal_likelihood returns a pseudo log marginal likelihood
    rather than log p(Y). The same rescaling is applied consistently in the E-step,
    in Viterbi decoding, and in the factorized evidence estimators, so quantities
    are comparable across them; they are not comparable to an unnormalized fit.
    It does not enter the closed-form Gaussian M-step, which remains a
    responsibility-weighted ridge regression on the raw samples.

    Args:
        data: GLMHMMData object.

    Returns:
        True if any non-Gaussian emission family is present.
    """
    return any(
        f is not None
        for f in (data.x_bern, data.x_multinom, data.x_negbin, data.x_gamma)
    )


@partial(jit, static_argnames=("normalize_gauss",))
def compute_total_emissions(
    params: GLMHMMParams,
    x_b, y_b,
    x_mn, y_mn,
    x_nb, y_nb,
    x_gm, y_gm,
    x_gs, y_gs, gs_first, gs_last, gs_segment_ids,
    normalize_gauss: Optional[bool] = None
    ) -> jnp.ndarray:
    """
    Computes total log-likelihoods by summing across all active emission distributions.

    Args:
        normalize_gauss: Whether to rescale segmented Gaussian log-likelihoods.
                         None (default) rescales whenever another family is active.
                         See gauss_normalization_default.
    """

    if x_b is not None:
        T = x_b.shape[0]
    elif x_mn is not None:
        T = x_mn.shape[0]
    elif x_nb is not None:
        T = x_nb.shape[0]
    elif x_gm is not None:
        T = x_gm.shape[0]
    elif gs_first is not None:
        T = gs_first.shape[0]
    elif x_gs is not None:
        T = x_gs.shape[0]
    else:
        T = 0 

    K = params.log_pi0.shape[0]
    ll_total = jnp.zeros((T, K))

    if params.W_bern is not None and x_b is not None:
        ll_total += bernoulli_glmhmm_loglikelihood(x_b, y_b, params.W_bern)

    if params.W_multinom is not None and x_mn is not None:
        ll_total += multinomial_glmhmm_loglikelihood(x_mn, y_mn, params.W_multinom)
        
    if params.W_negbin is not None and x_nb is not None:
         ll_total += negbin_glmhmm_loglikelihood(x_nb, y_nb, params.W_negbin, params.phi_negbin)

    if params.W_gamma is not None and x_gm is not None:
         ll_total += gamma_glmhmm_loglikelihood(x_gm, y_gm, params.W_gamma, params.alpha_gamma)
         
    if params.W_gauss is not None and x_gs is not None:
        loglik_fn = jax.vmap(gaussian_glmhmm_loglikelihoods, in_axes=(None, None, 2, 2), out_axes=1)
        raw_ll = loglik_fn(x_gs, y_gs, params.W_gauss, params.Sigma_gauss) # (N_total, K)
        
        has_other = False
        if x_b is not None: 
            has_other = True
        elif x_mn is not None: 
            has_other = True
        elif x_nb is not None: 
            has_other = True
        elif x_gm is not None: 
            has_other = True
        
        do_normalize = has_other if normalize_gauss is None else normalize_gauss
        n_regions = y_gs.shape[1]
        
        if gs_first is not None:
             norm_fn = jax.vmap(normalized_gauss_loglikelihoods, in_axes=(1, None, None, None, None, None), out_axes=1)
             normalized_ll = norm_fn(raw_ll, gs_first, gs_last, n_regions, gs_segment_ids, do_normalize) # (T, K)
        else:
             normalized_ll = raw_ll
        
        ll_total += normalized_ll
    
    return ll_total

def compute_log_marginal_likelihood(
    params: GLMHMMParams,
    data: GLMHMMData,
    normalize_gauss: Optional[bool] = None
) -> float:
    """
    Computes the total log marginal likelihood of the data given the parameters.
    Sum of log_c from the forward pass across all sequences.

    Args:
        params: GLMHMM parameters.
        data: Dataset containing all observation sequences and covariates.
        normalize_gauss: Gaussian log-lik rescaling; see gauss_normalization_default.
                         When active the return value is a pseudo log marginal
                         likelihood rather than log p(Y).
    """
    total_lml = 0.0

    seq_sources = [data.x_trans, data.x_bern, data.x_multinom, data.x_negbin, data.x_gamma]

    if data.x_gauss is not None:
         if isinstance(data.x_gauss, list) and len(data.x_gauss) > 0:
             n_sequences = len(data.x_gauss)
         else:
             n_sequences = 0
             if n_sequences == 0:
                  n_sequences = next((len(s) for s in seq_sources if s is not None), 0)
    else:
        n_sequences = next((len(s) for s in seq_sources if s is not None), 0)

    for i in range(n_sequences): 
        x_b  = data.x_bern[i] if data.x_bern is not None else None
        y_b  = data.y_bern[i] if data.y_bern is not None else None
        x_mn = data.x_multinom[i] if data.x_multinom is not None else None
        y_mn = data.y_multinom[i] if data.y_multinom is not None else None
        x_nb = data.x_negbin[i] if data.x_negbin is not None else None
        y_nb = data.y_negbin[i] if data.y_negbin is not None else None
        x_gm  = data.x_gamma[i] if data.x_gamma is not None else None
        y_gm  = data.y_gamma[i] if data.y_gamma is not None else None
        
        if data.x_gauss is not None and isinstance(data.x_gauss, list):
             if isinstance(data.x_gauss[i], (list, tuple)):
                 x_gs = jnp.concatenate(data.x_gauss[i], axis=0)
                 y_gs = jnp.concatenate(data.y_gauss[i], axis=0)
             else:
                 x_gs = data.x_gauss[i]
                 y_gs = data.y_gauss[i]
        else:
             x_gs, y_gs = None, None

        gs_first = data.gauss_first_idx[i] if data.gauss_first_idx else None
        gs_last = data.gauss_last_idx[i] if data.gauss_last_idx else None
        
        gs_segment_ids = None
        if gs_first is not None:
            n_segs = gs_first.shape[0]
            lengths = gs_last - gs_first
            gs_segment_ids = jnp.repeat(jnp.arange(n_segs), lengths)

        ll_t = compute_total_emissions(params, x_b, y_b, x_mn, y_mn, x_nb, y_nb, x_gm, y_gm, x_gs, y_gs, gs_first, gs_last, gs_segment_ids, normalize_gauss)

        if params.theta is not None:
            data_x_trans_i = data.x_trans[i] if data.x_trans is not None else None

            if data_x_trans_i is None:
                 raise ValueError("params.theta is present but data.x_trans is None")
            A_t = compute_A_from_theta_and_inputs(data_x_trans_i, params.theta)
            log_A_t = jnp.log(A_t)
        else:
            T_len = ll_t.shape[0]
            K = params.log_A.shape[0]
            log_A_t = jnp.broadcast_to(params.log_A[jnp.newaxis, :, :], (T_len, K, K))
            
        _, log_c = compute_log_forward_message(ll_t, params.log_pi0, log_A_t)
        
        total_lml += jnp.sum(log_c)

    return total_lml


def e_step(
    params: GLMHMMParams,
    data: GLMHMMData,
    normalize_gauss: Optional[bool] = None
):
    """
    Performs the E-step of the EM algorithm for GLMHMM.

    Computes posterior probabilities (responsibilities) for states and transitions
    using the forward-backward algorithm. Processes all sequences in the dataset.

    Args:
        params: Current GLMHMM parameters.
        data: Dataset containing all observation sequences and covariates.
        normalize_gauss: Gaussian log-lik rescaling; see gauss_normalization_default.

    Returns:
        gammas: List of state responsibilities for each sequence.
                Each element has shape (T_i, K) where T_i is the length of sequence i.
        xis: List of transition responsibilities for each sequence.
             Each element has shape (T_i-1, K, K).
        total_lml: Total log marginal likelihood across all sequences (scalar).
    """
    total_lml = 0.0
    gammas, xis = [], []
    
    seq_sources = [data.x_trans, data.x_bern, data.x_multinom, data.x_negbin, data.x_gamma]

    if data.x_gauss is not None and isinstance(data.x_gauss, list) and len(data.x_gauss) > 0:
          n_sequences = len(data.x_gauss)
    else:
          n_sequences = next((len(s) for s in seq_sources if s is not None), 0)

    for i in range(n_sequences): 
        x_b  = data.x_bern[i] if data.x_bern is not None else None
        y_b  = data.y_bern[i] if data.y_bern is not None else None
        x_mn = data.x_multinom[i] if data.x_multinom is not None else None
        y_mn = data.y_multinom[i] if data.y_multinom is not None else None
        x_nb = data.x_negbin[i] if data.x_negbin is not None else None
        y_nb = data.y_negbin[i] if data.y_negbin is not None else None
        x_gm  = data.x_gamma[i] if data.x_gamma is not None else None
        y_gm  = data.y_gamma[i] if data.y_gamma is not None else None
        
        if data.x_gauss is not None and isinstance(data.x_gauss, list):
             if isinstance(data.x_gauss[i], (list, tuple)):
                 x_gs = jnp.concatenate(data.x_gauss[i], axis=0)
                 y_gs = jnp.concatenate(data.y_gauss[i], axis=0)
             else:
                 x_gs = data.x_gauss[i]
                 y_gs = data.y_gauss[i]
        else:
             x_gs, y_gs = None, None

        gs_first = data.gauss_first_idx[i] if data.gauss_first_idx else None
        gs_last = data.gauss_last_idx[i] if data.gauss_last_idx else None
        
        gs_segment_ids = None
        if gs_first is not None:
            n_segs = gs_first.shape[0]
            lengths = gs_last - gs_first
            gs_segment_ids = jnp.repeat(jnp.arange(n_segs), lengths)

        ll_t = compute_total_emissions(params, x_b, y_b, x_mn, y_mn, x_nb, y_nb, x_gm, y_gm, x_gs, y_gs, gs_first, gs_last, gs_segment_ids, normalize_gauss)

        if params.theta is not None:
            data_x_trans_i = data.x_trans[i] if data.x_trans is not None else None
            A_t = compute_A_from_theta_and_inputs(data_x_trans_i, params.theta)
            log_A_t = jnp.log(A_t)
        else:
            T_len = ll_t.shape[0]
            K = params.log_A.shape[0]
            log_A_t = jnp.broadcast_to(params.log_A[jnp.newaxis, :, :], (T_len, K, K))
            
        log_alpha, log_c = compute_log_forward_message(ll_t, params.log_pi0, log_A_t)
        log_beta = compute_log_backward_message(ll_t, log_A_t, log_c)
        xi_t, gamma_t = compute_expectations(log_alpha, log_beta, log_c, ll_t, log_A_t)
        
        total_lml += jnp.sum(log_c)
        gammas.append(gamma_t)
        xis.append(xi_t)
        
    return gammas, xis, total_lml

def m_step(
    params: GLMHMMParams, 
    data: GLMHMMData, 
    gammas: List[jnp.ndarray], 
    xis: List[jnp.ndarray], 
    config: GLMHMMConfig
    ) -> GLMHMMParams:
    """
    Performs the M-step of the EM algorithm for GLMHMM.
    
    Updates all model parameters by maximizing the expected complete-data log-likelihood
    using the responsibilities computed in the E-step. Each emission distribution and
    the transition parameters are updated independently.
    
    Args:
        params: Current GLMHMM parameters.
        data: Dataset containing all observation sequences and covariates.
        gammas: State responsibilities from E-step, list of arrays with shape (T_i, K).
        xis: Transition responsibilities from E-step, list of arrays with shape (T_i-1, K, K).
        config: Configuration containing regularization parameters and other settings.
    
    Returns:
        Updated GLMHMM parameters with optimized emission weights, dispersion parameters,
        and transition parameters (theta or log_A depending on model type).
    """
    gamma_concat = jnp.concatenate(gammas, axis=0)
    
    # Calculate bias ridges from config
    bias_ridge_emit = 1.0 / config.prior_var_bias_emit
    bias_ridge_trans = 1.0 / config.prior_var_bias_trans
    
    new_W_bern = params.W_bern
    if params.W_bern is not None:
        X_all = jnp.concatenate(data.x_bern, axis=0)
        Y_all = jnp.concatenate(data.y_bern, axis=0)
        new_W_bern, _ = bern_m_step_optax(
            X_all, Y_all, gamma_concat, params.W_bern, config.l2_reg_bern, bias_ridge_emit
        )

    new_W_multinom = params.W_multinom
    if params.W_multinom is not None:
        X_all = jnp.concatenate(data.x_multinom, axis=0)
        Y_all = jnp.concatenate(data.y_multinom, axis=0)
        new_W_multinom, _ = multinomial_obs_m_step_optax(
            X_all, Y_all, gamma_concat, params.W_multinom, config.l2_reg_multinom, bias_ridge_emit
        )

    new_W_nb, new_phi_nb = params.W_negbin, params.phi_negbin
    if params.W_negbin is not None:
        X_all = jnp.concatenate(data.x_negbin, axis=0)
        Y_all = jnp.concatenate(data.y_negbin, axis=0)
        new_W_nb, new_phi_nb, _ = negbin_m_step_optax(
            X_all, Y_all, gamma_concat, params.W_negbin, params.phi_negbin, config.l2_reg_negbin, bias_ridge_emit
        )

    new_W_gamma, new_alpha_gamma = params.W_gamma, params.alpha_gamma
    if params.W_gamma is not None:
            X_all = jnp.concatenate(data.x_gamma, axis=0)
            Y_all = jnp.concatenate(data.y_gamma, axis=0)
            new_W_gamma, new_alpha_gamma, _ = gamma_m_step_optax(
                X_all, Y_all, gamma_concat, params.W_gamma, params.alpha_gamma, config.l2_reg_gamma, bias_ridge_emit
            )

    new_W_gauss, new_Sigma_gauss = params.W_gauss, params.Sigma_gauss
    if params.W_gauss is not None:
         if isinstance(data.x_gauss[0], list):
             x_gs_list = [jnp.concatenate(seq, axis=0) for seq in data.x_gauss]
             y_gs_list = [jnp.concatenate(seq, axis=0) for seq in data.y_gauss]
         else:
             x_gs_list = data.x_gauss
             y_gs_list = data.y_gauss

         expanded_gammas = []
         for i, g in enumerate(gammas):
             if data.gauss_first_idx is not None:
                 first = data.gauss_first_idx[i]
                 last = data.gauss_last_idx[i]
                 repeats = last - first
                 total_len = x_gs_list[i].shape[0]
                 g_exp = jnp.repeat(g, repeats, axis=0, total_repeat_length=total_len)
                 expanded_gammas.append(g_exp)
             else:
                 expanded_gammas.append(g)
             
         D_g = x_gs_list[0].shape[1]
         M_g = y_gs_list[0].shape[1]
         K = params.log_pi0.shape[0]
         
         new_W_gauss, new_Sigma_gauss = batch_m_step_updates(
             x_gs_list, y_gs_list, expanded_gammas,
             config.l2_reg_gauss, D_g, M_g, K, bias_ridge=bias_ridge_emit
         )

    new_theta, new_log_A = params.theta, params.log_A
    if params.theta is not None:
        X_trans_all = jnp.concatenate(data.x_trans, axis=0)
        xi_all = jnp.concatenate(xis, axis=0)
        new_theta, _ = transitions_m_step_optax_reduced(
             X_trans_all, xi_all, params.theta, jnp.ones(config.num_states), bias_ridge_trans, config.l2_reg_trans
        )
    else:
        xi_sum = jnp.sum(jnp.concatenate(xis, axis=0), axis=0)
        if config.transition_stickiness > 0.0:
            diag_indices = jnp.arange(xi_sum.shape[0])
            xi_sum = xi_sum.at[diag_indices, diag_indices].add(config.transition_stickiness)
            
        new_A = xi_sum / jnp.sum(xi_sum, axis=1, keepdims=True)
        new_log_A = jnp.log(new_A + 1e-9)
    
    new_log_pi0 = initial_distribution_m_step(gammas, dirichlet_alpha=1.0001)

    return GLMHMMParams(
        log_pi0=new_log_pi0, 
        log_A=new_log_A,
        theta=new_theta,
        W_bern=new_W_bern,
        W_multinom=new_W_multinom,
        W_negbin=new_W_nb,
        phi_negbin=new_phi_nb,
        W_gamma=new_W_gamma,
        alpha_gamma=new_alpha_gamma,
        W_gauss=new_W_gauss,
        Sigma_gauss=new_Sigma_gauss
    )

def _validate_data_shapes(data: GLMHMMData):
    """
    Validates that x_trans has length T-1 relative to observations of length T.
    """
    if data.x_trans is None:
        return

    seq_sources = [data.x_bern, data.x_multinom, data.x_negbin, data.x_gamma, data.x_gauss]
    obs_source = next((s for s in seq_sources if s is not None), None)
    
    if obs_source is None:
        return

    n_sequences = len(obs_source)
    if len(data.x_trans) != n_sequences:
        raise ValueError(f"Number of x_trans sequences ({len(data.x_trans)}) does not match number of observation sequences ({n_sequences}).")

    for i in range(n_sequences):
        if data.x_bern is not None: 
            T = data.x_bern[i].shape[0]
        elif data.x_multinom is not None: 
            T = data.x_multinom[i].shape[0]
        elif data.x_negbin is not None: 
            T = data.x_negbin[i].shape[0]
        elif data.x_gamma is not None: 
            T = data.x_gamma[i].shape[0]
        elif data.x_gauss is not None:
             if data.gauss_first_idx is not None:
                 T = data.gauss_first_idx[i].shape[0]
             else:
                 T = data.x_gauss[i].shape[0]
        else:
             continue

        T_trans = data.x_trans[i].shape[0]
        if T_trans != T - 1:
            raise ValueError(
                f"Shape mismatch in sequence {i}: Observations have length T={T}, "
                f"but x_trans has length {T_trans}. Input-driven transitions operate "
                f"on the interval t -> t+1, so x_trans must have length T-1."
            )

def fit_glmhmm(
    data: GLMHMMData, 
    config: GLMHMMConfig, 
    initial_params: Optional[GLMHMMParams] = None
    ):
    """
    Fits a Generalized Linear Model Hidden Markov Model using the EM algorithm.
    
    This is the main entry point for training a GLMHMM. It alternates between:
    - E-step: Computing state/transition responsibilities using forward-backward
    - M-step: Updating parameters to maximize expected complete-data likelihood
    
    The algorithm continues until convergence (change in log-likelihood < tolerance)
    or the maximum number of iterations is reached.
    
    Args:
        data: GLMHMMData object containing:
            - Observation sequences for each emission type
            - Covariate sequences for emissions and transitions
            All sequences should have the same length within each list.
        config: GLMHMMConfig containing:
            - num_states: Number of hidden states (K)
            - num_em_iters: Maximum EM iterations
            - tol: Convergence tolerance for log-likelihood
            - l2_reg_*: Regularization parameters for each distribution
        initial_params: Optional GLMHMMParams object to initialize the model. 
                        If None, parameters are initialized randomly.
    
    Returns:
        params: Fitted GLMHMM parameters including:
            - log_pi0: Initial state log-probabilities
            - log_A or theta: Transition parameters
            - W_*: Emission weights for each active distribution
            - phi_negbin, alpha_gamma: Dispersion parameters if applicable
        lml_history: List of log marginal likelihoods, one per EM iteration plus a
                     final entry scoring the returned parameters. lml_history[-1]
                     therefore always corresponds to the params that are returned.

    Example:
        >>> data = GLMHMMData(x_bern=[X1, X2], y_bern=[y1, y2], ...)
        >>> config = GLMHMMConfig(num_states=3, num_em_iters=100, l2_reg_bern=1.0)
        >>> params, lml_history = fit_glmhmm(data, config)
    """
    _validate_data_shapes(data)

    normalize_gauss = config.normalize_gauss_loglik

    if initial_params is None:
        params = initialize_params(data, config, seed=config.seed)
    else:
        params = initial_params
    lml_history = []
    converged = False

    print("Starting EM...")
    for epoch in tqdm(range(config.num_em_iters)):

        gammas, xis, lml = e_step(params, data, normalize_gauss)

        # record before breaking, so the final LML always reaches the history
        converged = (
            epoch > config.min_em_iters
            and abs(lml - lml_history[-1]) < config.tol
        )
        lml_history.append(lml)

        if converged:
            print(f"Converged at epoch {epoch}")
            break

        params = m_step(params, data, gammas, xis, config)

    if not converged:
        # the loop exited straight after an M-step (or never ran), so the last
        # recorded LML predates the returned params; score them explicitly
        lml_history.append(
            compute_log_marginal_likelihood(params, data, normalize_gauss)
        )

    if params.theta is not None:
        params = params._replace(log_A=None)
        
    return params, lml_history

def fit_with_restarts(
    data: GLMHMMData, 
    config: GLMHMMConfig, 
    n_fits: int = 5, 
    seeds: Optional[List[int]] = None
    ) -> Tuple[GLMHMMParams, List[float]]:
    """
    Runs fit_glmhmm multiple times with different random seeds and returns the best fit.
    
    Args:
        data: GLMHMMData object.
        config: GLMHMMConfig object.
        n_fits: Number of independent fits to run.
        seeds: Optional list of seeds. If None, generated randomly.
    
    Returns:
        best_params: Parameters achieving the highest LML.
        best_lml_hist: LML history of the best fit.
    """
    import numpy as np
    
    if seeds is None:
        rng = np.random.default_rng(config.seed)
        seeds = rng.integers(0, 100000, size=n_fits).tolist()
    
    if len(seeds) != n_fits:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_fits ({n_fits})")
        
    best_lml = -np.inf
    best_params = None
    best_hist = None
    
    print(f"Running {n_fits} fits with restarts...")
    
    for i, seed in enumerate(seeds):
        print(f"Fit {i+1}/{n_fits} (Seed: {seed})...")
        current_config = config._replace(seed=seed)
        
        try:
            params, lml_hist = fit_glmhmm(data, current_config)
            final_lml = lml_hist[-1]
            
            if final_lml > best_lml:
                best_lml = final_lml
                best_params = params
                best_hist = lml_hist
                print(f"New best LML: {best_lml:.4f}")
            else:
                print(f"LML: {final_lml:.4f} (Best: {best_lml:.4f})")
                
        except Exception as e:
            print(f"Fit failed with error: {e}")
            
    if best_params is None:
        raise RuntimeError("All fits failed.")
        
    print(f"Best LML achieved: {best_lml:.4f}")
    return best_params, best_hist