"""
Cross-Validation Model Selection for GLM-HMM.

This module implements cross-validation for GLM-HMM model selection.
It supports:
- Sequence-level splitting (KFold) for multiple independent sequences.
- Temporal splitting (TimeSeriesSplit) for single long sequences.

Two scoring rules are available:
- cross_validate_k: plug-in predictive log p(Y_test | Theta_hat_train).
- posterior_predictive_cv: posterior predictive log p(Y_test | Y_train),
  obtained as a difference of marginal likelihoods.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional, Callable, Dict
from sklearn.model_selection import KFold, TimeSeriesSplit

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import fit_glmhmm, fit_with_restarts, compute_log_marginal_likelihood, e_step


def subset_data(data: GLMHMMData, indices: List[int]) -> GLMHMMData:
    """
    Creates a new GLMHMMData object with only the specified sequence indices.
    
    Args:
        data: Original GLMHMMData object (with list-structured sequences)
        indices: List of sequence indices to include
        
    Returns:
        GLMHMMData with only the specified sequences
    """
    def subset_list(lst, idx):
        if lst is None:
            return None
        if isinstance(lst, list):
            return [lst[i] for i in idx]
        else:
            # Handle non-list (single array) case
            return lst
    
    return GLMHMMData(
        x_bern=subset_list(data.x_bern, indices),
        y_bern=subset_list(data.y_bern, indices),
        x_trans=subset_list(data.x_trans, indices),
        x_gauss=subset_list(data.x_gauss, indices),
        y_gauss=subset_list(data.y_gauss, indices),
        x_multinom=subset_list(data.x_multinom, indices),
        y_multinom=subset_list(data.y_multinom, indices),
        x_negbin=subset_list(data.x_negbin, indices),
        y_negbin=subset_list(data.y_negbin, indices),
        x_gamma=subset_list(data.x_gamma, indices),
        y_gamma=subset_list(data.y_gamma, indices),
    )


def slice_single_sequence(data: GLMHMMData, start: int, end: int) -> GLMHMMData:
    """
    Slices a single-sequence GLMHMMData temporally (for TimeSeriesSplit).
    
    Args:
        data: GLMHMMData with a single sequence (arrays, not lists)
        start: Start time index
        end: End time index
        
    Returns:
        GLMHMMData with the time slice [start:end]
    """
    def slice_arr(arr, s, e):
        if arr is None:
            return None
        if isinstance(arr, list):
            # If it's a list of one element, slice the inner array
            return [arr[0][s:e]]
        return arr[s:e]
    
    # For x_trans, we need to be careful - it's T-1 length
    def slice_trans(arr, s, e):
        if arr is None:
            return None
        if isinstance(arr, list):
            return [arr[0][max(0, s-1):max(0, e-1)]]
        return arr[max(0, s-1):max(0, e-1)]
    
    return GLMHMMData(
        x_bern=slice_arr(data.x_bern, start, end),
        y_bern=slice_arr(data.y_bern, start, end),
        x_trans=slice_trans(data.x_trans, start, end),
        x_gauss=slice_arr(data.x_gauss, start, end),
        y_gauss=slice_arr(data.y_gauss, start, end),
        x_multinom=slice_arr(data.x_multinom, start, end),
        y_multinom=slice_arr(data.y_multinom, start, end),
        x_negbin=slice_arr(data.x_negbin, start, end),
        y_negbin=slice_arr(data.y_negbin, start, end),
        x_gamma=slice_arr(data.x_gamma, start, end),
        y_gamma=slice_arr(data.y_gamma, start, end),
    )


def get_num_sequences(data: GLMHMMData) -> int:
    """Returns the number of sequences in the data."""
    if data.x_bern is not None:
        return len(data.x_bern) if isinstance(data.x_bern, list) else 1
    if data.y_bern is not None:
        return len(data.y_bern) if isinstance(data.y_bern, list) else 1
    if data.x_trans is not None:
        return len(data.x_trans) if isinstance(data.x_trans, list) else 1
    if data.x_gauss is not None:
        return len(data.x_gauss) if isinstance(data.x_gauss, list) else 1
    if data.x_multinom is not None:
        return len(data.x_multinom) if isinstance(data.x_multinom, list) else 1
    raise ValueError("No valid data found to determine number of sequences")


def get_sequence_length(data: GLMHMMData) -> int:
    """Returns the length of the first sequence (for single-sequence CV)."""
    if data.x_bern is not None:
        arr = data.x_bern[0] if isinstance(data.x_bern, list) else data.x_bern
        return arr.shape[0]
    if data.y_bern is not None:
        arr = data.y_bern[0] if isinstance(data.y_bern, list) else data.y_bern
        return arr.shape[0]
    if data.x_trans is not None:
        arr = data.x_trans[0] if isinstance(data.x_trans, list) else data.x_trans
        return arr.shape[0] + 1  # x_trans is T-1
    raise ValueError("No valid data found")


def cross_validate_k(
    data: GLMHMMData,
    config: GLMHMMConfig,
    n_folds: int = 5,
    n_fits: int = 3,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[float, float, List[float]]:
    """
    Performs K-fold cross-validation for a GLM-HMM model.
    
    Uses sklearn's KFold for robust fold splitting. For each fold,
    fits the model on training data and evaluates on held-out test data.
    
    Args:
        data: GLMHMMData object with all sequences
        config: GLMHMMConfig specifying model (including num_states)
        n_folds: Number of CV folds (default: 5)
        n_fits: Number of random restarts per fold (default: 3)
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        mean_test_ll: Mean test log-likelihood across folds
        std_test_ll: Standard deviation of test log-likelihood
        fold_scores: List of test log-likelihoods for each fold
    """
    n_sequences = get_num_sequences(data)
    
    fold_scores = []
    
    if n_sequences == 1:
        # Single sequence: use TimeSeriesSplit for temporal splitting
        seq_length = get_sequence_length(data)
        
        if verbose:
            print(f"  Single sequence detected (T={seq_length}), using TimeSeriesSplit")
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(range(seq_length))):
            train_start, train_end = train_idx[0], train_idx[-1] + 1
            test_start, test_end = test_idx[0], test_idx[-1] + 1
            
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}: train=[0:{train_end}], test=[{test_start}:{test_end}]")
            
            train_data = slice_single_sequence(data, 0, train_end)
            test_data = slice_single_sequence(data, test_start, test_end)

            # Fit on training data. _replace keeps every other config field
            # (regularization, priors, init scale, ...) rather than silently
            # reverting the ones not listed here to their defaults.
            fold_config = config._replace(
                seed=config.seed + fold_idx * 1000 if config.seed else seed + fold_idx * 1000
            )

            params, _ = fit_with_restarts(train_data, fold_config, n_fits=n_fits)

            # Evaluate on test data
            test_ll = compute_log_marginal_likelihood(
                params, test_data, config.normalize_gauss_loglik
            )
            fold_scores.append(float(test_ll))
            
            if verbose:
                print(f"    Test LL: {test_ll:.2f}")
    else:
        # Multiple sequences: use KFold to split at sequence level
        if n_sequences < n_folds:
            raise ValueError(f"Not enough sequences ({n_sequences}) for {n_folds}-fold CV")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(range(n_sequences))):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}: train={len(train_idx)}, test={len(test_idx)}")
            
            train_data = subset_data(data, list(train_idx))
            test_data = subset_data(data, list(test_idx))

            # Fit on training data (see note above on _replace)
            fold_config = config._replace(
                seed=config.seed + fold_idx * 1000 if config.seed else seed + fold_idx * 1000
            )

            params, _ = fit_with_restarts(train_data, fold_config, n_fits=n_fits)

            # Evaluate on test data
            test_ll = compute_log_marginal_likelihood(
                params, test_data, config.normalize_gauss_loglik
            )
            fold_scores.append(float(test_ll))
            
            if verbose:
                print(f"    Test LL: {test_ll:.2f}")
    
    mean_ll = np.mean(fold_scores)
    std_ll = np.std(fold_scores)
    
    return mean_ll, std_ll, fold_scores


def _fit_and_compute_evidence(
    data: GLMHMMData,
    config: GLMHMMConfig,
    n_fits: int,
    seed: int,
    evidence_fn: Callable,
    evidence_kwargs: Dict
) -> Tuple[float, float]:
    """
    Fits a model to `data` and returns (log_evidence, var_log_evidence).

    Normalises over the two return conventions in the package: the BQ estimator
    returns (log_Z, var) while the SMC one returns a scalar.
    """
    params, _ = fit_with_restarts(data, config._replace(seed=seed), n_fits=n_fits)
    gammas, xis, _ = e_step(params, data, config.normalize_gauss_loglik)
    out = evidence_fn(data, params, config, gammas, xis, **evidence_kwargs)
    if isinstance(out, tuple):
        return float(out[0]), float(out[1])
    return float(out), 0.0


def posterior_predictive_cv(
    data: GLMHMMData,
    config: GLMHMMConfig,
    n_folds: int = 5,
    n_fits: int = 3,
    seed: int = 42,
    evidence_fn: Optional[Callable] = None,
    evidence_kwargs: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[float, float, List[float]]:
    """
    Cross-validation scored by the *posterior* predictive rather than a plug-in.

    cross_validate_k reports log p(Y_test | Theta_hat), which conditions on a
    point estimate and so ignores parameter uncertainty -- it does not penalise
    extra states. The quantity we actually want is

        log p(Y_test | Y_train)
            = log p(Y_train, Y_test) - log p(Y_train)
            = log Z(Y_train U Y_test) - log Z(Y_train),

    a difference of marginal likelihoods, exact by definition of conditional
    probability. Both terms are already computable, so no posterior over
    parameters has to be represented or sampled.

    Two structural benefits over the plug-in score:

    - log K! cancels exactly between the two terms, so the label-switching
      convention is irrelevant here.
    - In the single-sequence (temporal) case the latent chain is carried across
      the split correctly, because Z(union) is the marginal likelihood of the
      whole prefix. The plug-in path instead restarts the test segment from pi0.
      Both slices used here also start at index 0, which sidesteps the
      transition-input misalignment that affects mid-sequence slices.

    With multiple sequences the union is the entire dataset for every fold, so
    log Z(union) is computed once and reused; fold scores are therefore
    correlated through that shared term and the reported std reflects only the
    spread of the training-set evidences.

    Args:
        data: GLMHMMData with all sequences.
        config: GLMHMMConfig specifying the model (including num_states).
        n_folds: Number of CV folds.
        n_fits: Random restarts per fit.
        seed: Base random seed.
        evidence_fn: Marginal-likelihood estimator with signature
            (data, params, config, gammas, xis, **kwargs). Defaults to the
            factorized Bayesian-quadrature estimator.
        evidence_kwargs: Extra arguments forwarded to evidence_fn.
        verbose: Print per-fold progress.

    Returns:
        mean_score: Mean held-out log predictive density across folds.
        std_score: Standard deviation across folds.
        fold_scores: Per-fold log p(Y_fold | Y_rest).

    NOTE: this inherits every approximation of the underlying evidence
    estimator, above all the soft-count substitution -- see
    docs/model_and_model_selection.qmd. It is a difference of two independently
    approximated quantities, each of which may be large relative to the
    difference, so errors do not cancel systematically. Check that the
    propagated standard deviation is small next to the differences between
    candidate K before trusting a ranking.
    """
    if evidence_fn is None:
        # imported lazily so that tinygp is not required to import this module
        from .model_selection_factorized_bq import compute_factorized_evidence
        evidence_fn = compute_factorized_evidence
    if evidence_kwargs is None:
        evidence_kwargs = {"verbose": False}

    n_sequences = get_num_sequences(data)
    fold_scores: List[float] = []
    fold_vars: List[float] = []

    if n_sequences == 1:
        seq_length = get_sequence_length(data)
        if verbose:
            print(f"  Single sequence (T={seq_length}), TimeSeriesSplit")

        tscv = TimeSeriesSplit(n_splits=n_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(range(seq_length))):
            train_end = train_idx[-1] + 1
            test_end = test_idx[-1] + 1

            train_data = slice_single_sequence(data, 0, train_end)
            union_data = slice_single_sequence(data, 0, test_end)

            lz_union, var_union = _fit_and_compute_evidence(
                union_data, config, n_fits, seed + fold_idx * 1000,
                evidence_fn, evidence_kwargs)
            lz_train, var_train = _fit_and_compute_evidence(
                train_data, config, n_fits, seed + fold_idx * 1000 + 7,
                evidence_fn, evidence_kwargs)

            fold_scores.append(lz_union - lz_train)
            fold_vars.append(var_union + var_train)
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}: train=[0:{train_end}] "
                      f"test=[{train_end}:{test_end}]  "
                      f"log p(Y_test|Y_train) = {fold_scores[-1]:.2f}")
    else:
        if n_sequences < n_folds:
            raise ValueError(f"Not enough sequences ({n_sequences}) for {n_folds}-fold CV")

        # union == the full dataset for every fold, so evaluate it once
        if verbose:
            print("  Computing log Z on the full dataset (shared across folds)...")
        lz_full, var_full = _fit_and_compute_evidence(
            data, config, n_fits, seed, evidence_fn, evidence_kwargs)
        if verbose:
            print(f"  log Z(full) = {lz_full:.2f}")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(range(n_sequences))):
            train_data = subset_data(data, list(train_idx))
            lz_train, var_train = _fit_and_compute_evidence(
                train_data, config, n_fits, seed + fold_idx * 1000,
                evidence_fn, evidence_kwargs)

            fold_scores.append(lz_full - lz_train)
            fold_vars.append(var_full + var_train)
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}: held out {len(test_idx)} sequences  "
                      f"log p(Y_test|Y_train) = {fold_scores[-1]:.2f}")

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    if verbose:
        est_sd = float(np.sqrt(np.mean(fold_vars))) if any(fold_vars) else 0.0
        print(f"  Mean log p(Y_test|Y_train): {mean_score:.2f} +/- {std_score:.2f} "
              f"(estimator sd ~{est_sd:.2f})")

    return mean_score, std_score, fold_scores


def select_k_by_posterior_predictive_cv(
    data: GLMHMMData,
    k_range: List[int],
    base_config: GLMHMMConfig,
    n_folds: int = 5,
    n_fits: int = 3,
    seed: int = 42,
    evidence_fn: Optional[Callable] = None,
    evidence_kwargs: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[int, Dict[int, Tuple[float, float]]]:
    """
    Selects K by the posterior predictive CV score (higher is better).

    Unlike the plug-in score, this penalises extra states automatically: the
    marginal likelihoods it differences already contain the Occam factor.

    Args:
        data: GLMHMMData with all sequences.
        k_range: Candidate state counts.
        base_config: Base config; num_states and seed are overridden per K.
        n_folds: CV folds.
        n_fits: Restarts per fit.
        seed: Base seed.
        evidence_fn: See posterior_predictive_cv.
        evidence_kwargs: See posterior_predictive_cv.
        verbose: Print progress.

    Returns:
        best_k: K with the highest mean posterior predictive score.
        results: Dict K -> (mean, std).
    """
    results = {}

    if verbose:
        print("Posterior-predictive CV  (log Z(union) - log Z(train))")
        print(f"K values: {k_range} | folds: {n_folds} | fits: {n_fits}")
        print("=" * 58)

    for K in k_range:
        if verbose:
            print(f"\n--- K = {K} ---")
        config = base_config._replace(num_states=K, seed=seed + K * 10000)
        mean_s, std_s, _ = posterior_predictive_cv(
            data, config, n_folds=n_folds, n_fits=n_fits, seed=seed,
            evidence_fn=evidence_fn, evidence_kwargs=evidence_kwargs, verbose=verbose
        )
        results[K] = (mean_s, std_s)

    best_k = max(results, key=lambda k: results[k][0])

    if verbose:
        print("\n" + "=" * 58)
        print("POSTERIOR-PREDICTIVE CV RESULTS (higher is better)")
        print("=" * 58)
        for K in k_range:
            m, s = results[K]
            marker = " <-- BEST" if K == best_k else ""
            print(f"K={K}: {m:.2f} +/- {s:.2f}{marker}")
        print(f"\nSelected K: {best_k}")

    return best_k, results


def select_k_by_cv(
    data: GLMHMMData,
    k_range: List[int],
    base_config: GLMHMMConfig,
    n_folds: int = 5,
    n_fits: int = 3,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[int, Dict[int, Tuple[float, float]]]:
    """
    Selects the best number of states K using cross-validation.
    
    For each K in the range, performs K-fold CV and selects the K
    with the highest mean test log-likelihood.
    
    Args:
        data: GLMHMMData object with all sequences
        k_range: List of K values to compare (e.g., [2, 3, 4, 5])
        base_config: Base GLMHMMConfig (num_states will be overridden)
        n_folds: Number of CV folds (default: 5)
        n_fits: Number of random restarts per fold (default: 3)
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        best_k: The K with highest mean test log-likelihood
        cv_results: Dict mapping K -> (mean_ll, std_ll)
        
    Example:
        >>> best_k, results = select_k_by_cv(data, [2, 3, 4], config)
        >>> print(f"Best K: {best_k}")
        >>> for k, (mean, std) in results.items():
        ...     print(f"K={k}: {mean:.2f} +/- {std:.2f}")
    """
    cv_results = {}
    
    if verbose:
        print(f"Cross-Validation Model Selection")
        print(f"K values: {k_range}")
        print(f"Folds: {n_folds}, Fits per fold: {n_fits}")
        print("=" * 50)
    
    for K in k_range:
        if verbose:
            print(f"\n--- K = {K} ---")
        
        # Create config for this K, preserving every other field of base_config
        config = base_config._replace(num_states=K, seed=seed + K * 10000)
        
        mean_ll, std_ll, _ = cross_validate_k(
            data, config, n_folds=n_folds, n_fits=n_fits, 
            seed=seed, verbose=verbose
        )
        
        cv_results[K] = (mean_ll, std_ll)
        
        if verbose:
            print(f"  Mean Test LL: {mean_ll:.2f} ± {std_ll:.2f}")
    
    # Select best K
    best_k = max(cv_results, key=lambda k: cv_results[k][0])
    
    if verbose:
        print("\n" + "=" * 50)
        print("CV RESULTS SUMMARY")
        print("=" * 50)
        for K in k_range:
            mean, std = cv_results[K]
            marker = " <-- BEST" if K == best_k else ""
            print(f"K={K}: {mean:.2f} +/- {std:.2f}{marker}")
        print(f"\nSelected K: {best_k}")
    
    return best_k, cv_results
