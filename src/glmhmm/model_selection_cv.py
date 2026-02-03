"""
Cross-Validation Model Selection for GLM-HMM.

This module provides K-fold cross-validation for model selection,
which is more robust than evidence-based methods when priors are uncertain.

For MULTIPLE sequences: Uses sklearn's KFold to split at the sequence level.
    Each sequence (trial/session) is treated as an independent unit.
    
For SINGLE sequence: Uses sklearn's TimeSeriesSplit to split temporally.
    Train on earlier time steps, test on later ones (respects causality).
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional, Callable, Dict
from sklearn.model_selection import KFold, TimeSeriesSplit

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import fit_glmhmm, fit_with_restarts, compute_log_marginal_likelihood


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
            
            # Fit on training data
            fold_config = GLMHMMConfig(
                num_states=config.num_states,
                num_em_iters=config.num_em_iters,
                l2_reg_trans=config.l2_reg_trans,
                l2_reg_bern=config.l2_reg_bern,
                l2_reg_multinom=config.l2_reg_multinom,
                l2_reg_negbin=config.l2_reg_negbin,
                l2_reg_gamma=config.l2_reg_gamma,
                transition_stickiness=config.transition_stickiness,
                tol=config.tol,
                seed=config.seed + fold_idx * 1000 if config.seed else seed + fold_idx * 1000
            )
            
            params, _ = fit_with_restarts(train_data, fold_config, n_fits=n_fits)
            
            # Evaluate on test data
            test_ll = compute_log_marginal_likelihood(params, test_data)
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
            
            # Fit on training data
            fold_config = GLMHMMConfig(
                num_states=config.num_states,
                num_em_iters=config.num_em_iters,
                l2_reg_trans=config.l2_reg_trans,
                l2_reg_bern=config.l2_reg_bern,
                l2_reg_multinom=config.l2_reg_multinom,
                l2_reg_negbin=config.l2_reg_negbin,
                l2_reg_gamma=config.l2_reg_gamma,
                transition_stickiness=config.transition_stickiness,
                tol=config.tol,
                seed=config.seed + fold_idx * 1000 if config.seed else seed + fold_idx * 1000
            )
            
            params, _ = fit_with_restarts(train_data, fold_config, n_fits=n_fits)
            
            # Evaluate on test data
            test_ll = compute_log_marginal_likelihood(params, test_data)
            fold_scores.append(float(test_ll))
            
            if verbose:
                print(f"    Test LL: {test_ll:.2f}")
    
    mean_ll = np.mean(fold_scores)
    std_ll = np.std(fold_scores)
    
    return mean_ll, std_ll, fold_scores


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
        ...     print(f"K={k}: {mean:.2f} ± {std:.2f}")
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
        
        # Create config for this K
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=base_config.num_em_iters,
            l2_reg_trans=base_config.l2_reg_trans,
            l2_reg_bern=base_config.l2_reg_bern,
            l2_reg_multinom=base_config.l2_reg_multinom,
            l2_reg_negbin=base_config.l2_reg_negbin,
            l2_reg_gamma=base_config.l2_reg_gamma,
            transition_stickiness=base_config.transition_stickiness,
            tol=base_config.tol,
            seed=seed + K * 10000
        )
        
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
            print(f"K={K}: {mean:.2f} ± {std:.2f}{marker}")
        print(f"\nSelected K: {best_k}")
    
    return best_k, cv_results
