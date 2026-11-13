"""
Compares the three model-selection routes -- cross-validation, BIC, and the
factorized Bayesian-quadrature evidence -- on the same data.

Setup: Bernoulli observations with input-driven emissions AND input-driven
transitions, i.e. both the choice GLM and the transition GLM read covariates.
Data is generated from a true K = 3 model in the classic three-state form (left
bias / right bias / stimulus driven); candidates K = 2, 3, 4 are then scored by
each method.

All three are reported on a "higher is better" scale so they can be read
side by side: CV uses held-out log-likelihood, BIC is converted to its implied
log evidence -BIC/2, and BQ reports log Z directly.

Runtime is dominated by cross-validation, which refits the model
len(K_RANGE) * N_FOLDS * N_FITS times; expect on the order of an hour at the
settings below. Lower N_FOLDS, N_FITS, N_SEQUENCES or NUM_EM_ITERS for a
faster, noisier version.
"""

import os
import sys
import itertools
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.fit import fit_with_restarts, e_step
from glmhmm.decode import viterbi_all_sequences
from glmhmm.utils import simulate_data_for_glmhmm
from glmhmm.model_selection import compute_bic, compute_implied_log_z
from glmhmm.model_selection_cv import cross_validate_k
from glmhmm.model_selection_factorized_bq import compute_factorized_evidence

TRUE_K = 3
K_RANGE = [2, 3, 4]
N_SEQUENCES = 10
N_STEPS = 250
N_BERN_FEATURES = 2      # bias + stimulus
N_TRANS_FEATURES = 2     # bias + transition covariate
SEED = 11

N_FOLDS = 5
N_FITS = 2
NUM_EM_ITERS = 100


def make_true_params():
    """
    Three states in the standard behavioural form:
      0: left-biased, weakly stimulus sensitive
      1: right-biased, weakly stimulus sensitive
      2: engaged, strongly stimulus driven with no bias

    Transitions are sticky (large positive diagonal in the bias row) and
    modulated by the transition covariate in the second row.
    """
    W_bern = jnp.array([
        [2.0, -2.0, 0.0],    # bias per state
        [0.2, 0.2, 3.0],     # stimulus weight per state
    ])

    theta = jnp.zeros((N_TRANS_FEATURES, TRUE_K, TRUE_K))
    theta = theta.at[0].set(2.5 * jnp.eye(TRUE_K))          # stickiness
    theta = theta.at[1].set(jnp.array([                      # input modulation
        [0.0, 1.2, -0.8],
        [-0.8, 0.0, 1.2],
        [1.2, -0.8, 0.0],
    ]))

    return {
        "pi0": jnp.ones(TRUE_K) / TRUE_K,
        "theta": theta,
        "W_bern": W_bern,
    }


def state_recovery_accuracy(true_states, decoded_states, n_states):
    """Best accuracy over all relabelings, since state identity is arbitrary."""
    true_cat = np.concatenate([np.asarray(s) for s in true_states])
    dec_cat = np.concatenate([np.asarray(s) for s in decoded_states])
    best = 0.0
    for perm in itertools.permutations(range(n_states)):
        mapped = np.array(perm)[dec_cat]
        best = max(best, float(np.mean(mapped == true_cat)))
    return best


def run_comparison():
    key = jax.random.PRNGKey(SEED)
    true_params = make_true_params()

    print("=" * 74)
    print("Model selection comparison: CV vs BIC vs BQ")
    print("=" * 74)
    print(f"True K = {TRUE_K} | {N_SEQUENCES} sequences x {N_STEPS} trials "
          f"| input-driven emissions and transitions")
    print(f"Candidates: K = {K_RANGE}\n")

    sim = simulate_data_for_glmhmm(
        key, true_params,
        n_sequences=N_SEQUENCES,
        n_hmm_steps=N_STEPS,
        n_bern_features=N_BERN_FEATURES,
        n_trans_features=N_TRANS_FEATURES,
    )

    data = GLMHMMData(
        x_bern=sim["x_set_bern"],
        y_bern=sim["y_set_bern"],
        x_trans=sim["x_trans_set"],
    )
    true_states = sim["true_states"]

    occupancy = np.bincount(
        np.concatenate([np.asarray(s) for s in true_states]), minlength=TRUE_K
    )
    print(f"True state occupancy: {occupancy / occupancy.sum()}")
    print(f"Choice rate: {float(np.mean(np.concatenate([np.asarray(y) for y in sim['y_set_bern']]))):.3f}\n")

    results = {}

    for K in K_RANGE:
        print(f"--- K = {K} " + "-" * 58)
        config = GLMHMMConfig(
            num_states=K,
            num_em_iters=NUM_EM_ITERS,
            tol=1e-4,
            l2_reg_bern=1.0,
            l2_reg_trans=1.0,
            seed=SEED + 100 * K,
        )

        params, lml_history = fit_with_restarts(data, config, n_fits=N_FITS)
        fit_lml = float(lml_history[-1])

        bic = compute_bic(data, params, max_log_lik=fit_lml)
        implied_log_z = compute_implied_log_z(bic)

        gammas, xis, _ = e_step(params, data)
        key, subkey = jax.random.split(key)
        log_z_bq, var_bq = compute_factorized_evidence(
            data, params, config, gammas, xis, key=subkey, verbose=False
        )

        cv_mean, cv_std, _ = cross_validate_k(
            data, config, n_folds=N_FOLDS, n_fits=N_FITS, seed=SEED, verbose=False
        )

        if K == TRUE_K:
            acc = state_recovery_accuracy(true_states, viterbi_all_sequences(params, data), K)
            print(f"  state recovery accuracy (best relabeling): {acc:.3f}")

        results[K] = {
            "lml": fit_lml,
            "bic": float(bic),
            "implied_log_z": float(implied_log_z),
            "bq": float(log_z_bq),
            "bq_sd": float(np.sqrt(var_bq)),
            "cv_mean": float(cv_mean),
            "cv_std": float(cv_std),
        }
        r = results[K]
        print(f"  fit LML     : {r['lml']:12.2f}")
        print(f"  CV test LL  : {r['cv_mean']:12.2f} +/- {r['cv_std']:.2f}")
        print(f"  BIC         : {r['bic']:12.2f}  (implied log Z {r['implied_log_z']:.2f})")
        print(f"  BQ log Z    : {r['bq']:12.2f} +/- {r['bq_sd']:.2f}\n")

    # ---------------------------------------------------------------- summary
    print("=" * 74)
    print("SUMMARY  (all columns higher is better)")
    print("=" * 74)
    print(f"{'K':>3}  {'CV test LL':>22}  {'implied log Z (-BIC/2)':>24}  {'BQ log Z':>14}")
    for K in K_RANGE:
        r = results[K]
        marker = " *" if K == TRUE_K else "  "
        print(f"{K:>3}{marker}{r['cv_mean']:>21.2f}  {r['implied_log_z']:>24.2f}  {r['bq']:>14.2f}")
    print("   (* = true K)")

    selections = {
        "CV": max(K_RANGE, key=lambda k: results[k]["cv_mean"]),
        "BIC": min(K_RANGE, key=lambda k: results[k]["bic"]),
        "BQ": max(K_RANGE, key=lambda k: results[k]["bq"]),
    }

    print("\nSelected K:")
    for name, sel in selections.items():
        verdict = "recovered" if sel == TRUE_K else f"MISSED (true K={TRUE_K})"
        print(f"  {name:<4} -> K = {sel}   {verdict}")

    # margin over the runner-up, to show how decisive each method was
    print("\nMargin over runner-up (nats):")
    for name, key_fn, better in [("CV", lambda k: results[k]["cv_mean"], max),
                                 ("BIC", lambda k: results[k]["implied_log_z"], max),
                                 ("BQ", lambda k: results[k]["bq"], max)]:
        scores = sorted((key_fn(k) for k in K_RANGE), reverse=True)
        print(f"  {name:<4} {scores[0] - scores[1]:8.2f}")

    # ------------------------------------------------------------ assertions
    for K in K_RANGE:
        r = results[K]
        assert np.isfinite(r["cv_mean"]), f"CV score not finite at K={K}"
        assert np.isfinite(r["bic"]), f"BIC not finite at K={K}"
        assert np.isfinite(r["bq"]), f"BQ evidence not finite at K={K}"

    n_recovered = sum(1 for sel in selections.values() if sel == TRUE_K)
    print(f"\n{n_recovered}/3 methods recovered the true K = {TRUE_K}.")

    # A regression guard, not a claim that every method must always be right:
    # model selection on finite data legitimately disagrees. Requiring a
    # majority catches a broken estimator without being flaky about the rest.
    assert n_recovered >= 2, (
        f"only {n_recovered}/3 methods recovered K={TRUE_K}; "
        f"selections were {selections}"
    )
    print("PASS: a majority of methods recovered the true K.")

    return results, selections


if __name__ == "__main__":
    run_comparison()
