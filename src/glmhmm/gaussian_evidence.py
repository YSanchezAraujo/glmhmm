"""
Closed-form marginal likelihood for the Gaussian emission factor.

The Gaussian factor of the factorized evidence is a responsibility-weighted
multivariate linear regression. Under a conjugate matrix-normal / inverse-Wishart
prior its marginal likelihood is available analytically, so this factor needs
neither Bayesian quadrature nor SMC: it is exact, cheaper, and unlike the
sampling-based path it integrates over the noise covariance instead of plugging
in a point estimate.

Model, for one state k:

    y_s | W, Sigma  ~  N(W^T x_s, Sigma)        weighted by w_s
    W | Sigma       ~  MN(0, Lambda^-1, Sigma)  row precision Lambda, column cov Sigma
    Sigma           ~  IW(Psi, nu)

The matrix-normal prior (column covariance Sigma) is not an arbitrary choice: it
is exactly the prior implied by the closed-form Gaussian M-step, whose normal
equations (X^T Gamma X + Lambda) W = X^T Gamma Y are the stationarity condition
of this likelihood-plus-prior for any Sigma. An isotropic prior on vec(W) would
not be, and would leave the fitted W off the mode of the density being
integrated.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from typing import Optional


def multigammaln(a: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Log of the multivariate gamma function Gamma_dim(a).

    Args:
        a: Argument (scalar).
        dim: Dimension.

    Returns:
        log Gamma_dim(a).
    """
    j = jnp.arange(1, dim + 1, dtype=jnp.result_type(float))
    return 0.25 * dim * (dim - 1) * jnp.log(jnp.pi) + jnp.sum(gammaln(a + 0.5 * (1.0 - j)))


def gaussian_sample_weights(
    gammas_flat: jnp.ndarray,
    state_idx: int,
    n_channels: int,
    segment_ids: Optional[jnp.ndarray] = None,
    segment_lengths: Optional[jnp.ndarray] = None,
    normalize: bool = False
) -> jnp.ndarray:
    """
    Per-sample weights that this state contributes to the Gaussian likelihood.

    These reproduce exactly what the forward pass uses. With segmented (variable
    rate) data the step-level responsibility is spread over the samples of its
    segment, and if the pseudo-likelihood rescaling is active each sample also
    carries the 1 / (L_t * M) factor -- so the rescaling stays a per-sample
    weight and the conjugate algebra still applies.

    Args:
        gammas_flat: Responsibilities, (T_total, K) for segmented data or
                     (N_total, K) when samples map one-to-one to steps.
        state_idx: Which state.
        n_channels: Number of Gaussian output channels M.
        segment_ids: Sample -> step map, (N_total,). None for one-to-one data.
        segment_lengths: Segment lengths L_t, (T_total,). Required if normalize.
        normalize: Whether the 1 / (L_t * M) rescaling is active.

    Returns:
        Weights, shape (N_total,).
    """
    if segment_ids is None:
        return gammas_flat[:, state_idx]

    w = gammas_flat[segment_ids, state_idx]
    if normalize:
        w = w / (segment_lengths[segment_ids] * n_channels)
    return w


def gaussian_factor_log_evidence(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    weights: jnp.ndarray,
    row_precisions: jnp.ndarray,
    iw_scale: jnp.ndarray,
    iw_dof: float
) -> jnp.ndarray:
    """
    Exact log marginal likelihood of a weighted multivariate linear regression
    under a matrix-normal / inverse-Wishart prior, integrating out both W and
    Sigma.

    With S_xx = X^T diag(w) X + Lambda, S_xy = X^T diag(w) Y,
    S_yy = Y^T diag(w) Y, n = sum(w) and Psi_N = Psi + S_yy - S_xy^T S_xx^-1 S_xy:

        log Z = -(n M / 2) log(pi)
                + (M / 2) (log|Lambda| - log|S_xx|)
                + (nu / 2) log|Psi| - ((nu + n) / 2) log|Psi_N|
                + log Gamma_M((nu + n) / 2) - log Gamma_M(nu / 2)

    Note the base is pi rather than 2 pi: the factors of 2 are absorbed by the
    inverse-Wishart normaliser.

    Args:
        X: Design matrix, (N, D). Column 0 is the bias.
        Y: Observations, (N, M).
        weights: Per-sample responsibility weights, (N,). May be fractional.
        row_precisions: Prior precision per row of W, (D,). Row 0 is the bias.
        iw_scale: Inverse-Wishart scale matrix Psi, (M, M).
        iw_dof: Inverse-Wishart degrees of freedom nu. Must exceed M - 1.

    Returns:
        Scalar log evidence for this factor.
    """
    M = Y.shape[1]

    Xw = X * weights[:, jnp.newaxis]
    S_xx = X.T @ Xw + jnp.diag(row_precisions)
    S_xy = Xw.T @ Y
    S_yy = Y.T @ (Y * weights[:, jnp.newaxis])
    n_eff = jnp.sum(weights)

    L_xx = jnp.linalg.cholesky(S_xx)
    sol = jax.scipy.linalg.solve_triangular(L_xx, S_xy, lower=True)   # (D, M)
    Psi_N = iw_scale + S_yy - sol.T @ sol
    dof_N = iw_dof + n_eff

    logdet_Lambda = jnp.sum(jnp.log(row_precisions))
    logdet_S_xx = 2.0 * jnp.sum(jnp.log(jnp.diag(L_xx)))
    logdet_Psi = jnp.linalg.slogdet(iw_scale)[1]
    logdet_Psi_N = jnp.linalg.slogdet(Psi_N)[1]

    return (
        -0.5 * n_eff * M * jnp.log(jnp.pi)
        + 0.5 * M * (logdet_Lambda - logdet_S_xx)
        + 0.5 * iw_dof * logdet_Psi
        - 0.5 * dof_N * logdet_Psi_N
        + multigammaln(0.5 * dof_N, M)
        - multigammaln(0.5 * iw_dof, M)
    )


def resolve_iw_prior(config, n_channels: int):
    """
    Resolves the inverse-Wishart hyperparameters from a GLMHMMConfig.

    Defaults to nu = M + 2, the smallest integer degrees of freedom for which the
    prior mean E[Sigma] = Psi / (nu - M - 1) exists, and Psi = iw_scale_gauss * I.

    NOTE: Psi carries the units of y_gauss squared. The default of 1.0 assumes
    roughly standardised Gaussian data; set config.iw_scale_gauss to about the
    expected residual variance otherwise.

    Args:
        config: GLMHMMConfig.
        n_channels: Number of Gaussian output channels M.

    Returns:
        (iw_scale, iw_dof) with iw_scale of shape (M, M).
    """
    dof = config.iw_dof_gauss
    if dof is None:
        dof = float(n_channels) + 2.0
    if dof <= n_channels - 1:
        raise ValueError(
            f"iw_dof_gauss must exceed M - 1 = {n_channels - 1} for a proper "
            f"inverse-Wishart prior, got {dof}."
        )
    scale = config.iw_scale_gauss * jnp.eye(n_channels)
    return scale, float(dof)
