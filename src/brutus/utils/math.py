#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mathematical utility functions for brutus.

This module contains mathematical utility functions including matrix operations,
statistical distributions, and numerical utilities. Many functions are JIT-compiled
with numba for performance.

Functions
---------
galactic_to_galactocentric_cyl : Coordinate transform
    Convert Galactic (l, b, distance) to galactocentric cylindrical (R, phi, Z)
inverse3 : Matrix inversion
    Fast 3x3 matrix inversion with optional diagonal preconditioning
isPSD : Matrix check
    Check if matrix is positive semi-definite
chisquare_logpdf : Chi-square log-PDF
    Log-probability density for chi-square distribution
truncnorm_pdf : Truncated normal PDF
    Probability density for truncated normal
truncnorm_logpdf : Truncated normal log-PDF
    Log-probability density for truncated normal

See Also
--------
brutus.utils.sampling : Sampling utilities
brutus.priors.galactic : Uses truncated normal distributions

Notes
-----
The numba-compiled functions provide significant speedups for tight loops
in Bayesian inference. The `inverse3` function is specifically optimized
for the (scale, A_V, R_V) covariance matrices used throughout brutus.

Matrix regularization in `inverse3` prevents numerical issues when matrices
are near-singular by adding a small value to the diagonal when eigenvalues
are too small.

Examples
--------
>>> import numpy as np
>>> from brutus.utils.math import inverse3, isPSD
>>>
>>> # Create a 3x3 covariance matrix
>>> cov = np.array([[1.0, 0.1, 0.05],
...                 [0.1, 0.5, 0.02],
...                 [0.05, 0.02, 0.3]])
>>>
>>> # Check if positive semi-definite
>>> is_valid = isPSD(cov)
>>>
>>> # Invert with diagonal preconditioning / regularization
>>> icov = inverse3(cov, regularize=True)
"""

from math import lgamma, log

import numpy as np
from numba import jit, prange
from scipy.special import log_ndtr

# Proxy for log(0) in log-probabilities. It stays finite under arithmetic,
# where -inf would give NaN (for example -inf - -inf).
LOG_ZERO = -1e300

__all__ = [
    "galactic_to_galactocentric_cyl",
    "inverse3",
    "isPSD",
    "chisquare_logpdf",
    "truncnorm_pdf",
    "truncnorm_logpdf",
]


@jit(nopython=True, cache=True)
def _matrix_det_3x3(A):
    """Compute 3x3 matrix determinant - numba compatible."""
    return (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


@jit(nopython=True, cache=True)
def _invert_3x3_analytical(A):
    """
    Invert a 3x3 matrix using analytical formulas.

    This is numba-compatible and numerically stable for well-conditioned matrices.
    Uses the standard analytical inversion formula with explicit determinant calculation.
    """
    # Extract matrix elements
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]

    # Calculate determinant
    det = _matrix_det_3x3(A)

    # Singularity test must be relative to the matrix scale: det scales as
    # (matrix scale)^3, so an absolute cutoff misclassifies well-conditioned
    # small-magnitude matrices (e.g. diag(1e-6)) as singular. Compare against
    # the product of row inf-norms (~Hadamard bound), which tracks the
    # magnitude det would have for a well-conditioned matrix of this scale.
    r0 = max(abs(a11), abs(a12), abs(a13))
    r1 = max(abs(a21), abs(a22), abs(a23))
    r2 = max(abs(a31), abs(a32), abs(a33))
    det_scale = r0 * r1 * r2
    if abs(det) < 1e-15 * det_scale or det_scale == 0.0:
        # Return matrix filled with inf for the (numerically) singular case
        inv = np.empty_like(A)
        inv.fill(np.inf)
        return inv

    # Calculate inverse using cofactor method
    inv = np.empty_like(A)

    inv[0, 0] = (a22 * a33 - a23 * a32) / det
    inv[0, 1] = (a13 * a32 - a12 * a33) / det
    inv[0, 2] = (a12 * a23 - a13 * a22) / det

    inv[1, 0] = (a23 * a31 - a21 * a33) / det
    inv[1, 1] = (a11 * a33 - a13 * a31) / det
    inv[1, 2] = (a13 * a21 - a11 * a23) / det

    inv[2, 0] = (a21 * a32 - a22 * a31) / det
    inv[2, 1] = (a12 * a31 - a11 * a32) / det
    inv[2, 2] = (a11 * a22 - a12 * a21) / det

    return inv


@jit(nopython=True, cache=True, parallel=True)
def _batch_min_eig_of_inverse_sym3(P_batch):
    """
    Smallest eigenvalue of the inverse of each symmetric 3x3 matrix.

    The eigenvalues ``mu`` of ``P`` come from the closed-form trigonometric
    solution (Smith 1961); those of ``inv(P)`` are ``1 / mu``. Working on a
    unit-diagonal ``P`` keeps every ``mu`` accurate to rounding, while the
    same formula applied to ``inv(P)`` cancels when its eigenvalues span
    many orders of magnitude.
    """
    N = P_batch.shape[0]
    out = np.empty(N)
    for n in prange(N):
        a11 = P_batch[n, 0, 0]
        a22 = P_batch[n, 1, 1]
        a33 = P_batch[n, 2, 2]
        a12 = P_batch[n, 0, 1]
        a13 = P_batch[n, 0, 2]
        a23 = P_batch[n, 1, 2]
        p1 = a12 * a12 + a13 * a13 + a23 * a23
        if p1 == 0.0:
            mu0 = min(a11, a22, a33)
            mu2 = max(a11, a22, a33)
            mu1 = a11 + a22 + a33 - mu0 - mu2
        else:
            q = (a11 + a22 + a33) / 3.0
            d11 = a11 - q
            d22 = a22 - q
            d33 = a33 - q
            p = np.sqrt((d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1) / 6.0)
            # B = (P - qI)/p ; r = det(B)/2
            b11 = d11 / p
            b22 = d22 / p
            b33 = d33 / p
            b12 = a12 / p
            b13 = a13 / p
            b23 = a23 / p
            r = 0.5 * (
                b11 * (b22 * b33 - b23 * b23)
                - b12 * (b12 * b33 - b23 * b13)
                + b13 * (b12 * b23 - b22 * b13)
            )
            if r <= -1.0:
                phi = np.pi / 3.0
            elif r >= 1.0:
                phi = 0.0
            else:
                phi = np.arccos(r) / 3.0
            mu2 = q + 2.0 * p * np.cos(phi)
            mu0 = q + 2.0 * p * np.cos(phi + (2.0 * np.pi / 3.0))
            mu1 = 3.0 * q - mu0 - mu2
        if mu0 > 0.0:
            out[n] = 1.0 / mu2
        else:
            # The most negative 1/mu comes from the negative mu nearest 0.
            m = mu2 if mu2 < 0.0 else (mu1 if mu1 < 0.0 else mu0)
            out[n] = 1.0 / m if m != 0.0 else -np.inf
    return out


@jit(nopython=True, cache=True)
def _batch_invert_3x3(A_batch):
    """
    Numba-compiled batch 3x3 matrix inversion using analytical method.

    Parameters
    ----------
    A_batch : ndarray of shape (N, 3, 3)
        Batch of 3x3 matrices to invert.

    Returns
    -------
    inv_batch : ndarray of shape (N, 3, 3)
        Batch of inverted matrices.
    """
    N = A_batch.shape[0]
    result = np.empty_like(A_batch)

    for i in range(N):
        result[i] = _invert_3x3_analytical(A_batch[i])

    return result


def _invert_3x3_preconditioned(P, min_eigenval_threshold=1e-12):
    """
    Invert a single 3x3 symmetric positive-definite matrix using diagonal
    preconditioning for numerical stability.

    Normalizes the precision matrix to a correlation-like matrix (1s on
    diagonal) before inversion, then transforms back. This reduces the
    condition number from O(max_diag/min_diag) to O(1/(1-rho_max)),
    which depends only on correlations, not parameter scales.

    Parameters
    ----------
    P : ndarray of shape (3, 3)
        Symmetric positive-definite precision matrix.
    min_eigenval_threshold : float
        Minimum eigenvalue threshold for the normalized covariance.

    Returns
    -------
    C : ndarray of shape (3, 3)
        The inverse (covariance) matrix.
    """
    # Step 1: Diagonal scaling factors
    d0 = np.sqrt(max(P[0, 0], 1e-30))
    d1 = np.sqrt(max(P[1, 1], 1e-30))
    d2 = np.sqrt(max(P[2, 2], 1e-30))
    d_inv = np.array([1.0 / d0, 1.0 / d1, 1.0 / d2])

    # Step 2: Symmetrize and normalize to correlation-like matrix (1s on diagonal)
    P_norm = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            P_norm[i, j] = 0.5 * (P[i, j] + P[j, i]) * d_inv[i] * d_inv[j]

    # Step 3: Pre-regularize if the normalized matrix is singular.
    # In normalized space, a small additive shift is well-scaled since
    # all diagonals are 1.0.
    det_norm = _matrix_det_3x3(P_norm)
    if abs(det_norm) < 1e-10:
        P_norm[0, 0] += 1e-3
        P_norm[1, 1] += 1e-3
        P_norm[2, 2] += 1e-3

    # Invert the normalized matrix analytically
    C_norm = _invert_3x3_analytical(P_norm)

    # Step 4: Handle inversion failure (inf/nan from truly degenerate input)
    if not np.all(np.isfinite(C_norm)):
        # Fallback: return diagonal covariance from precision diagonal
        C = np.zeros((3, 3))
        C[0, 0] = 1.0 / max(P[0, 0], 1e-30)
        C[1, 1] = 1.0 / max(P[1, 1], 1e-30)
        C[2, 2] = 1.0 / max(P[2, 2], 1e-30)
        return C

    # Step 5: Symmetrize and regularize in normalized space.
    # Use exact eigenvalues (not Gershgorin) because highly correlated
    # parameters (ρ > 0.9, common for distance-extinction degeneracy)
    # cause Gershgorin to give wildly pessimistic bounds that trigger
    # massive false regularization.
    C_sym = 0.5 * (C_norm + C_norm.T)
    min_eig = np.min(np.linalg.eigvalsh(C_sym))
    if min_eig < min_eigenval_threshold:
        shift = min_eigenval_threshold - min_eig
        C_sym[0, 0] += shift
        C_sym[1, 1] += shift
        C_sym[2, 2] += shift
        C_norm = C_sym

    # Step 6: Transform back to original parameter space
    # C_original = D^{-1} C_norm D^{-1}
    C = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = C_norm[i, j] * d_inv[i] * d_inv[j]

    return C


def _batch_invert_3x3_preconditioned(P_batch, min_eigenval_threshold=1e-12):
    """
    Batch-invert N 3x3 matrices with diagonal preconditioning.

    Fully vectorized: operates on the entire (N, 3, 3) array at once
    using numpy broadcasting. Avoids Python loops over N matrices.
    """

    # Step 1: Diagonal scaling (vectorized)
    diag = np.sqrt(np.maximum(np.diagonal(P_batch, axis1=1, axis2=2), 1e-30))  # (N, 3)
    d_inv = 1.0 / diag  # (N, 3)

    # Step 2: Normalize to correlation form (vectorized)
    P_sym = 0.5 * (P_batch + np.swapaxes(P_batch, 1, 2))  # symmetrize
    P_norm = P_sym * d_inv[:, :, None] * d_inv[:, None, :]  # (N, 3, 3)

    # Step 3: Pre-regularize singular matrices
    dets = (
        P_norm[:, 0, 0]
        * (P_norm[:, 1, 1] * P_norm[:, 2, 2] - P_norm[:, 1, 2] * P_norm[:, 2, 1])
        - P_norm[:, 0, 1]
        * (P_norm[:, 1, 0] * P_norm[:, 2, 2] - P_norm[:, 1, 2] * P_norm[:, 2, 0])
        + P_norm[:, 0, 2]
        * (P_norm[:, 1, 0] * P_norm[:, 2, 1] - P_norm[:, 1, 1] * P_norm[:, 2, 0])
    )  # (N,)
    singular = np.abs(dets) < 1e-10
    if np.any(singular):
        P_norm[singular, 0, 0] += 1e-3
        P_norm[singular, 1, 1] += 1e-3
        P_norm[singular, 2, 2] += 1e-3

    # Step 4: Batch analytical inversion
    C_norm = _batch_invert_3x3(P_norm)

    # Step 5: Handle inversion failures
    bad = ~np.all(np.isfinite(C_norm), axis=(1, 2))
    if np.any(bad):
        C_norm[bad] = 0.0
        for i in range(3):
            C_norm[bad, i, i] = 1.0 / np.maximum(P_batch[bad, i, i], 1e-30)
        # For bad matrices, d_inv should be identity-like
        d_inv[bad] = 1.0

    # Step 6: Symmetrize
    C_sym = 0.5 * (C_norm + np.swapaxes(C_norm, 1, 2))

    # Step 7: Batch eigenvalue check and regularization. Only the smallest
    # eigenvalue of C_norm is needed; it comes from the eigenvalues of
    # P_norm, which is about 30 times faster than a batched eigvalsh.
    min_eigs = _batch_min_eig_of_inverse_sym3(P_norm)  # (N,)
    needs_reg = (min_eigs < min_eigenval_threshold) & (~bad)
    if np.any(needs_reg):
        shifts = min_eigenval_threshold - min_eigs[needs_reg]
        C_sym[needs_reg, 0, 0] += shifts
        C_sym[needs_reg, 1, 1] += shifts
        C_sym[needs_reg, 2, 2] += shifts

    # Step 8: Transform back to original parameter space
    C = C_sym * d_inv[:, :, None] * d_inv[:, None, :]  # (N, 3, 3)

    return C


def inverse3(A, regularize=False, min_eigenval_threshold=1e-12):
    """
    Compute the inverse of a series of 3x3 matrices.

    When ``regularize=True``, uses diagonal preconditioning: the precision
    matrix is normalized to a correlation-like matrix (1s on diagonal)
    before inversion, then transformed back. This reduces condition
    numbers from O(max_diag/min_diag) to O(1/(1-rho_max)), making the
    inversion numerically stable even when parameters have very different
    scales (e.g. scale ~ 10^5 vs R(V) precision ~ 30).

    Parameters
    ----------
    A : array_like of shape `(3, 3)` or `(N, 3, 3)`
        A 3x3 matrix or a stack of them.
    regularize : bool, optional
        Whether to apply diagonal preconditioning and regularization
        to ensure positive semi-definiteness. Default: False.
    min_eigenval_threshold : float, optional
        Minimum acceptable eigenvalue for OUTPUT matrices in the
        normalized space. Default: 1e-12.

    Returns
    -------
    A_inv : `~numpy.ndarray` of the shape of `A`
        Inverse matrices, guaranteed to be positive semi-definite if
        regularize=True.

    Raises
    ------
    ValueError
        If `A` does not have shape `(3, 3)` or `(N, 3, 3)`.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim not in (2, 3) or A.shape[-2:] != (3, 3):
        raise ValueError(f"`A` must have shape (3, 3) or (N, 3, 3); got {A.shape}.")
    if not regularize:
        if A.ndim == 2:
            return _invert_3x3_analytical(A)
        else:
            return _batch_invert_3x3(A)

    if A.ndim == 2:
        return _invert_3x3_preconditioned(A, min_eigenval_threshold)
    else:
        return _batch_invert_3x3_preconditioned(A, min_eigenval_threshold)


def isPSD(A):
    """
    Check if `A` is a positive semidefinite matrix.

    A matrix is positive semidefinite if all its eigenvalues are non-negative.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(N, N)`
        Square matrix to test.

    Returns
    -------
    is_psd : bool
        True if the matrix is positive semidefinite, False otherwise.
    """
    # Check if matrix is symmetric (within numerical precision)
    if not np.allclose(A, A.T, rtol=1e-10, atol=1e-10):
        return False

    # Check eigenvalues are non-negative (eigvalsh is faster for symmetric matrices)
    eigenvals = np.linalg.eigvalsh(A)
    return np.all(eigenvals >= -1e-10)  # Allow small numerical errors


def chisquare_logpdf(x, df, loc=0, scale=1):
    """
    Compute log-PDF of a chi-square distribution.

    `chisquare_logpdf(x, df, loc, scale)` is equal to
    `chisquare_logpdf(y, df) - ln(scale)`, where `y = (x-loc)/scale`.
    NOTE: This function replicates `~scipy.stats.chi2.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    df : float
        Degrees of freedom. Uses ``math.lgamma`` internally, so there is no
        overflow risk for large ``df`` (unlike ``math.gamma`` which overflows
        for ``df`` > ~340).

    loc : float, optional
        Offset of distribution. Default is 0.

    scale : float, optional
        Scaling of distribution. Default is 1.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The natural log of the PDF values.
    """
    if isinstance(x, list):
        x = np.asarray(x)

    y = (x - loc) / scale
    is_scalar = isinstance(y, (float, int))

    if is_scalar:
        if y <= 0:
            return -np.inf
    else:
        keys = y <= 0
        y = np.where(keys, 0.1, y)  # temporary value, will be set to -inf below

    # Compute log-pdf
    ans = -(df / 2.0) * log(2) - lgamma(df / 2.0)
    ans = ans + (df / 2.0 - 1.0) * np.log(y) - y / 2.0 - log(scale)

    if not is_scalar:
        ans = np.where(keys, -np.inf, ans)

    return ans


def _truncnorm_lognorm(a, b):
    """
    Compute ``log(Phi(b) - Phi(a))`` for standardized bounds ``a < b``.

    Evaluated tail-stably with `scipy.special.log_ndtr`: when both bounds sit
    in the upper tail the mass is mirrored to the lower tail (Phi(b) - Phi(a)
    = Phi(-a) - Phi(-b)), so the result keeps full relative precision even
    when a direct CDF difference would cancel catastrophically (e.g. bounds
    beyond ~8 sigma, where erf differences underflow to exactly 0).

    Parameters
    ----------
    a, b : float or `~numpy.ndarray`
        Standardized truncation bounds (``a`` lower, ``b`` upper).

    Returns
    -------
    lognorm : float or `~numpy.ndarray`
        Natural log of the enclosed Gaussian mass. ``-inf`` when ``a == b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Where both bounds are in the upper tail, mirror to the lower tail
    # (Phi(b) - Phi(a) = Phi(-a) - Phi(-b)) so log_ndtr keeps precision.
    flip = a > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    # log(exp(hi_mass) - exp(lo_mass)) with hi_mass >= lo_mass.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mass_lo = log_ndtr(lo)
        log_mass_hi = log_ndtr(hi)
        out = log_mass_hi + np.log1p(-np.exp(log_mass_lo - log_mass_hi))
    if out.ndim == 0:
        return float(out)
    return out


def truncnorm_pdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.pdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower bound in standardized units. The actual lower cutoff
        in data space is ``scale * a + loc``.

    b : float
        Upper bound in standardized units. The actual upper cutoff
        in data space is ``scale * b + loc``.

    loc : float, optional
        Mean of normal distribution. Default is 0.0.

    scale : float, optional
        Standard deviation of normal distribution. Default is 1.0.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The PDF values: 0 outside the bounds, and a scalar for scalar `x`.
    """
    return np.exp(truncnorm_logpdf(x, a, b, loc=loc, scale=scale))


def truncnorm_logpdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute log-PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower bound in standardized units. The actual lower cutoff
        in data space is ``scale * a + loc``.

    b : float
        Upper bound in standardized units. The actual upper cutoff
        in data space is ``scale * b + loc``.

    loc : float, optional
        Mean of normal distribution. Default is 0.0.

    scale : float, optional
        Standard deviation of normal distribution. Default is 1.0.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The natural log PDF values: `LOG_ZERO` (-1e300) outside the
        bounds, and a scalar for scalar `x`.
    """
    x = np.asarray(x, dtype=float)
    _a = scale * a + loc
    _b = scale * b + loc
    xi = (x - loc) / scale

    lnphi = -np.log(np.sqrt(2 * np.pi)) - 0.5 * np.square(xi)
    # The log-mass is tail-stable: an erf difference underflows to 0 for
    # same-side bounds beyond about 8 sigma.
    lndenom = np.log(scale) + _truncnorm_lognorm(a, b)
    return np.where((x < _a) | (x > _b), LOG_ZERO, lnphi - lndenom)[()]


def galactic_to_galactocentric_cyl(dists, ell, b, R_solar=8.2, Z_solar=0.025):
    """
    Convert Galactic coordinates to Galactocentric cylindrical coordinates.

    Converts heliocentric Galactic coordinates (l, b, distance) to
    Galactocentric cylindrical coordinates (R, Z) using a simple rotation
    and translation. This is a fast NumPy-based replacement for
    astropy SkyCoord coordinate transformations.

    Parameters
    ----------
    dists : array_like
        Heliocentric distances in kpc.
    ell : float or array_like
        Galactic longitude in degrees.
    b : float or array_like
        Galactic latitude in degrees.
    R_solar : float, optional
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above the Galactic midplane in kpc. Default is 0.025.

    Returns
    -------
    R : ndarray
        Galactocentric cylindrical radius in kpc.
    Z : ndarray
        Height above/below the Galactic midplane in kpc.

    Notes
    -----
    The conversion assumes:

    - The Sun is located at ``(x, y, z) = (R_solar, 0, Z_solar)`` in
      Galactocentric Cartesian coordinates.
    - Galactic longitude ``l=0`` points toward the Galactic center.
    - The Galactic midplane is at ``Z=0``.

    The Cartesian positions relative to the Sun are:

    .. math::

        x = d \\cos(b) \\cos(l)

        y = d \\cos(b) \\sin(l)

    And the Galactocentric cylindrical coordinates are:

    .. math::

        R = \\sqrt{(x - R_\\odot)^2 + y^2}

        Z = d \\sin(b) + Z_\\odot

    See Also
    --------
    brutus.priors.galactic.logp_galactic_structure : Uses this for coordinate
        conversion instead of astropy SkyCoord.

    Examples
    --------
    >>> import numpy as np
    >>> from brutus.utils.math import galactic_to_galactocentric_cyl
    >>> R, Z = galactic_to_galactocentric_cyl(
    ...     dists=np.array([1.0, 5.0]), ell=90.0, b=0.0
    ... )
    """
    ell_rad = np.deg2rad(ell)
    b_rad = np.deg2rad(b)
    cos_b = np.cos(b_rad)
    x = dists * cos_b * np.cos(ell_rad)
    y = dists * cos_b * np.sin(ell_rad)
    R = np.sqrt((x - R_solar) ** 2 + y**2)
    Z = dists * np.sin(b_rad) + Z_solar

    return R, Z
