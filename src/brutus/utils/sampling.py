#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling utility functions for brutus.

This module contains functions for statistical sampling, quantile computation,
and random number generation used in Bayesian inference workflows. These
utilities are essential for posterior sampling and uncertainty quantification.

Functions
---------
quantile : Weighted quantiles
    Compute (weighted) quantiles from samples
draw_sar : Posterior sampling
    Draw from (scale, A_V, R_V) posterior
sample_multivariate_normal : Gaussian sampling
    Sample from multivariate normal with bounds

See Also
--------
brutus.analysis.individual.BruteForce : Uses these for posterior sampling
brutus.utils.math : Mathematical utilities

Notes
-----
The `quantile` function supports weighted samples, which is crucial for
computing credible intervals from posterior samples with non-uniform weights
(e.g., from importance sampling or nested sampling).

The `draw_sar` function is specifically designed for sampling the joint
posterior of distance scale, extinction, and reddening curve shape from
BruteForce fitting results.

Examples
--------
Weighted quantile computation:

>>> import numpy as np
>>> from brutus.utils.sampling import quantile
>>>
>>> # Samples with different weights
>>> samples = np.array([1, 2, 3, 4, 5])
>>> weights = np.array([1, 1, 1, 1, 10])  # Last sample heavily weighted
>>>
>>> # Compute median and 68% credible interval
>>> q = np.array([0.16, 0.5, 0.84])
>>> intervals = quantile(samples, q, weights=weights)
>>> print(f"Median: {intervals[1]:.2f}")

Drawing from posterior:

>>> from brutus.utils.sampling import draw_sar
>>>
>>> # Posterior means and covariances from fitting
>>> # scales, avs, rvs, covs_sar = ... (from BruteForce)
>>>
>>> # Generate posterior samples
>>> # samples = draw_sar(scales, avs, rvs, covs_sar, ndraws=1000)
"""

import warnings

import numpy as np
from numba import jit, prange

__all__ = ["quantile", "draw_sar", "sample_multivariate_normal"]


def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    This function computes quantiles from a set of samples, optionally
    with weights, using the midpoint-CDF convention: the empirical CDF is
    evaluated at ``(cumsum(w) - 0.5 * w) / sum(w)`` and linearly
    interpolated. Omitting `weights` is exactly equivalent to passing
    uniform weights, so the reported quantiles do not depend on whether a
    trivially-uniform weights array happens to be supplied.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape `(nsamps,)`
        Input samples.

    q : `~numpy.ndarray` with shape `(nquantiles,)` or float
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape `(nsamps,)`, optional
        The associated weight from each sample. If None, all samples
        are weighted equally.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape `(nquantiles,)` or float
        The (weighted) sample quantiles computed at `q`.

    Raises
    ------
    ValueError
        If quantiles are outside [0, 1] or if dimensions don't match.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> q = np.array([0.25, 0.5, 0.75])
    >>> quantile(x, q)
    array([1.75, 3.  , 4.25])

    >>> weights = np.array([1, 1, 1, 1, 10])  # Last sample heavily weighted
    >>> quantile(x, q, weights=weights)
    array([4.        , 4.63636364, 5.        ])
    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    # Unweighted samples use unit weights so that both paths share the same
    # midpoint-CDF quantile convention (weights=None must agree with
    # explicitly-uniform weights).
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")

    idx = np.argsort(x)  # sort samples
    sw = weights[idx]  # sort weights
    # Compute CDF at sample midpoints for proper quantile calculation
    cdf = (np.cumsum(sw, dtype=float) - 0.5 * sw) / np.sum(sw)
    return np.interp(q, cdf, x[idx])


def draw_sar(
    scales,
    avs,
    rvs,
    covs_sar,
    ndraws=500,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    rstate=None,
    max_attempts=10000,
):
    """
    Generate random draws from the joint scale-A_V-R_V posterior for a
    given object.

    This function generates Monte Carlo samples from the joint posterior
    of scale factors, reddening (A_V), and reddening curve shape (R_V)
    for stellar fitting applications.

    Parameters
    ----------
    scales : `~numpy.ndarray` of shape `(Nsamps)`
        An array of scale factors `s` derived between the models and the data.

    avs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of reddenings `A(V)` derived for the models.

    rvs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of reddening shapes `R(V)` derived for the models.

    covs_sar : `~numpy.ndarray` of shape `(Nsamps, 3, 3)`
        An array of covariance matrices corresponding to `(scales, avs, rvs)`.

    ndraws : int, optional
        The number of desired random draws. Default is `500`.

    avlim : 2-tuple, optional
        The A_V limits used to truncate results. Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The R_V limits used to truncate results. Default is `(1., 8.)`.

    rstate : `~numpy.random.RandomState`, `~numpy.random.Generator`, or
        module, optional
        Random state used for the draws (anything exposing a
        ``normal(loc, scale, size)`` method, including the `numpy.random`
        module itself). If None, uses the default numpy random state.

    max_attempts : int, optional
        Maximum number of rejection-sampling passes per posterior sample
        before the remaining slots are padded with the mean values (with a
        warning). Default is `10000`.

    Returns
    -------
    sdraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Scale-factor samples.

    adraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Reddening (A_V) samples.

    rdraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Reddening shape (R_V) samples.

    Notes
    -----
    The function samples from multivariate normal distributions defined by
    the means (scales, avs, rvs) and covariances (covs_sar), then applies
    rejection sampling to ensure all samples fall within the specified
    limits for A_V and R_V.

    All `Nsamps` distributions are drawn in a single batched
    `sample_multivariate_normal` call per rejection pass (rather than one
    `numpy.random.multivariate_normal` call per distribution), which avoids
    per-call dispatch and per-matrix SVD overhead. The accepted draws follow
    the same truncated Gaussian distribution as before, but the underlying
    RNG stream differs, so individual draws are not reproducible against
    older versions even with a fixed seed.

    Examples
    --------
    >>> import numpy as np
    >>> scales = np.array([1.0, 1.1])
    >>> avs = np.array([0.1, 0.2])
    >>> rvs = np.array([3.1, 3.3])
    >>> covs_sar = np.array([[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
    ...                      [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]]])
    >>> sdraws, adraws, rdraws = draw_sar(scales, avs, rvs, covs_sar, ndraws=100)
    >>> sdraws.shape
    (2, 100)
    """
    if rstate is None:
        rstate = np.random

    scales = np.asarray(scales, dtype=float)
    avs = np.asarray(avs, dtype=float)
    rvs = np.asarray(rvs, dtype=float)
    covs_sar = np.asarray(covs_sar, dtype=float)

    # Generate realizations for each (scale, av, rv, cov_sar) set.
    nsamps = len(scales)
    sdraws, adraws, rdraws = np.zeros((3, nsamps, ndraws))
    means = np.column_stack((scales, avs, rvs))

    # Rejection-sample all still-deficient distributions together: each pass
    # draws `ndraws` candidates per active distribution in ONE batched
    # Cholesky-based call (instead of one numpy `multivariate_normal` per
    # distribution) and fills accepted draws in order.
    nfilled = np.zeros(nsamps, dtype=np.int64)
    active = np.arange(nsamps)
    n_attempts = 0
    while active.size > 0 and n_attempts < max_attempts:
        n_attempts += 1
        # Draw samples; shape (3, ndraws, Nactive).
        draws = sample_multivariate_normal(
            means[active], covs_sar[active], size=ndraws, rstate=rstate
        )
        s_mc, a_mc, r_mc = draws[0].T, draws[1].T, draws[2].T  # (Nactive, ndraws)
        # Flag draws that are out of bounds.
        inbounds = (
            (s_mc >= 0.0)
            & (a_mc >= avlim[0])
            & (a_mc <= avlim[1])
            & (r_mc >= rvlim[0])
            & (r_mc <= rvlim[1])
        )
        for k in range(active.size):
            i = active[k]
            good = np.flatnonzero(inbounds[k])
            take = min(good.size, ndraws - nfilled[i])
            if take > 0:
                sel = good[:take]
                fill = slice(nfilled[i], nfilled[i] + take)
                sdraws[i, fill] = s_mc[k, sel]
                adraws[i, fill] = a_mc[k, sel]
                rdraws[i, fill] = r_mc[k, sel]
                nfilled[i] += take
        active = active[nfilled[active] < ndraws]

    # Any distribution still deficient after `max_attempts` passes gets its
    # remaining slots padded with the mean values (matching the historical
    # per-sample fallback).
    for i in active:
        warnings.warn(
            f"draw_sar: only collected {nfilled[i]}/{ndraws} "
            f"in-bounds samples after {max_attempts} attempts for "
            f"sample {i}. Padding with mean values.",
            RuntimeWarning,
            stacklevel=2,
        )
        sdraws[i, nfilled[i] :] = scales[i]
        adraws[i, nfilled[i] :] = avs[i]
        rdraws[i, nfilled[i] :] = rvs[i]

    return sdraws, adraws, rdraws


@jit(nopython=True, cache=True)
def _cholesky_3x3(A):
    """
    Compute the Cholesky factor of a 3x3 positive SEMI-definite matrix.

    Uses explicit formulas optimized for the 3x3 case. Semi-definite inputs
    are handled with the standard rank-deficient completion: each pivot
    argument is clamped at zero before the square root, and the entries below
    an exactly-zero pivot are set to zero rather than divided (their target
    values are zero for any exact PSD matrix). For strictly positive-definite
    input this is bit-identical to the textbook factorization; for singular
    PSD input it returns a valid factor with ``L @ L.T == A`` instead of
    dividing by zero (which, inside a parallel numba kernel, silently leaves
    the output buffer uninitialized).
    """
    L = np.zeros_like(A)

    # L[0,0] = sqrt(A[0,0])
    L[0, 0] = np.sqrt(max(A[0, 0], 0.0))

    if L[0, 0] > 0.0:
        # L[1,0] = A[1,0] / L[0,0]; L[2,0] = A[2,0] / L[0,0]
        L[1, 0] = A[1, 0] / L[0, 0]
        L[2, 0] = A[2, 0] / L[0, 0]

    # L[1,1] = sqrt(A[1,1] - L[1,0]^2)
    L[1, 1] = np.sqrt(max(A[1, 1] - L[1, 0] * L[1, 0], 0.0))

    if L[1, 1] > 0.0:
        # L[2,1] = (A[2,1] - L[2,0] * L[1,0]) / L[1,1]
        L[2, 1] = (A[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1]

    # L[2,2] = sqrt(A[2,2] - L[2,0]^2 - L[2,1]^2)
    L[2, 2] = np.sqrt(max(A[2, 2] - L[2, 0] * L[2, 0] - L[2, 1] * L[2, 1], 0.0))

    return L


@jit(nopython=True, cache=True, parallel=True)
def _sample_multivariate_normal_jit(mean, cov, size, eps, random_samples):
    """
    Numba-accelerated core multivariate normal sampling.

    Parameters
    ----------
    mean : ndarray of shape (Ndist, dim)
        Means of the multivariate distributions.
    cov : ndarray of shape (Ndist, dim, dim)
        Covariances of the multivariate distributions.
    size : int
        Number of samples to draw from each distribution.
    eps : float
        Regularization parameter for numerical stability.
    random_samples : ndarray of shape (Ndist, dim, size)
        Pre-generated standard normal samples.

    Returns
    -------
    samples : ndarray of shape (dim, size, Ndist)
        Transformed samples.

    Notes
    -----
    Parallelized over the distribution index ``n`` (each distribution's
    regularization, Cholesky factor, and sample transform are independent),
    so the result is bitwise-identical to a serial evaluation.
    """
    N, d = mean.shape

    # Per-distribution: regularize, Cholesky-factor, transform. All independent
    # across n, so prange is safe and bitwise-identical to a serial loop.
    result = np.empty((d, size, N))
    for n in prange(N):
        Kn = cov[n].copy()
        for i in range(d):
            Kn[i, i] += eps
        Ln = _cholesky_3x3(Kn)
        for s in range(size):
            for i in range(d):
                val = mean[n, i]
                for j in range(d):
                    val += Ln[i, j] * random_samples[n, j, s]
                result[i, s, n] = val

    return result


def _antithetic_normals(rstate, N, d, size):
    """
    Standard-normal draws of shape ``(N, d, size)`` arranged in antithetic
    pairs along the last (sample) axis.

    For each base draw ``z`` we also emit ``-z``. Because ``z`` and ``-z`` are
    each marginally standard normal, any Monte-Carlo average over these samples
    remains unbiased, while the negative correlation within a pair cancels the
    linear component of the integrand's variance (variance reduction). When
    ``size`` is odd the final sample is an unpaired ordinary draw. This also
    halves the number of underlying Gaussian draws generated.
    """
    nh = (size + 1) // 2
    base = rstate.normal(loc=0, scale=1, size=d * nh * N).reshape(N, d, nh)
    z = np.empty((N, d, size))
    z[:, :, :nh] = base
    z[:, :, nh:] = -base[:, :, : size - nh]
    return z


def sample_multivariate_normal(
    mean, cov, size=1, eps=1e-30, rstate=None, antithetic=False
):
    """
    Draw samples from many multivariate normal distributions.

    Returns samples from an arbitrary number of multivariate distributions.
    The multivariate distributions must all have the same dimension.
    This function is optimized for drawing from many distributions
    simultaneously using Cholesky decomposition.

    Parameters
    ----------
    mean : `~numpy.ndarray` of shape `(Ndist, dim)` or `(dim,)`
        Means of the various multivariate distributions, where
        `Ndist` is the number of desired distributions and
        `dim` is the dimension of the distributions.

    cov : `~numpy.ndarray` of shape `(Ndist, dim, dim)` or `(dim, dim)`
        Covariances of the various multivariate distributions, where
        `Ndist` is the number of desired distributions and
        `dim` is the dimension of the distributions.

    size : int, optional
        Number of samples to draw from each distribution. Default is `1`.

    eps : float, optional
        Small factor added to covariances prior to Cholesky decomposition.
        Helps ensure numerical stability and should have no effect on the
        outcome. Default is `1e-30`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If None, uses default numpy
        random state.

    antithetic : bool, optional
        If True (only effective for the 3D fast path), draw the underlying
        standard normals in antithetic pairs ``(z, -z)`` along the sample axis.
        This leaves every Monte-Carlo estimate unbiased but reduces its variance
        (and halves the number of Gaussian draws generated). Default is False.

    Returns
    -------
    samples : `~numpy.ndarray` of shape `(dim, size, Ndist)` or `(dim, size)`
        Sampled values. For a single distribution, returns `(dim, size)`.
        For multiple distributions, returns `(dim, size, Ndist)`.

    Notes
    -----
    Provided covariances must be positive semi-definite. Use the `isPSD`
    function from `brutus.utils.math` to check individual matrices if unsure.

    For a single distribution, this function simply calls numpy's
    multivariate_normal. For multiple distributions, it uses Cholesky
    decomposition for efficiency.

    Examples
    --------
    >>> import numpy as np
    >>> # Single distribution
    >>> mean = np.array([0, 1])
    >>> cov = np.array([[1, 0.5], [0.5, 1]])
    >>> samples = sample_multivariate_normal(mean, cov, size=100)
    >>> samples.shape
    (2, 100)

    >>> # Multiple distributions
    >>> means = np.array([[0, 1], [2, 3]])  # 2 distributions, 2D each
    >>> covs = np.array([[[1, 0], [0, 1]], [[2, 0.5], [0.5, 2]]])
    >>> samples = sample_multivariate_normal(means, covs, size=50)
    >>> samples.shape
    (2, 50, 2)
    """
    if rstate is None:
        rstate = np.random

    # If we have a single distribution, just revert to `numpy.random` version.
    if len(np.shape(mean)) == 1:
        samples = rstate.multivariate_normal(mean, cov, size=size)
        return samples.T  # Transpose to match expected (dim, size) format

    # For multiple distributions, check dimension compatibility
    N, d = np.shape(mean)

    if d == 3:
        # Use numba-accelerated version for 3D case
        if antithetic:
            z = _antithetic_normals(rstate, N, d, size)
        else:
            z = rstate.normal(loc=0, scale=1, size=d * size * N).reshape(N, d, size)
        ans = _sample_multivariate_normal_jit(mean, cov, size, eps, z)
    else:
        # Fall back to numpy for non-3D cases
        ans = []
        for i in range(N):
            samples_i = rstate.multivariate_normal(mean[i], cov[i], size=size)
            ans.append(samples_i.T)  # Transpose to match expected format
        ans = np.array(ans)  # Shape: (N, d, size)
        ans = np.transpose(ans, (1, 2, 0))  # Convert to (d, size, N)

    return ans
