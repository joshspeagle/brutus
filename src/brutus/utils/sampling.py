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
from numba import jit

__all__ = ["quantile", "draw_sar", "sample_multivariate_normal"]


def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    This function computes quantiles from a set of samples, optionally
    with weights. For unweighted samples, it uses numpy.percentile.
    For weighted samples, it computes the cumulative distribution function
    and interpolates to find the desired quantiles.

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
    array([2., 3., 4.])

    >>> weights = np.array([1, 1, 1, 1, 10])  # Last sample heavily weighted
    >>> quantile(x, q, weights=weights)
    array([5., 5., 5.])
    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
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
):
    """
    Generate random draws from the joint scale-A_V-R_V posterior for a.

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

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If None, uses default numpy
        random state (or intel-specific version if available).

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

    # Generate realizations for each (scale, av, rv, cov_sar) set.
    nsamps = len(scales)
    sdraws, adraws, rdraws = np.zeros((3, nsamps, ndraws))

    max_attempts = 10000

    for i, (s, a, r, c) in enumerate(zip(scales, avs, rvs, covs_sar)):
        s_chunks, a_chunks, r_chunks = [], [], []
        n_collected = 0
        n_attempts = 0

        # Loop in case a significant chunk of draws are out-of-bounds.
        while n_collected < ndraws:
            n_attempts += 1
            # Draw samples.
            s_mc, a_mc, r_mc = rstate.multivariate_normal([s, a, r], c, size=ndraws).T
            # Flag draws that are out of bounds.
            inbounds = (
                (s_mc >= 0.0)
                & (a_mc >= avlim[0])
                & (a_mc <= avlim[1])
                & (r_mc >= rvlim[0])
                & (r_mc <= rvlim[1])
            )
            s_mc, a_mc, r_mc = s_mc[inbounds], a_mc[inbounds], r_mc[inbounds]

            # Accumulate in-bounds samples.
            s_chunks.append(s_mc)
            a_chunks.append(a_mc)
            r_chunks.append(r_mc)
            n_collected += len(s_mc)

            if n_attempts >= max_attempts:
                warnings.warn(
                    f"draw_sar: only collected {n_collected}/{ndraws} "
                    f"in-bounds samples after {max_attempts} attempts for "
                    f"sample {i}. Padding with mean values.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

        # Concatenate collected samples.
        if n_collected > 0:
            s_temp = np.concatenate(s_chunks)
            a_temp = np.concatenate(a_chunks)
            r_temp = np.concatenate(r_chunks)
        else:
            s_temp = np.array([])
            a_temp = np.array([])
            r_temp = np.array([])

        # If we have enough, cull extras; otherwise pad with mean values.
        if n_collected >= ndraws:
            sdraws[i] = s_temp[:ndraws]
            adraws[i] = a_temp[:ndraws]
            rdraws[i] = r_temp[:ndraws]
        else:
            sdraws[i, :n_collected] = s_temp[:n_collected]
            adraws[i, :n_collected] = a_temp[:n_collected]
            rdraws[i, :n_collected] = r_temp[:n_collected]
            # Pad remaining slots with the mean values.
            sdraws[i, n_collected:] = s
            adraws[i, n_collected:] = a
            rdraws[i, n_collected:] = r

    return sdraws, adraws, rdraws


@jit(nopython=True, cache=True)
def _cholesky_3x3(A):
    """
    Compute Cholesky decomposition of a 3x3 positive definite matrix.

    Uses explicit formulas optimized for 3x3 case to avoid numba limitations.
    """
    L = np.zeros_like(A)

    # L[0,0] = sqrt(A[0,0])
    L[0, 0] = np.sqrt(A[0, 0])

    # L[1,0] = A[1,0] / L[0,0]
    L[1, 0] = A[1, 0] / L[0, 0]

    # L[1,1] = sqrt(A[1,1] - L[1,0]^2)
    L[1, 1] = np.sqrt(A[1, 1] - L[1, 0] * L[1, 0])

    # L[2,0] = A[2,0] / L[0,0]
    L[2, 0] = A[2, 0] / L[0, 0]

    # L[2,1] = (A[2,1] - L[2,0] * L[1,0]) / L[1,1]
    L[2, 1] = (A[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1]

    # L[2,2] = sqrt(A[2,2] - L[2,0]^2 - L[2,1]^2)
    L[2, 2] = np.sqrt(A[2, 2] - L[2, 0] * L[2, 0] - L[2, 1] * L[2, 1])

    return L


@jit(nopython=True, cache=True)
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
    """
    N, d = mean.shape

    # Add regularization to covariance matrices
    K = cov.copy()
    for n in range(N):
        for i in range(d):
            K[n, i, i] += eps

    # Cholesky decomposition using custom 3x3 implementation
    L = np.empty_like(K)
    for n in range(N):
        L[n] = _cholesky_3x3(K[n])

    # Transform samples and write directly to output layout: (dim, size, Ndist)
    result = np.empty((d, size, N))
    for n in range(N):
        for s in range(size):
            for i in range(d):
                val = mean[n, i]
                for j in range(d):
                    val += L[n, i, j] * random_samples[n, j, s]
                result[i, s, n] = val

    return result


def sample_multivariate_normal(mean, cov, size=1, eps=1e-30, rstate=None):
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
