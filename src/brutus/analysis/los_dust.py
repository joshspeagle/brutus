#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Line-of-sight dust extinction analysis for 3D dust mapping.

This module provides functionality for modeling 3D dust distribution along
lines-of-sight using stellar posterior samples from individual star fitting.
The implementation uses multi-cloud extinction models with various smoothing
kernels and is designed for use with nested sampling codes like dynesty.

The core approach models cumulative extinction along a line-of-sight as a
series of discrete "clouds" at different distances, each contributing a mean
extinction. This allows reconstruction of 3D dust maps from stellar photometry.

Model Components
----------------
For n clouds, the model has 2n + 4 parameters per line-of-sight:

- **Outlier fraction** (P_b): Fraction of stars that don't follow the model
- **Foreground dispersion** (s_fore): Dispersion scale for the foreground layer
- **Background dispersion** (s_back): Dispersion scale for background clouds
- **Foreground extinction** (A_V,fore): Extinction before first cloud
- **n cloud pairs**: (distance_i, extinction_i) for each cloud

The cumulative extinction at distance d is computed by summing contributions
from all clouds closer than d. Each cloud layer has a mean extinction, and the
kernel (Gaussian, Lorentzian, or top-hat) defines how individual stellar
extinction samples scatter around that mean.

Functions
---------
los_clouds_priortransform : Prior transformation
    Transform unit cube to physical parameters for nested sampling
los_clouds_loglike_samples : Likelihood function
    Compute log-likelihood given stellar distance/extinction samples
kernel_tophat : Top-hat kernel
    Uniform dispersion of extinction within a fixed range around the mean
kernel_gauss : Gaussian kernel
    Gaussian dispersion of extinction around the cloud mean
kernel_lorentz : Lorentzian kernel
    Heavy-tailed dispersion of extinction around the cloud mean

See Also
--------
brutus.analysis.individual.BruteForce : Provides stellar posterior samples
brutus.priors.extinction : Extinction priors
brutus.dust.maps : 3D dust map utilities

Notes
-----
The likelihood computation accounts for:

1. **Cloud contributions**: Each cloud adds extinction for stars behind it
2. **Dispersion**: Kernel function defines scatter of extinction values around
   the cloud mean at each distance step
3. **Outlier model**: Uniform distribution in (distance, extinction)
4. **Foreground component**: Extinction before first cloud

The model is fit using nested sampling (e.g., dynesty) which requires:
- Prior transform: los_clouds_priortransform
- Log-likelihood: los_clouds_loglike_samples

Typical workflow:
1. Fit individual stars to get (distance, extinction) posteriors
2. For each sightline, run nested sampling with n-cloud model
3. Compare evidences for different n to select best model
4. Reconstruct 3D dust map from cloud parameters

Examples
--------
Setting up nested sampling for 2-cloud model:

>>> import numpy as np
>>> from dynesty import NestedSampler
>>> from brutus.analysis.los_dust import (
...     los_clouds_priortransform,
...     los_clouds_loglike_samples
... )
>>>
>>> # Stellar posteriors: distance (DM) and extinction (A_V)
>>> # dsamps shape: (n_stars, n_samples)
>>> # rsamps shape: (n_stars, n_samples)
>>>
>>> # For 2 clouds: pb, s0, s, fred, d1, r1, d2, r2 (8 params)
>>> ndim = 8
>>>
>>> def prior_transform(u):
...     return los_clouds_priortransform(
...         u, rlims=(0, 6), dlims=(4, 19)
...     )
>>>
>>> def loglike(theta):
...     return los_clouds_loglike_samples(
...         theta, dsamps, rsamps, kernel='gauss'
...     )
>>>
>>> sampler = NestedSampler(loglike, prior_transform, ndim)
>>> sampler.run_nested()
>>> results = sampler.results
"""

import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm

__all__ = [
    "los_clouds_priortransform",
    "los_clouds_loglike_samples",
    "kernel_tophat",
    "kernel_gauss",
    "kernel_lorentz",
]


def los_clouds_priortransform(
    u,
    rlims=(0.0, 6.0),
    dlims=(4.0, 19.0),
    pb_params=(-3.0, 0.7, -np.inf, 0.0),
    s_params=(-3.0, 0.3, -np.inf, 0.0),
    dust_template=False,
    nlims=(0.2, 2),
):
    """
    Transform unit cube samples to physical parameters for LOS dust fitting.

    The "prior transform" for the LOS fit that converts from draws on the
    N-dimensional unit cube to samples from the prior. Used in nested sampling
    methods. Assumes uniform priors for distance and reddening
    and a (truncated) log-normal in outlier fraction.

    Parameters
    ----------
    u : array_like, shape (Nparams,)
        The `Nparams` values drawn from the unit cube.
        Contains the portion of outliers `P_b`, followed by the
        foreground smoothing `sfore` and background smoothing `sback`,
        followed by the foreground reddening `fred`, followed by a series of
        `(dist, red)` pairs for each "cloud" along the LOS.

    rlims : tuple of float, optional
        The reddening bounds within which we'd like to sample. Default is
        `(0., 6.)`, which assumes reddening is in units of A_V.

    dlims : tuple of float, optional
        The distance bounds within which we'd like to sample. Default is
        `(4., 19.)`, which assumes distance is in units of distance
        modulus.

    pb_params : tuple of float, optional
        Mean, standard deviation, lower bound, and upper bound for a
        truncated log-normal distribution used as a prior for the outlier
        model. The default is `(-3., 0.7, -np.inf, 0.)`, which corresponds
        to a mean of 0.05, a standard deviation of a factor of 2, a lower
        bound of 0, and an upper bound of 1.

    s_params : tuple of float, optional
        Mean, standard deviation, lower bound, and upper bound for a
        truncated log-normal distribution used as a prior for the
        smoothing along the reddening axis (in %). The default is
        `(-3., 0.3, -np.inf, 0.)`, which corresponds to a mean of 0.05, a
        standard deviation of a factor of 1.35, a lower bound of 0, and an
        upper bound of 1.

    dust_template : bool, optional
        Whether or not to use a spatial distribution for the dust based on
        a particular template. If true, dust along the line of sight
        will be in terms of rescalings of the template rather than
        A_V. Default is `False`.

    nlims : tuple of float, optional
        Lower and upper bounds for the uniform prior for the rescaling
        applied to the Planck spatial reddening template.
        Default is `(0.2, 2.)`.

    Returns
    -------
    x : ndarray, shape (Nparams,)
        The transformed physical parameters in order:
        [pb, s_fore, s_back, A_V_fore, d1, A_V1, d2, A_V2, ...]
        where di are sorted in increasing order.

    See Also
    --------
    los_clouds_loglike_samples : Likelihood function for these parameters
    kernel_gauss : Gaussian smoothing kernel
    kernel_lorentz : Lorentzian smoothing kernel

    Notes
    -----
    The prior distributions are:

    - **Outlier fraction** (P_b): Truncated log-normal with median ~0.05
    - **Smoothing** (s_fore, s_back): Truncated log-normal with median ~0.05
    - **Foreground extinction**: Uniform over rlims
    - **Cloud distances**: Uniform over dlims, sorted in increasing order
    - **Cloud extinctions**: Uniform over rlims, ordered by distance

    The sorting ensures clouds are ordered by increasing distance, which
    is required for the cumulative extinction calculation.

    The log-normal priors on outlier fraction and smoothing concentrate
    probability near zero while allowing occasional larger values.

    Examples
    --------
    >>> import numpy as np
    >>> # For 1-cloud model: pb, s0, s, fred, dist1, red1 (6 parameters)
    >>> u = np.random.uniform(0, 1, 6)
    >>> params = los_clouds_priortransform(u)
    >>> print(f"Outlier fraction: {params[0]:.3f}")
    >>> print(f"Cloud distance (DM): {params[4]:.1f}")
    >>> print(f"Cloud extinction (A_V): {params[5]:.2f}")
    """
    # Input validation
    u = np.asarray(u)
    if u.ndim != 1:
        raise ValueError("Input u must be a 1D array")
    if np.any((u < 0) | (u > 1)):
        raise ValueError("All values in u must be between 0 and 1")

    if len(rlims) != 2 or rlims[0] >= rlims[1]:
        raise ValueError("rlims must be a 2-tuple with rlims[0] < rlims[1]")
    if len(dlims) != 2 or dlims[0] >= dlims[1]:
        raise ValueError("dlims must be a 2-tuple with dlims[0] < dlims[1]")

    # Initialize values
    x = np.array(u)

    # pb (outlier fraction)
    pb_mean, pb_std, pb_low, pb_high = pb_params
    a = (pb_low - pb_mean) / pb_std  # set normalized lower bound
    b = (pb_high - pb_mean) / pb_std  # set normalized upper bound
    x[0] = np.exp(truncnorm.ppf(u[0], a, b, loc=pb_mean, scale=pb_std))

    # s (fractional smoothing)
    ns = 2  # 2 parameters for foreground + background smoothing
    s_mean, s_std, s_low, s_high = s_params
    a = (s_low - s_mean) / s_std  # set normalized lower bound
    b = (s_high - s_mean) / s_std  # set normalized upper bound
    x[1] = np.exp(truncnorm.ppf(u[1], a, b, loc=s_mean, scale=s_std))
    x[2] = np.exp(truncnorm.ppf(u[2], a, b, loc=s_mean, scale=s_std))

    # distances - must be sorted (monotonically increasing)
    x[ns + 2 :: 2] = np.sort(u[ns + 2 :: 2]) * (dlims[1] - dlims[0]) + dlims[0]

    # foreground reddening
    x[ns + 1] = u[ns + 1] * (rlims[1] - rlims[0]) + rlims[0]

    # cloud reddenings - sorted by distance order
    dsort = np.argsort(u[ns + 2 :: 2])  # sort distances
    x[ns + 3 :: 2] = (u[ns + 3 :: 2][dsort]) * (rlims[1] - rlims[0]) + rlims[0]

    if dust_template:
        # replace with rescalings for the template
        if len(nlims) != 2 or nlims[0] >= nlims[1]:
            raise ValueError("nlims must be a 2-tuple with nlims[0] < nlims[1]")
        x[ns + 3 :: 2] = u[ns + 3 :: 2][dsort] * (nlims[1] - nlims[0]) + nlims[0]

    return x


def los_clouds_loglike_samples(
    theta,
    dsamps,
    rsamps,
    kernel="gauss",
    rlims=(0.0, 6.0),
    template_reds=None,
    Ndraws=25,
    additive_foreground=False,
    monotonic=True,
):
    """
    Compute log-likelihood for multi-cloud extinction model along line-of-sight.

    Compute the log-likelihood for the cumulative reddening along the
    line of sight (LOS) parameterized by `theta`, given a set of input
    reddening and distance draws. Assumes a uniform outlier model in distance
    and reddening across our binned posteriors.

    Parameters
    ----------
    theta : array_like, shape (Nparams,)
        A collection of parameters that characterizes the cumulative
        reddening along the LOS. Contains the fraction of outliers `P_b`
        followed by the fractional reddening smoothing for the foreground `s0`
        and background `s` followed by the foreground reddening `fred`
        followed by a series of `(dist, red)` pairs for each
        "cloud" along the LOS.

    dsamps : array_like, shape (Nobj, Nsamps)
        Distance samples for each object. Follows the units used in `theta`.

    rsamps : array_like, shape (Nobj, Nsamps)
        Reddening samples for each object. Follows the units in `theta`.

    kernel : str or callable, optional
        The kernel used to weight the samples along the LOS. If a string is
        passed, a pre-specified kernel will be used. Options include
        `'lorentz'`, `'gauss'`, and `'tophat'`. Default is `'gauss'`.

    rlims : tuple of float, optional
        The reddening bounds within which we'd like to sample. Default is
        `(0., 6.)`, which assumes reddening is in units of A_V.

    template_reds : array_like, shape (Nobj,), optional
        Reddenings for each star based on a spatial dust template.
        If not provided, the same reddening value in a given distance
        bin will be fit to all stars. If provided, a rescaled version of the
        individual reddenings will be fit instead.

    Ndraws : int, optional
        The number of draws to use for each star. Default is `25`.

    additive_foreground : bool, optional
        Whether the foreground is treated as just another value or added
        to all background values. Default is `False`.

    monotonic : bool, optional
        Whether to enforce monotonicity in the fits so that the values
        must get larger with distance. Default is `True`.

    Returns
    -------
    loglike : float
        The computed log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate synthetic stellar samples
    >>> nstars = 10
    >>> nsamps = 50
    >>> dsamps = np.random.uniform(6, 12, (nstars, nsamps))  # distance modulus
    >>> rsamps = np.random.uniform(0, 2, (nstars, nsamps))   # A_V extinction
    >>>
    >>> # 1-cloud model parameters: [pb, s0, s, fred, dist1, red1]
    >>> theta = [0.1, 0.05, 0.05, 0.2, 8.0, 0.5]
    >>>
    >>> # Compute likelihood
    >>> loglike = los_clouds_loglike_samples(theta, dsamps, rsamps)
    >>> print(f"Log-likelihood: {loglike:.2f}")
    """
    # Input validation
    theta = np.asarray(theta)
    dsamps = np.asarray(dsamps)
    rsamps = np.asarray(rsamps)

    if theta.ndim != 1:
        raise ValueError("theta must be a 1D array")
    if dsamps.ndim != 2 or rsamps.ndim != 2:
        raise ValueError("dsamps and rsamps must be 2D arrays")
    if dsamps.shape != rsamps.shape:
        raise ValueError("dsamps and rsamps must have the same shape")

    # Check kernel
    KERNELS = {
        "tophat": kernel_tophat,
        "gauss": kernel_gauss,
        "lorentz": kernel_lorentz,
    }
    if kernel in KERNELS:
        kern = KERNELS[kernel]
    elif callable(kernel):
        kern = kernel
    else:
        raise ValueError(
            f"The kernel '{kernel}' is not valid. "
            f"Options: {list(KERNELS.keys())} or callable"
        )

    # Grab parameters
    if len(theta) < 6:
        raise ValueError("theta must have at least 6 parameters for 1-cloud model")
    if (len(theta) - 4) % 2 != 0:
        raise ValueError("theta must have 4 + 2*n parameters for n-cloud model")

    pb, s0, s = theta[0], theta[1], theta[2]
    reds, dists = np.atleast_1d(theta[3::2]), np.atleast_1d(theta[4::2])
    area = rlims[1] - rlims[0]
    rsmooth = s * area
    rsmooth0 = s0 * area

    # Validate parameter ranges
    if not (0 <= pb <= 1):
        raise ValueError(f"Outlier fraction pb must be in [0,1], got {pb}")
    if s0 < 0 or s < 0:
        raise ValueError(
            f"Smoothing parameters must be non-negative, got s0={s0}, s={s}"
        )

    # Check monotonicity
    if not np.all(np.sort(dists) == dists):
        raise ValueError("Distances must be monotonically increasing")
    if monotonic:
        if not np.all(np.sort(reds) == reds):
            # If monotonicity is enforced, non-monotonic solutions disallowed
            return -np.inf

    # Define cloud edges ("distance bounds")
    xedges = np.concatenate(([0], dists, [1e10]))

    # Sub-sample distance and reddening samples
    if Ndraws > dsamps.shape[1]:
        Ndraws = dsamps.shape[1]  # Use all available samples
    ds, rs = dsamps[:, :Ndraws], rsamps[:, :Ndraws]
    Nobj, Nsamps = ds.shape

    # Get reddenings to each star in each distance slice (kernel mean)
    reds_expanded = np.array([np.full_like(rs, r) for r in reds])

    # Adjust reddenings after the foreground if a spatial template is used
    if template_reds is not None:
        template_reds = np.asarray(template_reds)
        if template_reds.shape != (Nobj,):
            raise ValueError(
                f"template_reds must have shape ({Nobj},), "
                f"got {template_reds.shape}"
            )
        reds_expanded[1:] *= template_reds[None, :, None]  # reds[1:] are rescalings

    # Adjust reddenings after the foreground if needed
    if additive_foreground:
        reds_expanded[1:] += reds_expanded[0]  # add foreground to background

    # Create smoothing arrays for each distance bin
    rsmooth_full = np.full_like(rs, rsmooth)
    rsmooth0_full = np.full_like(rs, rsmooth0)

    # Define kernel parameters (mean, sigma) per LOS chunk
    kparams = []
    for i, r_exp in enumerate(reds_expanded):
        if i == 0:  # foreground
            kparams.append((r_exp, rsmooth0_full))
        else:  # background clouds
            kparams.append((r_exp, rsmooth_full))

    # Compute log-weights for samples along the LOS by evaluating reddening
    # samples within each segment against the associated centered kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        logw = np.array(
            [
                kern(rs, kp) + np.log((ds >= xl) & (ds < xh))
                for xl, xh, kp in zip(xedges[:-1], xedges[1:], kparams)
            ]
        )

    # Compute log-likelihoods across all samples and clouds
    logls = logsumexp(logw, axis=(0, 2)) - np.log(Nsamps)

    # Add in outlier mixture model
    logls = logsumexp(
        a=np.c_[logls, np.full_like(logls, -np.log(area))], b=[(1.0 - pb), pb], axis=1
    )

    # Compute total log-likelihood
    loglike = np.sum(logls)

    return loglike


def kernel_tophat(reds, kp):
    """
    Compute weighted log-probabilities using a Top-Hat kernel.

    The kernel defines uniform dispersion of extinction values around the
    cloud mean: samples within ``mean +/- half_bin_width`` are equally
    likely, and samples outside are assigned zero probability.

    Parameters
    ----------
    reds : array_like, shape (Nsamps,)
        Reddening samples for each object.

    kp : tuple of float
        The kernel parameters `(mean, half_bin_width)`.

    Returns
    -------
    logw : ndarray, shape (Nsamps,)
        Log-weights for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> reds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> kp = (0.5, 0.2)  # mean=0.5, half-width=0.2, so range [0.3, 0.7]
    >>> logw = kernel_tophat(reds, kp)
    >>> # Samples within [0.3, 0.7] should have higher weights
    """
    # Extract kernel parameters
    kmean, kwidth = kp[0], kp[1]

    # Handle array inputs by taking first element if needed
    if hasattr(kmean, "__len__"):
        kmean = kmean.flat[0] if hasattr(kmean, "flat") else kmean[0]
    if hasattr(kwidth, "__len__"):
        kwidth = kwidth.flat[0] if hasattr(kwidth, "flat") else kwidth[0]

    if kwidth <= 0:
        raise ValueError(f"Kernel width must be positive, got {kwidth}")

    klow, khigh = kmean - kwidth, kmean + kwidth  # tophat low/high edges
    norm = 2.0 * kwidth

    # Compute weights
    inbounds = (reds >= klow) & (reds < khigh)

    # Compute log-weights, avoiding log(0)
    logw = np.where(inbounds, -np.log(norm), -np.inf)

    return logw


def kernel_gauss(reds, kp):
    """
    Compute weighted log-probabilities using a Gaussian kernel.

    The kernel defines Gaussian dispersion of extinction values around the
    cloud mean: samples are weighted by a normal distribution centered on
    the mean extinction with the given standard deviation.

    Parameters
    ----------
    reds : array_like, shape (Nsamps,)
        Reddening samples for each object.

    kp : tuple of float
        The kernel parameters `(mean, standard_deviation)`.

    Returns
    -------
    logw : ndarray, shape (Nsamps,)
        Log-weights for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> reds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> kp = (0.5, 0.1)  # mean=0.5, std=0.1
    >>> logw = kernel_gauss(reds, kp)
    >>> # Sample at 0.5 should have highest weight
    """
    # Extract kernel parameters
    kmean, kstd = kp[0], kp[1]

    # Handle array inputs by taking first element if needed
    if hasattr(kmean, "__len__"):
        kmean = kmean.flat[0] if hasattr(kmean, "flat") else kmean[0]
    if hasattr(kstd, "__len__"):
        kstd = kstd.flat[0] if hasattr(kstd, "flat") else kstd[0]

    if kstd <= 0:
        raise ValueError(f"Kernel standard deviation must be positive, got {kstd}")

    norm = np.sqrt(2 * np.pi) * kstd

    # Compute log-weights
    logw = -0.5 * ((reds - kmean) / kstd) ** 2 - np.log(norm)

    return logw


def kernel_lorentz(reds, kp):
    """
    Compute weighted log-probabilities using a Lorentzian kernel.

    The kernel defines heavy-tailed dispersion of extinction values around
    the cloud mean: samples are weighted by a Cauchy/Lorentzian distribution
    centered on the mean extinction, allowing larger deviations from the
    mean than a Gaussian kernel.

    Parameters
    ----------
    reds : array_like, shape (Nsamps,)
        Reddening samples for each object.

    kp : tuple of float
        The kernel parameters `(mean, HWHM)` where HWHM is the
        half-width at half-maximum.

    Returns
    -------
    logw : ndarray, shape (Nsamps,)
        Log-weights for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> reds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> kp = (0.5, 0.1)  # mean=0.5, HWHM=0.1
    >>> logw = kernel_lorentz(reds, kp)
    >>> # Sample at 0.5 should have highest weight, with heavy tails
    """
    # Extract kernel parameters
    kmean, khwhm = kp[0], kp[1]

    # Handle array inputs by taking first element if needed
    if hasattr(kmean, "__len__"):
        kmean = kmean.flat[0] if hasattr(kmean, "flat") else kmean[0]
    if hasattr(khwhm, "__len__"):
        khwhm = khwhm.flat[0] if hasattr(khwhm, "flat") else khwhm[0]

    if khwhm <= 0:
        raise ValueError(f"Kernel HWHM must be positive, got {khwhm}")

    norm = np.pi * khwhm

    # Compute log-weights
    logw = -np.log(1.0 + ((reds - kmean) / khwhm) ** 2) - np.log(norm)

    return logw
