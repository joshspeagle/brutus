#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Line-of-sight dust extinction analysis for 3D dust mapping.

This module provides functionality for modeling 3D dust distribution along
lines-of-sight using stellar posterior samples from individual star fitting.
The implementation uses multi-cloud extinction models with various smoothing
kernels and is designed for use with nested sampling codes like dynesty.

The core approach models the line-of-sight as a series of discrete "clouds"
at different distances. Each cloud parameter pair specifies the *absolute*
mean extinction level for stars located between that cloud and the next one.
This allows reconstruction of 3D dust maps from stellar photometry.

Model Components
----------------
For n clouds, the model has 2n + 4 parameters per line-of-sight:

- **Outlier fraction** (P_b): Fraction of stars that don't follow the model
- **Foreground dispersion** (s_fore): Dispersion scale for the foreground layer
- **Background dispersion** (s_back): Dispersion scale for background clouds
- **Foreground extinction** (A_V,fore): Extinction before first cloud
- **n cloud pairs**: (distance_i, extinction_i) for each cloud, where
  extinction_i is the absolute extinction level behind cloud i

The mean extinction at distance d is the absolute level ``extinction_i`` of
the most recent cloud crossed (or the foreground level ``A_V,fore`` before
the first cloud). Cloud values are *not* summed cumulatively: only with
``additive_foreground=True`` is anything (the foreground) added to the cloud
levels. The kernel (Gaussian, Lorentzian, or top-hat) defines how individual
stellar extinction samples scatter around that mean.

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

1. **Cloud levels**: Each cloud sets the absolute mean extinction for stars
   between it and the next cloud (levels are not summed across clouds)
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
4. Reconstruct 3D dust map from cloud parameters (recall the cloud
   reddenings are absolute levels, not per-cloud increments)

When comparing evidences in step 3 with ``monotonic=True`` (the default in
`los_clouds_loglike_samples`), note that the prior transform draws the
reddening amplitudes as independent uniforms and the monotone constraint is
imposed by rejection in the likelihood. Only a fraction 1/(n+1)! of the
sampled prior volume survives for an n-cloud model, so reported
log-evidences include a -ln((n+1)!) offset relative to a properly
normalized monotone prior. Add ln((n+1)!) back to each lnZ before comparing
models with different numbers of clouds.

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
from scipy.special import logsumexp, ndtr, ndtri

__all__ = [
    "los_clouds_priortransform",
    "los_clouds_loglike_samples",
    "kernel_tophat",
    "kernel_gauss",
    "kernel_lorentz",
]


def _trunc_lognorm_ppf(q, mean, std, low, high):
    """
    Percent-point function of a log-normal whose log is a truncated normal.

    Equivalent to ``np.exp(truncnorm.ppf(q, (low - mean) / std,
    (high - mean) / std, loc=mean, scale=std))`` but ~50x faster: each
    scalar `scipy.stats` frozen-distribution call carries ~0.2 ms of
    argument-validation overhead, and this transform sits inside the
    nested-sampling inner loop (one call per proposed point).

    Parameters
    ----------
    q : float or ndarray
        Quantile(s) in [0, 1].

    mean, std : float
        Mean and standard deviation of the underlying (untruncated) normal.

    low, high : float
        Truncation bounds of the underlying normal (may be ``-np.inf`` /
        ``np.inf``).

    Returns
    -------
    x : float or ndarray
        ``exp`` of the truncated-normal quantile(s).
    """
    a = (low - mean) / std
    b = (high - mean) / std
    cdf_a, cdf_b = ndtr(a), ndtr(b)
    x = ndtri(cdf_a + q * (cdf_b - cdf_a)) * std + mean
    # the quantile lies in [low, high] by construction; clip protects the
    # q -> 0/1 edges where ndtr(a)/ndtr(b) round to exactly 0/1 and ndtri
    # would otherwise return -/+inf past the truncation bound
    return np.exp(np.clip(x, low, high))


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
    is required by the likelihood's per-segment extinction model.

    The log-normal priors on outlier fraction and smoothing concentrate
    probability near zero while allowing occasional larger values.

    The reddening amplitudes are drawn as *independent* uniforms (only the
    distances are sorted). When the likelihood is evaluated with
    ``monotonic=True`` (the default in `los_clouds_loglike_samples`),
    non-monotone amplitude orderings are rejected with -inf, so only a
    fraction 1/(n+1)! of the sampled prior volume is retained for an
    n-cloud model. The evidence computed by nested sampling is therefore
    lnZ = lnZ_monotone - ln((n+1)!) relative to a properly normalized
    monotone prior: add ln((n+1)!) back to each lnZ before comparing
    models with different numbers of clouds.

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

    # pb (outlier fraction): truncated log-normal via the closed-form
    # inverse CDF (see _trunc_lognorm_ppf for why scipy.stats is avoided)
    pb_mean, pb_std, pb_low, pb_high = pb_params
    x[0] = _trunc_lognorm_ppf(u[0], pb_mean, pb_std, pb_low, pb_high)

    # s (fractional smoothing)
    ns = 2  # 2 parameters for foreground + background smoothing
    s_mean, s_std, s_low, s_high = s_params
    x[1:3] = _trunc_lognorm_ppf(u[1:3], s_mean, s_std, s_low, s_high)

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
        "cloud" along the LOS. Each `red` is the *absolute* mean reddening
        level for stars between `dist` and the next cloud (values are not
        summed across clouds); if `template_reds` is provided the cloud
        values are instead dimensionless rescalings of each star's template
        reddening, and if `additive_foreground=True` they are increments
        above the foreground `fred`.

    dsamps : array_like, shape (Nobj, Nsamps)
        Distance samples for each object. Follows the units used in `theta`.

    rsamps : array_like, shape (Nobj, Nsamps)
        Reddening samples for each object. Follows the units in `theta`.
        Non-finite samples are ignored (treated as carrying zero weight).

    kernel : str or callable, optional
        The kernel used to weight the samples along the LOS. If a string is
        passed, a pre-specified kernel will be used. Options include
        `'lorentz'`, `'gauss'`, and `'tophat'`. The built-in kernels are
        renormalized for truncation to `rlims` (see the kernels' `rlims`
        parameter), since the stellar reddening samples cannot scatter
        outside that range. A callable must accept ``(reds, (kmean, ksig))``
        where `kmean`/`ksig` are arrays broadcastable against `reds`; it is
        responsible for its own normalization. Default is `'gauss'`.

    rlims : tuple of float, optional
        The reddening bounds within which we'd like to sample, used to
        normalize both the outlier model and the built-in kernels. Default
        is `(0., 6.)`, which assumes reddening is in units of A_V.

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
        Whether to enforce monotonicity in the fits so that the *effective*
        extinction values must get larger with distance. The constraint is
        applied after the `template_reds`/`additive_foreground` semantics
        are accounted for: with a template the (dimensionless) rescalings
        must be non-decreasing (the foreground, which is in A_V, is not
        compared against them); with an additive foreground the increments
        must be non-negative and non-decreasing. Default is `True`.

        Note that with `monotonic=True` the sampled prior
        (`los_clouds_priortransform`) is *not* renormalized to the monotone
        region, so nested-sampling evidences carry a -ln((n+1)!) offset for
        an n-cloud model; add ln((n+1)!) back before comparing evidences
        across different numbers of clouds.

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

    # Check monotonicity. Distances must always increase; with `monotonic`
    # the EFFECTIVE extinction profile (after the template_reds /
    # additive_foreground semantics below) must be non-decreasing -- raw
    # theta values are only directly comparable in the default mode.
    if not np.all(np.sort(dists) == dists):
        raise ValueError("Distances must be monotonically increasing")
    if monotonic:
        cloud_reds = reds[1:]
        monotone = bool(np.all(np.diff(cloud_reds) >= 0))
        if additive_foreground:
            # effective levels are fred + cloud value (times a positive
            # template rescaling, if any): foreground <= first cloud level
            # iff the first increment is non-negative
            monotone &= bool(cloud_reds[0] >= 0)
        elif template_reds is None:
            # cloud values are absolute levels in the same units as fred
            monotone &= bool(reds[0] <= cloud_reds[0])
        # with a template (and no additive foreground), fred is in A_V while
        # cloud values are dimensionless rescalings: no fred comparison
        if not monotone:
            return -np.inf

    # Sub-sample distance and reddening samples
    if Ndraws > dsamps.shape[1]:
        Ndraws = dsamps.shape[1]  # Use all available samples
    ds, rs = dsamps[:, :Ndraws], rsamps[:, :Ndraws]
    Nobj, Nsamps = ds.shape

    if template_reds is not None:
        template_reds = np.asarray(template_reds)
        if template_reds.shape != (Nobj,):
            raise ValueError(
                f"template_reds must have shape ({Nobj},), "
                f"got {template_reds.shape}"
            )

    # Assign every sample to its (disjoint) distance segment: index 0 is the
    # foreground, index i >= 1 is behind cloud i. A single kernel evaluation
    # with per-sample means/widths is exactly equivalent to evaluating each
    # segment's kernel over all samples and masking the other segments to
    # -inf (the previous implementation), but ~10x faster.
    idx = np.searchsorted(dists, ds, side="right")
    behind = idx > 0

    # Kernel mean per sample (the segment's absolute extinction level)
    kmean = reds[idx]
    if template_reds is not None:
        # cloud values are rescalings of each star's template reddening
        kmean = np.where(behind, kmean * template_reds[:, None], kmean)
    if additive_foreground:
        # foreground is added to (not replaced by) the cloud levels
        kmean = np.where(behind, kmean + reds[0], kmean)
    ksig = np.where(behind, rsmooth, rsmooth0)

    # Samples with distances outside [0, 1e10) fall in no segment, and
    # non-finite reddening samples carry no information: both get zero
    # weight (a NaN in rsamps would otherwise poison the whole sightline)
    valid = (ds >= 0) & (ds < 1e10) & np.isfinite(rs)

    # Compute log-weights by evaluating each reddening sample against the
    # kernel centered on its own segment's extinction level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        if isinstance(kernel, str):
            # built-in kernels renormalize for truncation to the admissible
            # reddening range (samples cannot scatter below A_V = rlims[0])
            logw = kern(rs, (kmean, ksig), rlims=rlims)
        else:
            logw = kern(rs, (kmean, ksig))
        logw = np.where(valid, logw, -np.inf)

    # Compute log-likelihoods across all samples
    logls = logsumexp(logw, axis=1) - np.log(Nsamps)

    # Add in outlier mixture model
    logls = logsumexp(
        a=np.c_[logls, np.full_like(logls, -np.log(area))], b=[(1.0 - pb), pb], axis=1
    )

    # Compute total log-likelihood
    loglike = np.sum(logls)

    return loglike


def kernel_tophat(reds, kp, rlims=None):
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

    rlims : tuple of float, optional
        If provided, the kernel support is clipped to
        ``[rlims[0], rlims[1]]`` and renormalized so the kernel integrates
        to 1 over the admissible reddening range. Without this, kernels
        centered near a boundary (e.g. A_V ~ 0) lose probability mass to
        values the reddening samples can never take, biasing boundary fits.
        Default is `None` (no truncation).

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
    # Extract kernel parameters. The mean may be a full per-object/per-sample
    # array (e.g. when a reddening template makes the cloud mean differ between
    # objects); keep it broadcastable instead of collapsing it to a scalar.
    # Collapsing the mean (the previous behaviour) applied object 0's mean to
    # every object, silently corrupting the likelihood for template fits.
    kmean, kwidth = kp[0], kp[1]
    kwidth = np.asarray(kwidth)

    if np.any(kwidth <= 0):
        raise ValueError(f"Kernel width must be positive, got {kwidth}")

    klow, khigh = kmean - kwidth, kmean + kwidth  # tophat low/high edges
    if rlims is None:
        norm = 2.0 * kwidth
    else:
        # clip the support to the admissible reddening range; a support that
        # lies entirely outside the range has no admissible mass (-inf)
        klow = np.maximum(klow, rlims[0])
        khigh = np.minimum(khigh, rlims[1])
        norm = khigh - klow

    # Compute weights
    inbounds = (reds >= klow) & (reds < khigh)

    # Compute log-weights, avoiding log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logw = np.where(inbounds & (norm > 0), -np.log(norm), -np.inf)

    return logw


def kernel_gauss(reds, kp, rlims=None):
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

    rlims : tuple of float, optional
        If provided, the kernel is renormalized for truncation to
        ``[rlims[0], rlims[1]]`` (divided by its probability mass over that
        range) so it is a proper density on the admissible reddening range.
        Without this, kernels centered near a boundary (e.g. A_V ~ 0) lose
        probability mass to values the reddening samples can never take,
        biasing boundary fits. Default is `None` (no truncation).

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
    # Extract kernel parameters. Keep the mean broadcastable (it may differ
    # per object/sample when a reddening template is applied); collapsing it to
    # a scalar would apply object 0's mean to every object.
    kmean, kstd = kp[0], kp[1]
    kstd = np.asarray(kstd)

    if np.any(kstd <= 0):
        raise ValueError(f"Kernel standard deviation must be positive, got {kstd}")

    norm = np.sqrt(2 * np.pi) * kstd

    # Compute log-weights
    logw = -0.5 * ((reds - kmean) / kstd) ** 2 - np.log(norm)

    if rlims is not None:
        # renormalize by the kernel mass inside the admissible range:
        # Phi((hi - mu)/sig) - Phi((lo - mu)/sig)
        mass = ndtr((rlims[1] - kmean) / kstd) - ndtr((rlims[0] - kmean) / kstd)
        with np.errstate(divide="ignore", invalid="ignore"):
            logw = np.where(mass > 0, logw - np.log(mass), -np.inf)

    return logw


def kernel_lorentz(reds, kp, rlims=None):
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

    rlims : tuple of float, optional
        If provided, the kernel is renormalized for truncation to
        ``[rlims[0], rlims[1]]`` (divided by its probability mass over that
        range) so it is a proper density on the admissible reddening range.
        Without this, kernels centered near a boundary (e.g. A_V ~ 0) lose
        probability mass to values the reddening samples can never take,
        biasing boundary fits. Default is `None` (no truncation).

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
    # Extract kernel parameters. Keep the mean broadcastable (it may differ
    # per object/sample when a reddening template is applied); collapsing it to
    # a scalar would apply object 0's mean to every object.
    kmean, khwhm = kp[0], kp[1]
    khwhm = np.asarray(khwhm)

    if np.any(khwhm <= 0):
        raise ValueError(f"Kernel HWHM must be positive, got {khwhm}")

    norm = np.pi * khwhm

    # Compute log-weights
    logw = -np.log(1.0 + ((reds - kmean) / khwhm) ** 2) - np.log(norm)

    if rlims is not None:
        # renormalize by the Cauchy mass inside the admissible range:
        # (arctan((hi - mu)/g) - arctan((lo - mu)/g)) / pi
        mass = (
            np.arctan((rlims[1] - kmean) / khwhm)
            - np.arctan((rlims[0] - kmean) / khwhm)
        ) / np.pi
        with np.errstate(divide="ignore", invalid="ignore"):
            logw = np.where(mass > 0, logw - np.log(mass), -np.inf)

    return logw
