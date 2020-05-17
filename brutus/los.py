#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Line-of-sight (LOS) fitting utilities.

"""

from __future__ import (print_function, division)

import warnings
import numpy as np
from scipy.stats import truncnorm

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["LOS_clouds_priortransform", "LOS_clouds_loglike_samples",
           "kernel_tophat", "kernel_gauss", "kernel_lorentz"]


def LOS_clouds_priortransform(u, rlims=(0., 6.), dlims=(4., 19.),
                              pb_params=(-3., 0.7, -np.inf, 0.),
                              s_params=(-3., 0.3, -np.inf, 0.),
                              dust_template=False, nlims=(0.2, 2)):
    """
    The "prior transform" for the LOS fit that converts from draws on the
    N-dimensional unit cube to samples from the prior. Used in nested sampling
    methods. Assumes uniform priors for distance and reddening
    and a (truncated) log-normal in outlier fraction.

    Parameters
    ----------
    u : `~numpy.ndarray` of shape `(Nparams)`
        The `2 + 2 * Nclouds` values drawn from the unit cube.
        Contains the portion of outliers `P_b`, followed by the smoothing `s`,
        followed by the foreground reddening `fred`, followed by a series of
        `(dist, red)` pairs for each "cloud" along the LOS.

    rlims : 2-tuple, optional
        The reddening bounds within which we'd like to sample. Default is
        `(0., 6.)`, which also assumes reddening is in units of Av.

    dlims : 2-tuple, optional
        The distance bounds within which we'd like to sample. Default is
        `(4., 19.)`, which also assumes distance is in units of distance
        modulus.

    pb_params : 4-tuple, optional
        Mean, standard deviation, lower bound, and upper bound for a
        truncated log-normal distribution used as a prior for the outlier
        model. The default is `(-3., 0.7, -np.inf, 0.)`, which corresponds
        to a mean of 0.05, a standard deviation of a factor of 2, a lower
        bound of 0, and an upper bound of 1.

    s_params : 4-tuple, optional
        Mean, standard deviation, lower bound, and upper bound for a
        truncated log-normal distribution used as a prior for the
        smoothing along the reddening axis (in %). The default is
        `(-3.5, 0.7, -np.inf, 0.)`, which corresponds to a mean of 0.05, a
        standard deviation of a factor of 1.35, a lower bound of 0, and an
        upper bound of 1.

    dust_template : bool, optional
        Whether or not to use a sptial distribution for the dust based on
        a particular template. If true, dust along the line of sight
        will be in terms of rescalings of the template rather than
        Av. Default is `False`.

    nlims : 2-tuple, optional
        Lower and upper bounds for the uniform prior for the rescaling
        applied to the Planck spatial reddening template.
        Default is `(0.2, 2.)`.

    Returns
    -------
    x : `~numpy.ndarray` of shape `(Nparams)`
        The transformed parameters.

    """

    # Initialize values.
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

    # distances
    x[ns + 2::2] = np.sort(u[ns + 2::2]) * (dlims[1] - dlims[0]) + dlims[0]

    # foreground reddening
    x[ns + 1] = u[ns + 1] * (rlims[1] - rlims[0]) + rlims[0]

    # cloud reddenings
    dsort = np.argsort(u[ns + 2::2])  # sort distances
    x[ns + 3::2] = (u[ns + 3::2][dsort]) * (rlims[1] - rlims[0]) + rlims[0]

    if dust_template:
        # replace with rescalings for the template
        x[ns + 3::2] = u[ns + 3::2][dsort] * (nlims[1] - nlims[0]) + nlims[0]

    return x


def LOS_clouds_loglike_samples(theta, dsamps, rsamps, kernel='gauss',
                               rlims=(0., 6.), template_reds=None,
                               Ndraws=25, additive_foreground=False):
    """
    Compute the log-likelihood for the cumulative reddening along the
    line of sight (LOS) parameterized by `theta`, given a set of input
    reddening and distance draws. Assumes a uniform outlier model in distance
    and reddening across our binned posteriors.

    Parameters
    ----------
    theta : `~numpy.ndarray` of shape `(Nparams,)`
        A collection of parameters that characterizes the cumulative
        reddening along the LOS. Contains the fraction of outliers `P_b`
        followed by the fractional reddening smoothing for the foreground `s0`
        and background `s` followed by the foreground reddening `fred`
        followed by a series of `(dist, red)` pairs for each
        "cloud" along the LOS.

    dsamps : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Distance samples for each object. Follows the units used in `theta`.

    rsamps : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Reddening samples for each object. Follows the units in `theta`.

    kernel : str or function, optional
        The kernel used to weight the samples along the LOS. If a string is
        passed, a pre-specified kernel will be used. Options include
        `'lorentz'`, `'gauss'`, and `'tophat'`. Default is `'gauss'`.

    rlims : 2-tuple, optional
        The reddening bounds within which we'd like to sample. Default is
        `(0., 6.)`, which also assumes reddening is in units of Av.

    template_reds : `~numpy.ndarray` of shape `(Nobj)`, optional
        Reddenings for each star based on a spatial dust template.
        If not provided, the same reddening value in a given distance
        bin will be fit to all stars. If provided, a rescaled version of the
        individual reddenings will be fit instead.

    Ndraws : int, optional
        The number of draws to use for each star. Default is `25`.

    additive_foreground : bool, optional
        Whether the foreground is treated as just another value or added
        to all background values. Default is `False`.

    Returns
    -------
    loglike : float
        The computed log-likelihood.

    """

    # Check kernel
    KERNELS = {'tophat': kernel_tophat, 'gauss': kernel_gauss,
               'lorentz': kernel_lorentz}
    if kernel in KERNELS:
        kern = KERNELS[kernel]
    elif callable(kernel):
        kern = kernel
    else:
        raise ValueError("The kernel provided is not a valid function nor "
                         "one of the pre-defined options. Please provide a "
                         "valid kernel.")

    # Grab parameters.
    pb, s0, s = theta[0], theta[1], theta[2]
    reds, dists = np.atleast_1d(theta[3::2]), np.atleast_1d(theta[4::2])
    area = (rlims[1] - rlims[0])
    rsmooth = s * area
    rsmooth0 = s0 * area

    # Define cloud edges ("distance bounds").
    xedges = np.concatenate(([0], dists, [1e10]))

    # Sub-sample distance and reddening samples.
    ds, rs = dsamps[:, :Ndraws], rsamps[:, :Ndraws]
    Nobj, Nsamps = ds.shape

    # Reshape sigmas to match samples.
    rsmooth, rsmooth0 = np.full_like(rs, rsmooth), np.full_like(rs, rsmooth0)

    # Get reddenings to each star in each distance slice (kernel mean).
    reds = np.array([np.full_like(rs, r) for r in reds])

    # Adjust reddenings after the foreground if a spatial template is used.
    if template_reds is not None:
        reds[1:] *= template_reds[None, :, None]  # reds[1:] are rescalings

    # Adjust reddenings after the foreground if needed.
    if additive_foreground:
        reds[1:] += reds[0]  # add foreground to background

    # Define kernel parameters (mean, sigma) per LOS chunk.
    kparams = np.array([(r, rsmooth) for r in reds])
    kparams[0][1] = rsmooth0

    # Compute log-weights for samples along the LOS by evaluating reddening
    # samples within each segment against the associated centered kernel.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        logw = np.array([kern(rs, kp) + np.log((ds >= xl) & (ds < xh))
                         for xl, xh, kp in zip(xedges[:-1], xedges[1:],
                                               kparams)])

    # Compute log-likelihoods across all samples and clouds.
    logls = logsumexp(logw, axis=(0, 2)) - np.log(Nsamps)

    # Add in outlier mixture model.
    logls = logsumexp(a=np.c_[logls, np.full_like(logls, -np.log(area))],
                      b=[(1. - pb), pb], axis=1)

    # Compute total log-likeihood.
    loglike = np.sum(logls)

    return loglike


def kernel_tophat(reds, kp):
    """
    Compute a weighted sum of the provided reddening draws using a Top-Hat
    kernel.

    Parameters
    ----------
    reds : `~numpy.ndarray` of shape `(Nsamps)`
        Distance samples for each object.

    kp : 2-tuple
        The kernel parameters `(mean, half-bin-width)`.

    Returns
    -------
    logw : `~numpy.ndarray` of shape `(Nsamps)`
        Log(weights).

    """

    # Extract kernel parameters.
    kmean, kwidth = kp[0], kp[1]
    klow, khigh = kmean - kwidth, kmean + kwidth  # tophat low/high edges
    norm = 2. * kwidth

    # Compute weights.
    inbounds = (reds >= klow) & (reds < khigh)

    # Compute log-sum.
    logw = np.log(inbounds) - np.log(norm)

    return logw


def kernel_gauss(reds, kp):
    """
    Compute a weighted sum of the provided reddening draws using a Gaussian
    kernel.

    Parameters
    ----------
    reds : `~numpy.ndarray` of shape `(Nsamps)`
        Distance samples for each object.

    kp : 2-tuple
        The kernel parameters `(mean, standard deviation)`.

    Returns
    -------
    logw : `~numpy.ndarray` of shape `(Nsamps)`
        Log(weights).

    """

    # Extract kernel parameters.
    kmean, kstd = kp[0], kp[1]
    norm = np.sqrt(2 * np.pi) * kstd

    # Compute log-weights.
    logw = -0.5 * ((reds - kmean) / kstd)**2 - np.log(norm)

    return logw


def kernel_lorentz(reds, kp):
    """
    Compute a weighted sum of the provided reddening draws using a Lorentzian
    kernel.

    Parameters
    ----------
    reds : `~numpy.ndarray` of shape `(Nsamps)`
        Distance samples for each object.

    kp : 2-tuple
        The kernel parameters `(mean, HWHM)`.

    Returns
    -------
    logw : `~numpy.ndarray` of shape `(Nsamps)`
        Log(weights).

    """

    # Extract kernel parameters.
    kmean, khwhm = kp[0], kp[1]
    norm = np.pi * khwhm

    # Compute log-weights.
    logw = -np.log(1. + ((reds - kmean) / khwhm)**2) - np.log(norm)

    return logw
