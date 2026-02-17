#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extinction priors for Bayesian stellar parameter estimation.

This module provides log-prior functions for dust extinction modeling
using 3D dust maps. These priors incorporate spatial dust distribution
information to constrain extinction in stellar fitting.

Functions
---------
logp_extinction : Dust map extinction prior
    Gaussian prior from 3D dust maps (e.g., Bayestar)

See Also
--------
brutus.dust.maps : 3D dust map utilities
brutus.priors.galactic : Galactic structure priors
brutus.analysis.individual.BruteForce : Uses extinction priors for fitting

Notes
-----
The extinction prior uses 3D dust maps (e.g., Bayestar from Green et al.
2015, 2018) which provide distance-dependent extinction estimates across
the sky.

The prior is Gaussian when dust map data is available, and uniform when
coverage is unavailable. This gracefully handles regions outside the
mapped volume.

Examples
--------
>>> from brutus.priors.extinction import logp_extinction
>>> from brutus.dust import Bayestar
>>> import numpy as np
>>>
>>> # Load 3D dust map and evaluate extinction prior at a distance
>>> # dustmap = Bayestar('bayestar2019_v1.h5')
>>> # logp = logp_extinction([0.1, 0.5], dustmap, [180.0, 30.0], distance=1.0)
"""

import numpy as np

__all__ = ["logp_extinction"]


def logp_extinction(avs, dustmap, coord, distance=None, return_components=False):
    r"""
    Log-prior for dust extinction using 3D dust maps.

    Implements Gaussian extinction priors based on dust maps with systematic
    uncertainty treatment. Supports both 3D dust maps (e.g., Bayestar) that
    return distance-resolved profiles and simpler maps that return a single
    mean and standard deviation.

    Parameters
    ----------
    avs : array_like
        Extinction values (A_V) in magnitudes to evaluate prior for.
    dustmap : object
        Dust map object with a ``query(coord)`` method. For 3D dust maps
        (e.g., ``Bayestar``), ``query`` returns ``(distances, av_mean,
        av_std)`` with distance-resolved profiles. For simpler maps,
        ``query`` returns ``(av_mean, av_std)`` scalars.
    coord : SkyCoord or array_like
        Sky coordinates for dust map query. Accepts
        ``astropy.coordinates.SkyCoord`` or ``[l, b]`` in degrees.
    distance : float or array_like, optional
        Distance(s) in kpc at which to evaluate the extinction prior.
        Required for 3D dust maps to interpolate the extinction profile.
        If ``avs`` and ``distance`` are both arrays, they must have the
        same shape and the prior is evaluated element-wise.
    return_components : bool, optional
        If True, returns tuple ``(logp, (av_mean, av_err))`` including
        the dust map statistics used. Default is False.

    Returns
    -------
    logp : ndarray
        Log-prior probability density for the input extinction values.
        Returns 0 (uniform prior) when no dust map coverage is available.
    components : tuple, optional
        If ``return_components=True``, returns ``(av_mean, av_err)``
        containing the dust map mean and standard deviation used.

    Notes
    -----
    The log-prior follows a Gaussian distribution when dust map data
    is available:

    .. math::
        \\log p(A_V | A_{V,\\text{map}}, \\sigma_{A_V}) =
        -\\frac{1}{2} \\left[ \\frac{(A_V - A_{V,\\text{map}})^2}{\\sigma_{A_V}^2} +
        \\log(2\\pi\\sigma_{A_V}^2) \\right]

    For 3D dust maps, the expected extinction and uncertainty at the
    requested distance are obtained by linear interpolation of the
    map's distance-resolved profiles. Distances outside the map range
    use the boundary values.

    For regions without dust map coverage (NaN values), a uniform
    (uninformative) prior is returned.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> coord = SkyCoord(l=90., b=0., unit='deg', frame='galactic')
    >>> # With a 3D dust map:
    >>> # logp = logp_extinction([0.1, 0.5], dustmap, coord, distance=1.0)
    >>> # logp, (mean, err) = logp_extinction([0.1], dustmap, coord,
    >>> #                                      distance=1.0,
    >>> #                                      return_components=True)
    """
    avs = np.asarray(avs, dtype=float)

    # Query the dust map
    try:
        result = dustmap.query(coord)
    except (AttributeError, TypeError):
        # dustmap doesn't have a query method (e.g., was passed as string)
        lnprior = np.zeros_like(avs, dtype=float)
        if return_components:
            return lnprior, (np.nan, np.nan)
        return lnprior

    # Handle 3D dust maps returning (distances, av_profile, std_profile)
    if isinstance(result, tuple) and len(result) == 3:
        map_distances, av_profile, std_profile = result

        # Squeeze single-coordinate queries that return (1, n_dist) arrays
        av_profile = np.squeeze(av_profile)
        std_profile = np.squeeze(std_profile)

        if distance is None:
            # Cannot evaluate distance-dependent prior without distance
            lnprior = np.zeros_like(avs, dtype=float)
            av_mean = np.nan
            av_err = np.nan
        else:
            distance = np.asarray(distance, dtype=float)

            # Interpolate dust map profiles to requested distance(s)
            av_mean = np.interp(distance, map_distances, av_profile)
            av_err = np.interp(distance, map_distances, std_profile)

            # Compute Gaussian prior where valid
            valid = np.isfinite(av_mean) & np.isfinite(av_err) & (av_err > 0)

            lnprior = np.zeros_like(avs, dtype=float)
            if np.any(valid):
                # Use safe denominator to avoid division by zero
                av_err_safe = np.where(valid, av_err, 1.0)
                chi2 = (avs - av_mean) ** 2 / av_err_safe**2
                lnorm = np.log(2.0 * np.pi * av_err_safe**2)
                lnprior = np.where(valid, -0.5 * (chi2 + lnorm), 0.0)

    # Handle simple dust maps returning (av_mean, av_std)
    elif isinstance(result, tuple) and len(result) == 2:
        av_mean, av_err = result

        if np.isfinite(av_mean) and np.isfinite(av_err) and av_err > 0:
            chi2 = (avs - av_mean) ** 2 / av_err**2
            lnorm = np.log(2.0 * np.pi * av_err**2)
            lnprior = -0.5 * (chi2 + lnorm)
        else:
            lnprior = np.zeros_like(avs, dtype=float)
            av_mean, av_err = np.nan, np.nan

    else:
        # Unrecognized return format
        lnprior = np.zeros_like(avs, dtype=float)
        av_mean = np.nan
        av_err = np.nan

    if return_components:
        return lnprior, (av_mean, av_err)
    return lnprior
