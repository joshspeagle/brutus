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

import warnings
from math import erf, log, pi, sqrt

import numpy as np
from numba import jit, prange
from scipy.special import erf as _np_erf

__all__ = ["logp_extinction"]

_SQRT2 = sqrt(2.0)


def _log_truncnorm_z(av_mean, av_err, av_lo, av_hi):
    """Log of the Gaussian mass on [av_lo, av_hi] (vectorized, erf-based).

    Uses the same erf formulation as the fused numba kernel so both code
    paths agree to floating-point roundoff. The mass is floored at 1e-300
    to keep the log finite when the map mean is far outside the support.
    """
    z = 0.5 * (
        _np_erf((av_hi - av_mean) / (av_err * _SQRT2))
        - _np_erf((av_lo - av_mean) / (av_err * _SQRT2))
    )
    return np.log(np.maximum(z, 1e-300))


@jit(nopython=True, parallel=True, cache=True)
def _extinction_prior_3d_fused(
    avs, distance, map_distances, av_profile, std_profile, av_lo, av_hi, truncate
):
    """Fused 3D dust prior: one binary search + Gaussian log-pdf per point.

    Interpolates the map's mean/std extinction profiles at ``distance[i]``
    (np.interp semantics: clamped at the profile boundaries) and returns
    the truncated-normal log-prior for ``avs[i]``. Points where the
    interpolated profile is non-finite or has non-positive spread get 0
    (uniform prior), matching the numpy fallback path.
    """
    N = avs.shape[0]
    n = map_distances.shape[0]
    out = np.empty(N)

    for i in prange(N):
        d = distance[i]

        # Interpolate both profiles from one shared bracket
        if d <= map_distances[0]:
            mu = av_profile[0]
            sig = std_profile[0]
        elif d >= map_distances[n - 1]:
            mu = av_profile[n - 1]
            sig = std_profile[n - 1]
        else:
            lo = 0
            hi = n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if map_distances[mid] <= d:
                    lo = mid
                else:
                    hi = mid
            if d == map_distances[lo]:
                # np.interp returns fp[lo] exactly at grid points, even
                # when the right neighbor is NaN
                mu = av_profile[lo]
                sig = std_profile[lo]
            else:
                # Slope form matches np.interp's arithmetic exactly, so
                # the numpy fallback classifies validity (sig > 0)
                # identically
                dx = map_distances[lo + 1] - map_distances[lo]
                dd = d - map_distances[lo]
                mu = (av_profile[lo + 1] - av_profile[lo]) / dx * dd + av_profile[lo]
                sig = (std_profile[lo + 1] - std_profile[lo]) / dx * dd + std_profile[
                    lo
                ]

        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            chi2 = (avs[i] - mu) ** 2 / sig**2
            lnorm = log(2.0 * pi * sig**2)
            lnp = -0.5 * (chi2 + lnorm)
            if truncate:
                z = 0.5 * (
                    erf((av_hi - mu) / (sig * _SQRT2))
                    - erf((av_lo - mu) / (sig * _SQRT2))
                )
                if z < 1e-300:
                    z = 1e-300
                lnp -= log(z)
            out[i] = lnp
        else:
            out[i] = 0.0

    return out


def logp_extinction(
    avs,
    dustmap,
    coord,
    distance=None,
    return_components=False,
    avlim=(0.0, 20.0),
):
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
        ``query`` returns ``(av_mean, av_std)`` scalars. Objects without
        a ``query`` method (e.g. an unloaded file path) raise TypeError.
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
    avlim : tuple of 2 floats or None, optional
        Support ``(av_min, av_max)`` over which the Gaussian prior is
        normalized (truncated normal). Should match the ``avlim`` bounds
        used to clip A(V) samples in the fitting pipeline. Default is
        ``(0.0, 20.0)`` (the ``BruteForce.fit`` default). Pass ``None``
        for an untruncated Gaussian normalized over the full real line.

    Returns
    -------
    logp : ndarray
        Log-prior probability density for the input extinction values.
        Returns 0 (uniform prior) when no dust map coverage is available.
    components : tuple, optional
        If ``return_components=True``, returns ``(av_mean, av_err)``
        containing the dust map mean and standard deviation used.

    Raises
    ------
    TypeError
        If ``dustmap`` has no ``query`` method (e.g. a ``str`` or
        ``pathlib.Path`` that was never loaded into a dust map object).

    Notes
    -----
    The log-prior follows a truncated Gaussian distribution when dust map
    data is available:

    .. math::
        \\log p(A_V | A_{V,\\text{map}}, \\sigma_{A_V}) =
        -\\frac{1}{2} \\left[ \\frac{(A_V - A_{V,\\text{map}})^2}{\\sigma_{A_V}^2} +
        \\log(2\\pi\\sigma_{A_V}^2) \\right] - \\log Z

    where :math:`Z = \\Phi((A_{V,\\max}-\\mu)/\\sigma) -
    \\Phi((A_{V,\\min}-\\mu)/\\sigma)` normalizes the density over the
    physical support ``avlim``. Because the map mean and spread vary with
    distance, omitting :math:`Z` would under-weight distances where the
    profile sits near the A(V) boundary (a bias of up to ``ln 2``).

    For 3D dust maps, the expected extinction and uncertainty at the
    requested distance are obtained by linear interpolation of the
    map's distance-resolved profiles. Distances outside the map range
    use the boundary values.

    For regions without dust map coverage (NaN values), a uniform
    (uninformative) prior is returned. If the dust map ``query`` call
    itself fails, a uniform prior is returned with a RuntimeWarning.

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

    if avlim is None:
        truncate = False
        av_lo, av_hi = 0.0, 0.0  # unused
    else:
        truncate = True
        av_lo, av_hi = float(avlim[0]), float(avlim[1])
        # Reversed or non-finite bounds would silently corrupt the
        # truncation normalization (the log-mass floor turns it into a huge
        # constant offset instead of an error).
        if not (np.isfinite(av_lo) and np.isfinite(av_hi)) or av_lo >= av_hi:
            raise ValueError(
                f"avlim must be finite with avlim[0] < avlim[1]; got {avlim}"
            )

    # A dust map must expose query(); silently returning a uniform prior for
    # e.g. a pathlib.Path would disable the dust prior for the whole fit.
    if not hasattr(dustmap, "query"):
        raise TypeError(
            f"`dustmap` has no query() method (got "
            f"{type(dustmap).__name__!r}). Pass a dust map object such as "
            f"brutus.dust.Bayestar; file paths must be loaded first "
            f"(only `str` paths are auto-converted inside BruteForce.fit)."
        )

    # Query the dust map
    try:
        result = dustmap.query(coord)
    except (AttributeError, TypeError) as e:
        # Query failed (e.g. coordinate outside the supported frame). Treat
        # as no coverage, but make the degradation visible.
        warnings.warn(
            f"Dust map query failed ({e}); returning a uniform extinction " f"prior.",
            RuntimeWarning,
            stacklevel=2,
        )
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

            # Fast path: fused numba kernel (one binary search per point, no
            # full-size temporaries) for the large elementwise-array case
            # used by BruteForce (Nmc * Nsel points per object). The numpy
            # branch below remains the fallback / NUMBA_DISABLE_JIT path.
            map_d = np.ascontiguousarray(map_distances, dtype=np.float64)
            av_p = np.ascontiguousarray(av_profile, dtype=np.float64)
            std_p = np.ascontiguousarray(std_profile, dtype=np.float64)
            if (
                not return_components
                and avs.ndim == 1
                and distance.shape == avs.shape
                and avs.size > 1000
                and map_d.ndim == 1
                and av_p.shape == map_d.shape
                and std_p.shape == map_d.shape
            ):
                try:
                    return _extinction_prior_3d_fused(
                        avs, distance, map_d, av_p, std_p, av_lo, av_hi, truncate
                    )
                except Exception as e:
                    warnings.warn(
                        f"Numba fused extinction prior failed, falling back "
                        f"to numpy: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

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
                lnprior = -0.5 * (chi2 + lnorm)
                if truncate:
                    # Normalize over the physical support [av_lo, av_hi]
                    lnprior -= _log_truncnorm_z(av_mean, av_err_safe, av_lo, av_hi)
                lnprior = np.where(valid, lnprior, 0.0)

    # Handle simple dust maps returning (av_mean, av_std)
    elif isinstance(result, tuple) and len(result) == 2:
        av_mean, av_err = result

        if np.isfinite(av_mean) and np.isfinite(av_err) and av_err > 0:
            chi2 = (avs - av_mean) ** 2 / av_err**2
            lnorm = np.log(2.0 * np.pi * av_err**2)
            lnprior = -0.5 * (chi2 + lnorm)
            if truncate:
                lnprior -= _log_truncnorm_z(av_mean, av_err, av_lo, av_hi)
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
