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
from math import log, pi

import numpy as np
from numba import jit, prange

from ..utils.math import AVLIM, _truncnorm_lognorm, _truncnorm_lognorm_scalar

__all__ = ["logp_extinction"]


def _log_truncnorm_pdf(avs, av_mean, av_err, av_lo, av_hi):
    """Gaussian log-density of ``avs`` truncated to [av_lo, av_hi]."""
    chi2 = (avs - av_mean) ** 2 / av_err**2
    lnorm = np.log(2.0 * np.pi * av_err**2)
    return -0.5 * (chi2 + lnorm) - _truncnorm_lognorm(
        (av_lo - av_mean) / av_err, (av_hi - av_mean) / av_err
    )


def _add_scatter(std, scatter):
    """Add ``scatter`` in quadrature to non-negative spreads.

    Negative spreads are invalid map values and are returned as NaN so they
    keep falling back to the uniform prior rather than being "repaired" by
    the squaring.
    """
    std = np.asarray(std, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.where(std >= 0, np.sqrt(std**2 + scatter**2), np.nan)
    return float(out) if out.ndim == 0 else out


@jit(nopython=True, parallel=True, cache=True)
def _extinction_prior_3d_fused(
    avs,
    distance,
    map_distances,
    av_profile,
    std_profile,
    av_lo,
    av_hi,
    lnp_uniform,
):
    """Fused 3D dust prior: one binary search + Gaussian log-pdf per point.

    Interpolates the map's mean/std extinction profiles at ``distance[i]``
    (np.interp semantics: clamped at the profile boundaries) and returns
    the truncated-normal log-prior for ``avs[i]``. Points where the
    interpolated profile is non-finite or has non-positive spread get
    ``lnp_uniform`` (the uniform prior over the A(V) support), matching the
    numpy fallback path.
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
            out[i] = lnp - _truncnorm_lognorm_scalar(
                (av_lo - mu) / sig, (av_hi - mu) / sig
            )
        else:
            out[i] = lnp_uniform

    return out


def logp_extinction(
    avs,
    dustmap,
    coord,
    distance=None,
    return_components=False,
    avlim=AVLIM,
    scatter=0.2,
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
    avlim : tuple of 2 floats, optional
        Support ``(av_min, av_max)`` over which the Gaussian prior is
        normalized (truncated normal), and over which the uniform prior of
        uninformative map bins is normalized. Should match the ``avlim``
        of the fit. Default is ``AVLIM``, (0, 20) (the ``BruteForce.fit``
        default).
    scatter : float, optional
        Systematic scatter (in magnitudes) added in quadrature to the map's
        A(V) spread, ``sigma = sqrt(sigma_map**2 + scatter**2)``. Bayestar's
        per-bin spreads are typically only ~0.01-0.05 mag and do not include
        sub-pixel structure or map systematics; without this floor the prior
        pins A(V) (and, through the A(V)-distance profile, the distance)
        far more tightly than the map supports. Default is ``0.2``, matching
        brutus <= 0.8.3. Set to ``0`` to use the raw map spreads.

    Returns
    -------
    logp : ndarray
        Log-prior probability density for the input extinction values.
        Where the map is uninformative (no coverage, or distance bins masked
        as unreliable), returns the normalized uniform density over
        ``avlim``, ``-ln(avlim[1] - avlim[0])``.
    components : tuple, optional
        If ``return_components=True``, returns ``(av_mean, av_err)``
        containing the dust map mean and standard deviation used.

    Raises
    ------
    ValueError
        If ``avlim`` is not two finite numbers with ``avlim[0] < avlim[1]``,
        or if a 3D dust map is given without ``distance``.
    TypeError
        If ``dustmap`` has no ``query`` method (e.g. a ``str`` or
        ``pathlib.Path`` that was never loaded into a dust map object), or
        if ``dustmap.query`` returns anything but a tuple, list or array of
        length 2 or 3.

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

    The spread :math:`\\sigma_{A_V}` is the map spread with ``scatter``
    added in quadrature.

    For regions without dust map coverage (NaN values, including distance
    bins masked by ``Bayestar``'s reliability mask), a uniform
    (uninformative) prior over ``avlim`` is returned. It is normalized like
    the Gaussian: an unnormalized ``ln p = 0`` would act as a density of
    1 per mag and up-weight masked distances relative to reliable ones by a
    unit-dependent factor. If the dust map ``query`` call itself fails, the
    uniform prior is returned with a RuntimeWarning.

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

    try:
        av_lo, av_hi = (float(v) for v in avlim)
    except (TypeError, ValueError):
        raise ValueError(f"avlim must be (av_min, av_max); got {avlim!r}") from None
    if not (np.isfinite(av_lo) and np.isfinite(av_hi)) or av_lo >= av_hi:
        raise ValueError(f"avlim must be finite with avlim[0] < avlim[1]; got {avlim}")
    # The uniform prior of uninformative bins, normalized like the Gaussian.
    lnp_unif = -log(av_hi - av_lo)

    scatter = float(scatter)
    if not (np.isfinite(scatter) and scatter >= 0):
        raise ValueError(f"scatter must be finite and non-negative; got {scatter}")

    # A dust map must expose query(); silently returning a uniform prior for
    # e.g. a pathlib.Path would disable the dust prior for the whole fit.
    if not hasattr(dustmap, "query"):
        raise TypeError(
            f"`dustmap` has no query() method (got "
            f"{type(dustmap).__name__!r}). Pass a dust map object such as "
            f"brutus.dust.Bayestar; file paths must be loaded first "
            f"(`str` and `os.PathLike` paths are auto-converted inside "
            f"BruteForce.fit)."
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
        lnprior = np.full_like(avs, lnp_unif, dtype=float)
        if return_components:
            return lnprior, (np.nan, np.nan)
        return lnprior

    if not (isinstance(result, (tuple, list, np.ndarray)) and len(result) in (2, 3)):
        raise TypeError(
            "dustmap.query must return (av_mean, av_std) or (distances, "
            f"av_mean, av_std); got {result!r:.80}"
        )

    # Handle 3D dust maps returning (distances, av_profile, std_profile)
    if len(result) == 3:
        if distance is None:
            raise ValueError(
                "distance is required with a 3D dust map: the prior is "
                "interpolated along the map's distance profile"
            )
        map_distances, av_profile, std_profile = result

        # Squeeze single-coordinate queries that return (1, n_dist) arrays
        av_profile = np.squeeze(av_profile)
        std_profile = np.squeeze(std_profile)

        # Systematic scatter in quadrature, applied to the profile so the
        # fused and numpy paths agree. NaN (masked) and negative (invalid)
        # spreads stay invalid; zero spreads become proper Gaussians.
        std_profile = _add_scatter(std_profile, scatter)

        distance = np.asarray(distance, dtype=float)

        # Fast path: fused numba kernel (one binary search per point, no
        # full-size temporaries) for the large elementwise-array case
        # used by BruteForce (Nmc * Nsel points per object). The numpy
        # branch below handles the other cases.
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
            return _extinction_prior_3d_fused(
                avs,
                distance,
                map_d,
                av_p,
                std_p,
                av_lo,
                av_hi,
                lnp_unif,
            )

        # Interpolate dust map profiles to requested distance(s)
        av_mean = np.interp(distance, map_distances, av_profile)
        av_err = np.interp(distance, map_distances, std_profile)

        # Compute Gaussian prior where valid
        valid = np.isfinite(av_mean) & np.isfinite(av_err) & (av_err > 0)

        lnprior = np.full_like(avs, lnp_unif, dtype=float)
        if np.any(valid):
            # Use safe denominator to avoid division by zero
            av_err_safe = np.where(valid, av_err, 1.0)
            lnprior = _log_truncnorm_pdf(avs, av_mean, av_err_safe, av_lo, av_hi)
            lnprior = np.where(valid, lnprior, lnp_unif)

    # Handle simple dust maps returning (av_mean, av_std)
    else:
        av_mean, av_err = result
        av_err = _add_scatter(av_err, scatter)

        if np.isfinite(av_mean) and np.isfinite(av_err) and av_err > 0:
            lnprior = _log_truncnorm_pdf(avs, av_mean, av_err, av_lo, av_hi)
        else:
            lnprior = np.full_like(avs, lnp_unif, dtype=float)
            av_mean, av_err = np.nan, np.nan

    if return_components:
        return lnprior, (av_mean, av_err)
    return lnprior
