#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data binning utilities for plotting posterior distributions.

This module provides functions to bin posterior samples of distance and
reddening for visualization purposes.
"""

import sys
import warnings
from functools import partial

import numpy as np
from scipy.ndimage import gaussian_filter as norm_kde

from ..priors import logp_galactic_structure as gal_lnprior
from ..utils.math import AVLIM, R_SOLAR, RVLIM, Z_SOLAR
from .utils import _require_coord, _sar_draws

__all__ = ["bin_pdfs_distred"]


def bin_pdfs_distred(
    data,
    cdf=False,
    ebv=False,
    dist_type="distance_modulus",
    lndistprior=None,
    coord=None,
    avlim=AVLIM,
    rvlim=RVLIM,
    weights=None,
    parallaxes=None,
    parallax_errors=None,
    Nr=100,
    bins=(750, 300),
    span=None,
    smooth=(0.01, 0.005),
    rstate=None,
    verbose=False,
    R_solar=R_SOLAR,
    Z_solar=Z_SOLAR,
):
    """
    Generate binned versions of the 2-D posteriors for the distance and
    reddening.

    Parameters
    ----------
    data : 3-tuple or 4-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of
        `(dists, reds, dreds)` that were saved, or a collection of
        `(scales, avs, rvs, covs_sar)` that will be used to regenerate
        `(dists, reds)` in conjunction with any applied distance
        and/or parallax priors. Prefer the saved draws: regeneration applies
        only the distance and parallax priors, not the dust-map and
        label-dependent priors of the fit.

    cdf : bool, optional
        Whether to compute the CDF along the reddening axis instead of the
        PDF. Useful when evaluating the MAP LOS fit. Default is `False`.

    ebv : bool, optional
        If provided, will convert from Av to E(B-V) when plotting using
        the provided Rv values. Default is `False`.

    dist_type : str, optional
        The distance format to be plotted. Options include `'parallax'`,
        `'scale'`, `'distance'`, and `'distance_modulus'`.
        Default is `'distance_modulus`.

    lndistprior : func, optional
        The log-distance prior function used. If not provided,
        `~brutus.priors.logp_galactic_structure` is assumed.

    coord : 2-tuple or `~numpy.ndarray` of shape `(Nobj, 2)`, optional
        The galactic `(l, b)` coordinates passed to `lndistprior` when
        re-generating the fits. A single `(l, b)` pair is shared across
        all objects; otherwise provide one pair per object.

    avlim : 2-tuple, optional
        The Av limits used to truncate regenerated draws. Without `span`,
        they also set the A(V) range of the bins and of the smoothing.
        Default is `AVLIM`, (0, 20), the `BruteForce.fit` default.

    rvlim : 2-tuple, optional
        The Rv limits used to truncate results. Default is `RVLIM`, (1, 8).

    weights : `~numpy.ndarray` of shape `(Nobj, Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.
        Must be finite and non-negative, with at least one positive weight
        per object. Each object's binned PDF is normalized by its total
        weight, so results are invariant to the absolute scale of `weights`.

    parallaxes : float or `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax estimates for the sources in mas. A float applies to
        every object.

    parallax_errors : float or `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax errors for the sources in mas. A float applies to
        every object.

    Nr : int, optional
        The number of Monte Carlo realizations used when sampling using the
        provided parallax prior. Default is `100`.

    bins : int or list of ints with length `(ndim,)`, optional
        The number of bins to be used in each dimension. Default is
        `(750, 300)` (distance, reddening).

    span : iterable with shape `(2, 2)`, optional
        `((av_min, av_max), (d_min, d_max))`: the A(V) range of the y-axis
        in mag and the distance range of the x-axis in kpc, converted to
        `dist_type`. Note the order: `span` starts with A(V), `bins` and
        `smooth` start with distance. If not provided, the y-axis spans
        `avlim` and the x-axis spans distance modulus 4 to 19 (0.063 to
        63.1 kpc).

    smooth : float or 2-tuple of floats, optional
        The standard deviation of the Gaussian kernel that smooths the 2-D
        PDFs, as `(distance, A(V))` or one value for both axes. A value
        below 1 is a fraction of that axis range, and a value of 1 or more
        is a number of bins. Without `span`, the A(V) range is `avlim`, so
        with the default `avlim` an A(V) value of 0.005 is 0.1 mag (of
        E(B-V) with `ebv=True`, since the axis keeps the `avlim` range).
        Default is `(0.01, 0.005)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    verbose : bool, optional
        Whether to print progress to `~sys.stderr`. Default is `False`.

    Returns
    -------
    binned_vals : `~numpy.ndarray` of shape `(Nobj, Nxbin, Nybin)`
        Binned versions of the PDFs or CDFs.

    xedges : `~numpy.ndarray` of shape `(Nxbin+1,)`
        The edges defining the bins in distance.

    yedges : `~numpy.ndarray` of shape `(Nybin+1,)`
        The edges defining the bins in reddening.

    """

    # Initialize values.
    nobjs, nsamps = data[0].shape
    if rstate is None:
        rstate = np.random
    if lndistprior is None:
        lndistprior = partial(gal_lnprior, R_solar=R_solar, Z_solar=Z_solar)
    parallaxes = _per_object(parallaxes, "parallaxes", nobjs)
    parallax_errors = _per_object(parallax_errors, "parallax_errors", nobjs)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != (nobjs, nsamps):
            raise ValueError(
                f"`weights` must have shape {(nobjs, nsamps)}; " f"got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("`weights` must be finite and non-negative.")
        wtots = weights.sum(axis=1)
        if np.any(wtots == 0):
            bad = np.where(wtots == 0)[0]
            raise ValueError(
                f"`weights` sum to zero for object(s) {bad}; each object "
                "needs at least one positive weight."
            )

    # Set up bins.
    if dist_type not in ["parallax", "scale", "distance", "distance_modulus"]:
        raise ValueError("The provided `dist_type` is not valid.")
    if span is None:
        avlims = avlim
        dlims = 10 ** (np.array([4.0, 19.0]) / 5.0 - 2.0)
    else:
        span = np.asarray(span, dtype=float)
        if span.shape != (2, 2):
            raise ValueError(
                "`span` must be ((av_min, av_max), (d_min, d_max)); "
                f"got shape {span.shape}."
            )
        avlims, dlims = span
    try:
        xbin, ybin = bins
    except (TypeError, ValueError):
        xbin = ybin = bins
    ylims = avlims
    if dist_type == "scale":
        xlims = (1.0 / dlims[::-1]) ** 2
    elif dist_type == "parallax":
        xlims = 1.0 / dlims[::-1]
    elif dist_type == "distance":
        xlims = dlims
    elif dist_type == "distance_modulus":
        xlims = 5.0 * np.log10(dlims) + 10.0
    xbins = np.linspace(xlims[0], xlims[1], xbin + 1)
    ybins = np.linspace(ylims[0], ylims[1], ybin + 1)
    dx, dy = xbins[1] - xbins[0], ybins[1] - ybins[0]
    xspan, yspan = xlims[1] - xlims[0], ylims[1] - ylims[0]

    # Set smoothing.
    try:
        if smooth[0] < 1:
            xsmooth = smooth[0] * xspan
        else:
            xsmooth = smooth[0] * dx
        if smooth[1] < 1:
            ysmooth = smooth[1] * yspan
        else:
            ysmooth = smooth[1] * dy
    except (TypeError, IndexError):
        if smooth < 1:
            xsmooth, ysmooth = smooth * xspan, smooth * yspan
        else:
            xsmooth, ysmooth = smooth * dx, smooth * dy

    # Compute binned PDFs.
    binned_vals = np.zeros((nobjs, xbin, ybin), dtype="float32")
    if len(data) == 3:
        # Saved (distance, A(V), R(V)) samples.
        ddraws, adraws, rdraws = data
        pdraws = 1.0 / ddraws
        xdraws = _distance_draws(dist_type, pdraws**2, pdraws, ddraws)
        ydraws = adraws / rdraws if ebv else adraws

        # Bin draws.
        for i, (xs, ys) in enumerate(zip(xdraws, ydraws)):
            # Print progress.
            if verbose:
                sys.stderr.write(f"\rBinning object {i + 1}/{nobjs}")
            wt = None if weights is None else weights[i]
            H = np.histogram2d(xs, ys, bins=(xbins, ybins), weights=wt)[0]
            # Normalize by the total weight (nsamps when unweighted) so the
            # binned mass equals the weighted fraction of samples inside the
            # span, independent of the absolute scale of `weights`.
            wtot = nsamps if weights is None else weights[i].sum()
            binned_vals[i] = H / wtot
    elif len(data) == 4:
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = data
        _require_coord(lndistprior, coord)
        if coord is None:
            # Custom `lndistprior` without coordinates: pass `None` per object.
            coord = [None] * nobjs
        else:
            # A bare (l, b) pair is shared by every object (e.g. a stack of
            # stars along one sightline). Anything else must supply one pair
            # per object -- a mis-shaped coord would otherwise be silently
            # mis-paired (and truncated) by the zip below.
            coord = np.asarray(coord)
            if coord.shape == (2,):
                coord = np.tile(coord, (nobjs, 1))
            elif coord.shape != (nobjs, 2):
                raise ValueError(
                    f"`coord` must be a single (l, b) pair or have shape "
                    f"{(nobjs, 2)}; got {coord.shape}."
                )

        # Generate parallax and Av realizations.
        for i in range(nobjs):
            # Print progress.
            if verbose:
                sys.stderr.write(f"\rBinning object {i + 1}/{nobjs}")

            sdraws, pdraws, ddraws, adraws, rdraws, pwt = _sar_draws(
                scales[i],
                avs[i],
                rvs[i],
                covs_sar[i],
                lndistprior,
                coord[i],
                parallaxes[i],
                parallax_errors[i],
                Nr,
                avlim,
                rvlim,
                rstate,
            )
            if weights is not None:
                # Fold per-sample importance weights into the per-draw ones.
                pwt *= weights[i][:, None]
            xdraws = _distance_draws(dist_type, sdraws, pdraws, ddraws)
            ydraws = adraws / rdraws if ebv else adraws

            # Generate 2-D histogram.
            H = np.histogram2d(
                xdraws.ravel(), ydraws.ravel(), bins=(xbins, ybins), weights=pwt.ravel()
            )[0]
            # Each sample's Nr draw weights sum to 1, so the total weight is
            # nsamps unweighted and sum(weights[i]) otherwise; dividing by it
            # keeps the result invariant to the absolute scale of `weights`.
            wtot = nsamps if weights is None else weights[i].sum()
            binned_vals[i] = H / wtot
    else:
        raise ValueError(
            "`data` must be (dists, reds, dreds) or (scales, avs, rvs, covs_sar); "
            f"got {len(data)} arrays."
        )

    # Apply smoothing.
    for i, (H, parallax, parallax_err) in enumerate(
        zip(binned_vals, parallaxes, parallax_errors)
    ):
        # Establish minimum smoothing in distance.
        p1sig = np.array([parallax + parallax_err, max(parallax - parallax_err, 1e-10)])
        if dist_type == "scale":
            x_min_smooth = abs(np.diff(p1sig**2)) / 2.0
        elif dist_type == "parallax":
            x_min_smooth = abs(np.diff(p1sig)) / 2.0
        elif dist_type == "distance":
            x_min_smooth = abs(np.diff(1.0 / p1sig)) / 2.0
        elif dist_type == "distance_modulus":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore bad values
                x_min_smooth = abs(np.diff(5.0 * np.log10(1.0 / p1sig))) / 2.0
        if np.isfinite(x_min_smooth):
            xsmooth_t = min(x_min_smooth, xsmooth)
        else:
            xsmooth_t = xsmooth
        try:
            xsmooth_t = xsmooth_t[0]  # catch possible list
        except (TypeError, IndexError):
            pass
        # Smooth 2-D PDF.
        binned_vals[i] = norm_kde(H, (xsmooth_t / dx, ysmooth / dy))

    # Compute CDFs.
    if cdf:
        for i, H in enumerate(binned_vals):
            # H has axis 0 = distance, axis 1 = reddening; the documented CDF
            # (used to evaluate MAP LOS reddening profiles) accumulates along
            # the *reddening* axis within each distance column.
            binned_vals[i] = H.cumsum(axis=1)

    return binned_vals, xbins, ybins


def _distance_draws(dist_type, sdraws, pdraws, ddraws):
    """Return the draws of the distance axis in `dist_type` units."""
    if dist_type == "scale":
        return sdraws
    if dist_type == "parallax":
        return pdraws
    if dist_type == "distance":
        return ddraws
    return 5.0 * np.log10(ddraws) + 10.0


def _per_object(value, name, nobjs):
    """Return `value` as a float array of shape `(nobjs,)`, NaN for `None`."""
    value = np.asarray(np.nan if value is None else value, dtype=float)
    if value.ndim == 0:
        return np.full(nobjs, value)
    if value.shape != (nobjs,):
        raise ValueError(
            f"`{name}` must be a float or have shape ({nobjs},); got {value.shape}."
        )
    return value
