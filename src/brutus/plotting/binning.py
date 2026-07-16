#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data binning utilities for plotting posterior distributions.

This module provides functions to bin posterior samples of distance and
reddening for visualization purposes.
"""

import copy
import sys
import warnings
from functools import partial

import numpy as np
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.special import logsumexp

from ..priors import logp_galactic_structure as gal_lnprior
from ..priors import logp_parallax
from ..utils.sampling import draw_sar

__all__ = ["bin_pdfs_distred"]


def bin_pdfs_distred(
    data,
    cdf=False,
    ebv=False,
    dist_type="distance_modulus",
    lndistprior=None,
    coord=None,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    weights=None,
    parallaxes=None,
    parallax_errors=None,
    Nr=100,
    bins=(750, 300),
    span=None,
    smooth=0.01,
    rstate=None,
    verbose=False,
    R_solar=8.2,
    Z_solar=0.025,
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
        and/or parallax priors.

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
        The log-distsance prior function used. If not provided, the galactic
        model from Green et al. (2014) will be assumed.

    coord : 2-tuple, optional
        The galactic `(l, b)` coordinates for the object, which is passed to
        `lndistprior` when re-generating the fits.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The Rv limits used to truncate results. Default is `(1., 8.)`.

    weights : `~numpy.ndarray` of shape `(Nobj, Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    parallaxes : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax estimates for the sources.

    parallax_errors : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax errors for the sources.

    Nr : int, optional
        The number of Monte Carlo realizations used when sampling using the
        provided parallax prior. Default is `100`.

    bins : int or list of ints with length `(ndim,)`, optional
        The number of bins to be used in each dimension. Default is
        `(750, 300)` (distance, reddening).

    span : iterable with shape `(ndim, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds. If not provided, the x-axis will use the
        provided Av bounds while the y-axis will span `(4., 19.)` in
        distance modulus (both appropriately transformed).

    smooth : float or list of floats with shape `(ndim,)`, optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.01` (1% smoothing).

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
    if parallaxes is None:
        parallaxes = np.full(nobjs, np.nan)
    if parallax_errors is None:
        parallax_errors = np.full(nobjs, np.nan)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != (nobjs, nsamps):
            raise ValueError(
                f"`weights` must have shape {(nobjs, nsamps)}; " f"got {weights.shape}."
            )

    # Set up bins.
    if dist_type not in ["parallax", "scale", "distance", "distance_modulus"]:
        raise ValueError("The provided `dist_type` is not valid.")
    if span is None:
        avlims = avlim
        dlims = 10 ** (np.array([4.0, 19.0]) / 5.0 - 2.0)
    else:
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
    try:
        # Grab (distance, reddening (Av), differential reddening (Rv)) samples.
        # Check if data is in direct format (3 values) vs SAR format (4 values)
        if len(data) == 3:
            ddraws, adraws, rdraws = copy.deepcopy(data)
        else:
            # Data is in SAR format - raise to trigger except block
            raise AttributeError("Data is in SAR format")
        pdraws = 1.0 / ddraws
        sdraws = pdraws**2
        dmdraws = 5.0 * np.log10(ddraws) + 10.0

        # Grab relevant draws.
        ydraws = adraws
        if ebv:
            ydraws /= rdraws
        if dist_type == "scale":
            xdraws = sdraws
        elif dist_type == "parallax":
            xdraws = pdraws
        elif dist_type == "distance":
            xdraws = ddraws
        elif dist_type == "distance_modulus":
            xdraws = dmdraws

        # Bin draws.
        for i, (xs, ys) in enumerate(zip(xdraws, ydraws)):
            # Print progress.
            if verbose:
                sys.stderr.write(f"\rBinning object {i + 1}/{nobjs}")
            wt = None if weights is None else weights[i]
            H, xedges, yedges = np.histogram2d(xs, ys, bins=(xbins, ybins), weights=wt)
            binned_vals[i] = H / nsamps
    except (AttributeError, KeyError):
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = copy.deepcopy(data)

        _is_default_prior = lndistprior is gal_lnprior or (
            hasattr(lndistprior, "func") and lndistprior.func is gal_lnprior
        )
        if _is_default_prior and coord is None:
            raise ValueError(
                "`coord` must be passed if the default distance " "prior was used."
            )
        if coord is None:
            # Custom `lndistprior` without coordinates: pass `None` per object.
            coord = [None] * nobjs

        # Generate parallax and Av realizations.
        for i, stuff in enumerate(
            zip(scales, avs, rvs, covs_sar, parallaxes, parallax_errors, coord)
        ):
            (
                scales_obj,
                avs_obj,
                rvs_obj,
                covs_sar_obj,
                parallax,
                parallax_err,
                crd,
            ) = stuff

            # Print progress.
            if verbose:
                sys.stderr.write(f"\rBinning object {i + 1}/{nobjs}")

            # Draw random samples.
            sdraws, adraws, rdraws = draw_sar(
                scales_obj,
                avs_obj,
                rvs_obj,
                covs_sar_obj,
                ndraws=Nr,
                avlim=avlim,
                rvlim=rvlim,
                rstate=rstate,
            )
            pdraws = np.sqrt(sdraws)
            ddraws = 1.0 / pdraws
            dmdraws = 5.0 * np.log10(ddraws) + 10.0

            # Re-apply distance and parallax priors to realizations.
            # Draws come from a Gaussian proposal in *scale* space (s = d^-2),
            # while `lndistprior` is a density over distance, so the
            # importance weight needs the change-of-variables Jacobian
            # |dd/ds| = d^3 / 2 (constant factor irrelevant after
            # normalization). Matches the ln-distance Jacobian handling in
            # `analysis.individual.logpost_grid`.
            lnp_draws = lndistprior(ddraws, crd) + 3.0 * np.log(ddraws)
            if parallax is not None and parallax_err is not None:
                lnp_draws += logp_parallax(pdraws, parallax, parallax_err)
            lnp = logsumexp(lnp_draws, axis=1)
            pwt = np.exp(lnp_draws - lnp[:, None])
            pwt /= pwt.sum(axis=1)[:, None]
            if weights is not None:
                # Fold per-sample importance weights into the per-draw ones.
                pwt *= weights[i][:, None]
            pwt = pwt.flatten()

            # Grab draws.
            ydraws = adraws.flatten()
            if ebv:
                ydraws /= rdraws.flatten()
            if dist_type == "scale":
                xdraws = sdraws.flatten()
            elif dist_type == "parallax":
                xdraws = pdraws.flatten()
            elif dist_type == "distance":
                xdraws = ddraws.flatten()
            elif dist_type == "distance_modulus":
                xdraws = dmdraws.flatten()

            # Generate 2-D histogram.
            H, xedges, yedges = np.histogram2d(
                xdraws, ydraws, bins=(xbins, ybins), weights=pwt
            )
            binned_vals[i] = H / nsamps

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

    return binned_vals, xedges, yedges
