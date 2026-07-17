#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distance and reddening visualization functions.

This module provides functions for plotting distance vs reddening
posterior distributions.
"""

import numpy as np
from matplotlib import pyplot as plt

from .binning import bin_pdfs_distred

__all__ = ["dist_vs_red"]


def dist_vs_red(
    data,
    ebv=None,
    dist_type="distance_modulus",
    lndistprior=None,
    coord=None,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    weights=None,
    parallax=None,
    parallax_err=None,
    Nr=300,
    cmap="Blues",
    bins=300,
    span=None,
    smooth=0.015,
    plot_kwargs=None,
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    rstate=None,
):
    """
    Generate a 2-D plot of distance vs reddening.

    Parameters
    ----------
    data : 3-tuple or 4-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of
        `(dists, reds, dreds)` that were saved, or a collection of
        `(scales, avs, rvs, covs_sar)` that will be used to regenerate
        `(dists, reds)` in conjunction with any applied distance
        and/or parallax priors. Arrays of shape `(Nobj, Nsamps)` are also
        accepted; the plotted histogram is then the average of the
        per-object 2-D PDFs.

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
        `lndistprior`.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The Rv limits used to truncate results. Default is `(1., 8.)`.

    weights : `~numpy.ndarray` of shape `(Nsamps)` or `(Nobj, Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    parallax : float, optional
        The parallax estimate for the source.

    parallax_err : float, optional
        The parallax error.

    Nr : int, optional
        The number of Monte Carlo realizations used when sampling using the
        provided parallax prior. Default is `300`.

    cmap : str, optional
        The colormap used when plotting. Default is `'Blues'`.

    bins : int or list of ints with length `(ndim,)`, optional
        The number of bins to be used in each dimension. Default is `300`.

    span : iterable with shape `(2, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds. If not provided, the x-axis will use the
        provided Av bounds while the y-axis will span `(4., 19.)` in
        distance modulus (both appropriately transformed).

    smooth : int/float or list of ints/floats with shape `(ndim,)`, optional
        The standard deviation (either a single value or a different value for
        each axis) for the Gaussian kernel used to smooth the 2-D
        marginalized posteriors. If an int is passed, the smoothing will
        be applied in units of the binning in that dimension. If a float
        is passed, it is expressed as a fraction of the span.
        Default is `0.015` (1.5% smoothing).
        **Cannot smooth by more than the provided parallax will allow.**

    plot_kwargs : dict, optional
        Extra keyword arguments to be used when plotting the smoothed
        2-D histograms.

    truths : iterable with shape `(ndim,)`, optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.

    truth_color : str or iterable with shape `(ndim,)`, optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    hist2d : (counts, xedges, yedges, `~matplotlib.figure.Image`)
        Output 2-D histogram.

    """

    # Initialize values.
    if truth_kwargs is None:
        truth_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()

    # Set defaults for truth plotting
    truth_kwargs["linestyle"] = truth_kwargs.get("linestyle", "solid")
    truth_kwargs["linewidth"] = truth_kwargs.get("linewidth", 2)
    truth_kwargs["alpha"] = truth_kwargs.get("alpha", 0.7)

    # Handle single object case - convert to array format expected by bin_pdfs_distred
    # bin_pdfs_distred expects (n_objects, n_samples) shape
    if len(data[0].shape) == 1:
        # Single object case: convert from (n_samples,) to (1, n_samples)
        if len(data) == 3:  # (dists, reds, dreds)
            data = tuple(arr[None, :] for arr in data)  # Add object dimension
        elif len(data) == 4:  # (scales, avs, rvs, covs_sar)
            data = (
                data[0][None, :],
                data[1][None, :],
                data[2][None, :],
                data[3][None, :],
            )
        single_object = True

        # Convert coord to list format
        if coord is not None:
            coord = [coord]

        # Convert weights to (Nobj, Nsamps) format
        if weights is not None:
            weights = np.atleast_2d(np.asarray(weights))

        # Convert parallax info to array format
        if parallax is not None:
            parallax = np.array([parallax])
        if parallax_err is not None:
            parallax_err = np.array([parallax_err])
    else:
        # Multi-object case
        single_object = False

        # A single set of per-sample weights is shared across all objects;
        # broadcast it to the (Nobj, Nsamps) shape bin_pdfs_distred expects.
        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim == 1:
                nobj, nsamps = data[0].shape
                if weights.shape[0] != nsamps:
                    raise ValueError(
                        f"`weights` has shape {weights.shape}, which matches "
                        f"neither (Nsamps,) = ({nsamps},) nor "
                        f"(Nobj, Nsamps) = {(nobj, nsamps)}."
                    )
                weights = np.broadcast_to(weights, (nobj, nsamps))

    # Use bin_pdfs_distred to do all the heavy lifting for data preparation
    binned_vals, xedges, yedges = bin_pdfs_distred(
        data,
        cdf=False,
        ebv=ebv,
        dist_type=dist_type,
        lndistprior=lndistprior,
        coord=coord,
        avlim=avlim,
        rvlim=rvlim,
        weights=weights,
        parallaxes=parallax,
        parallax_errors=parallax_err,
        Nr=Nr,
        bins=bins,
        span=span,
        smooth=smooth,
        rstate=rstate,
        verbose=False,
    )

    # For single object, extract the first (and only) object's data
    if single_object:
        H = binned_vals[0]
    else:
        # For multiple objects, stack (average) the per-object 2-D PDFs so the
        # plot reflects the whole sample, as in dust-mapping sightline stacks.
        H = binned_vals.mean(axis=0)

    # Set up axis labels
    if ebv:
        ylabel = r"$E(B-V)$ [mag]"
    else:
        ylabel = r"$A_v$ [mag]"

    if dist_type == "scale":
        xlabel = r"$s$"
    elif dist_type == "parallax":
        xlabel = r"$\pi$ [mas]"
    elif dist_type == "distance":
        xlabel = r"$d$ [kpc]"
    elif dist_type == "distance_modulus":
        xlabel = r"$\mu$"

    # Determine plot extent
    xlims = [xedges[0], xedges[-1]]
    ylims = [yedges[0], yedges[-1]]

    # Generate the plot
    img = plt.imshow(
        H.T,
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
        **plot_kwargs,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add truth values if provided
    if truths is not None:
        if truths[0] is not None:  # x-axis truth
            try:
                [plt.axvline(t, color=truth_color, **truth_kwargs) for t in truths[0]]
            except TypeError:
                plt.axvline(truths[0], color=truth_color, **truth_kwargs)
        if truths[1] is not None:  # y-axis truth
            try:
                [plt.axhline(t, color=truth_color, **truth_kwargs) for t in truths[1]]
            except TypeError:
                plt.axhline(truths[1], color=truth_color, **truth_kwargs)

    return H, xedges, yedges, img
