#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometric offset visualization functions.

This module provides functions for plotting photometric offsets
in 1D and 2D formats.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.offsets import _loo_weights
from ..core.sed_utils import get_seds
from ..utils.photometry import inv_magnitude, magnitude
from ..utils.sampling import quantile


def photometric_offsets(
    phot,
    err,
    mask,
    models,
    idxs,
    reds,
    dreds,
    dists,
    x=None,
    flux=True,
    weights=None,
    bins=100,
    offset=None,
    plot_thresh=0.0,
    cmap="viridis",
    xspan=None,
    yspan=None,
    titles=None,
    xlabel=None,
    plot_kwargs=None,
    fig=None,
):
    r"""
    Plot photometric offsets (`mag_pred - mag_obs`).

    The draws of each object are weighted, band by band, with the
    leave-one-out weights of `brutus.analysis.photometric_offsets`,
    :math:`w \propto \exp(\chi^2_i / 2)` times `weights`, where
    :math:`\chi^2_i` is the flux-space chi-square of band i. They use `phot`
    and `err` as given; `offset` only shifts the plotted residuals.

    Parameters
    ----------
    phot : `~numpy.ndarray` of shape `(Nobj, Nfilt)`, optional
        Observed data values (fluxes). If provided, these will be overplotted.

    err : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        Associated errors on the data values. If provided, these will be
        overplotted as error bars.

    mask : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        Binary mask (0/1) indicating whether the data value was observed.
        If provided, these will be used to mask missing/bad data values.

    models : `~numpy.ndarray` of shape `(Nmodels, Nfilts, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    idxs : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        An array of resampled indices corresponding to the set of models used
        to fit the data.

    reds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Reddening samples (in Av) associated with the model indices.

    dreds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        "Differential" reddening samples (in Rv) associated with
        the model indices.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Distance samples (in kpc) associated with the model indices.

    x : `~numpy.ndarray` with shape `(Nobj,)` or `(Nobj, Nsamps)`, optional
        Corresponding values to be plotted on the `x` axis. If not provided,
        the default behavior is to plot as a function of observed magnitude.

    flux : bool, optional
        Whether the photometry provided is in fluxes (instead of magnitudes).
        Default is `True`.

    weights : `~numpy.ndarray` of shape `(Nobj,)` or `(Nobj, Nsamps)`, optional
        Importance weights of the objects or of their samples. Any other
        shape raises `ValueError`.

    bins : int or sequence of `Nfilt` ints, optional
        The number of bins along both axes of each panel, for all bands or
        one value per band. Any other input raises `ValueError`.
        Default is `100`.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative flux offsets. They shift the plotted residuals by
        `2.5 log10(offset)` and do not change the weights.

    plot_thresh : float, optional
        The minimum summed weight of a bin to be colored, passed to
        `~matplotlib.axes.Axes.hist2d` as `cmin`. The draws of each object
        sum to 1 in each band. Default is `0.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'viridis'`.

    xspan : iterable with shape `(nfilt, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds for the x-axis for each plot.

    yspan : iterable with shape `(nfilt, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds for the y-axis for each plot.

    titles : iterable of str of length `Nfilt`, optional
        Titles for each of the subplots corresponding to each band.
        If not provided `Band #` will be used.

    xlabel : str, optional
        Label for the x-axis of each subplot. If not provided, it is the
        title of the subplot when `x` is not given, and `'Label'` otherwise.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.axes.Axes.hist2d`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    postpredplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        The associated figure and axes for the photometric offsets.

    """

    # Initialize values.
    nmodels, nfilt, ncoeff = models.shape
    nobj, nsamps = idxs.shape
    # Coerce to boolean: an integer 0/1 mask would otherwise fancy-index
    # (selecting rows 0/1) instead of masking.
    mask = np.asarray(mask, dtype=bool)
    if plot_kwargs is None:
        plot_kwargs = dict()
    weights = _sample_weights(weights, nobj, nsamps)
    bins = np.asarray(bins)
    if bins.ndim == 0:
        bins = np.full(nfilt, bins)
    if bins.shape != (nfilt,) or not np.issubdtype(bins.dtype, np.integer):
        raise ValueError(f"`bins` must be an int or {nfilt} ints; got {bins.tolist()}.")
    if titles is None:
        titles = ["Band {0}".format(i) for i in range(nfilt)]
    if xlabel is None:
        if x is None:
            xlabel = titles
        else:
            xlabel = ["Label" for i in range(nfilt)]
    else:
        xlabel = [xlabel for i in range(nfilt)]
    if offset is None:
        offset = np.ones(nfilt)

    # Compute posterior predictive SED magnitudes.
    mpred = get_seds(models[idxs.flatten()], av=reds.flatten(), rv=dreds.flatten())
    mpred += 5.0 * np.log10(dists.flatten())[:, None]
    mpred = mpred.reshape(nobj, nsamps, nfilt)

    # Convert observed data to magnitudes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        # The leave-one-out weights use the data of the fit; `offset` only
        # shifts the plotted residuals.
        if flux:
            fobs, feobs = phot, err
            magobs = magnitude(phot * offset, err * offset)[0]
        else:
            # `offset` is a multiplicative *flux* correction; in magnitude
            # space it is applied as -2.5 log10(offset).
            fobs, feobs = inv_magnitude(phot, err)
            magobs = phot - 2.5 * np.log10(offset)

    # Generate figure.
    if fig is None:
        ncols = 5
        nrows = (nfilt - 1) // ncols + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    else:
        fig, axes = fig
        nrows, ncols = axes.shape
    ax = axes.flatten()
    # Plot offsets.
    for i in range(nfilt):
        # Require at least 4 observed bands besides band i, and finite
        # magnitudes only in *observed* bands: unobserved
        # (masked) bands routinely hold nonpositive/NaN fluxes that map to
        # non-finite magnitudes and must not disqualify the object.
        s = (
            mask[:, i]
            & (np.sum(mask, axis=1) > 4)
            & (np.all(np.isfinite(magobs) | ~mask, axis=1))
            & (weights.sum(axis=1) > 0)
        )
        ax[i].set_xlabel(xlabel[i])
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(r"$\Delta\,$mag")
        if not np.any(s):
            continue  # no object qualifies: leave the panel blank
        wt = _loo_weights(
            fobs[s, i], feobs[s, i], 10 ** (-0.4 * mpred[s, :, i]), weights[s]
        )
        wt /= wt.sum(axis=1, keepdims=True)
        # Repeat to match up with `nsamps`.
        mobs = np.repeat(magobs[s, i], nsamps)
        if x is None:
            xp = mobs
        else:
            if x.shape == (nobj, nsamps):
                xp = x[s].flatten()
            else:
                xp = np.repeat(x[s], nsamps)
        # Plot 2-D histogram.
        mp = mpred[s, :, i].flatten()
        w = wt.flatten()
        if xspan is None:
            xlow, xhigh = quantile(xp, [0.02, 0.98], weights=w)
            bx = np.linspace(xlow, xhigh, bins[i] + 1)
        else:
            bx = np.linspace(xspan[i][0], xspan[i][1], bins[i] + 1)
        if yspan is None:
            ylow, yhigh = quantile(mp - mobs, [0.02, 0.98], weights=w)
            by = np.linspace(ylow, yhigh, bins[i] + 1)
        else:
            by = np.linspace(yspan[i][0], yspan[i][1], bins[i] + 1)
        ax[i].hist2d(
            xp,
            mp - mobs,
            bins=(bx, by),
            weights=w,
            cmin=plot_thresh,
            cmap=cmap,
            **plot_kwargs,
        )
    # Clear other axes.
    for i in range(nfilt, nrows * ncols):
        ax[i].set_frame_on(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()

    return fig, axes


def photometric_offsets_2d(
    phot,
    err,
    mask,
    models,
    idxs,
    reds,
    dreds,
    dists,
    x,
    y,
    flux=True,
    weights=None,
    bins=100,
    offset=None,
    plot_thresh=10.0,
    cmap="coolwarm",
    clims=(-0.05, 0.05),
    xspan=None,
    yspan=None,
    titles=None,
    show_off=True,
    xlabel=None,
    ylabel=None,
    plot_kwargs=None,
    fig=None,
):
    r"""
    Plot photometric offsets (`mag_pred - mag_obs`).

    The draws of each object are weighted, band by band, with the
    leave-one-out weights of `brutus.analysis.photometric_offsets`,
    :math:`w \propto \exp(\chi^2_i / 2)` times `weights`, where
    :math:`\chi^2_i` is the flux-space chi-square of band i. They use `phot`
    and `err` as given; `offset` only shifts the plotted residuals.

    Parameters
    ----------
    phot : `~numpy.ndarray` of shape `(Nobj, Nfilt)`, optional
        Observed data values (fluxes). If provided, these will be overplotted.

    err : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        Associated errors on the data values. If provided, these will be
        overplotted as error bars.

    mask : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        Binary mask (0/1) indicating whether the data value was observed.
        If provided, these will be used to mask missing/bad data values.

    models : `~numpy.ndarray` of shape `(Nmodels, Nfilts, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    idxs : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        An array of resampled indices corresponding to the set of models used
        to fit the data.

    reds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Reddening samples (in Av) associated with the model indices.

    dreds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        "Differential" reddening samples (in Rv) associated with
        the model indices.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Distance samples (in kpc) associated with the model indices.

    x : `~numpy.ndarray` with shape `(Nobj,)`
        Values to be plotted on the `x` axis. Required.

    y : `~numpy.ndarray` with shape `(Nobj,)`
        Values to be plotted on the `y` axis. Required.

    flux : bool, optional
        Whether the photometry provided is in fluxes (instead of magnitudes).
        Default is `True`.

    weights : `~numpy.ndarray` of shape `(Nobj,)` or `(Nobj, Nsamps)`, optional
        Importance weights of the objects or of their samples. Any other
        shape raises `ValueError`.

    bins : int, (int, int) or sequence of `Nfilt` of these, optional
        The number of bins of each map: one int for both axes, an
        `(nx, ny)` pair, or one of these per band. A sequence of length
        `Nfilt` is read as one entry per band. Default is `100`.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative flux offsets. They shift the plotted residuals by
        `2.5 log10(offset)` and do not change the weights.

    plot_thresh : float, optional
        The minimum number of objects in a bin. Bins with fewer objects, or
        with zero total weight, are masked. Default is `10.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'coolwarm'`.

    clims : 2-tuple, optional
        Plotting bounds for the colorbar. Default is `(-0.05, 0.05)`.

    xspan : iterable with shape `(nfilt, 2)`, optional
        The `(low, high)` range of the x bins of each map. If not provided,
        the bins span the range of `x`.

    yspan : iterable with shape `(nfilt, 2)`, optional
        The `(low, high)` range of the y bins of each map. If not provided,
        the bins span the range of `y`.

    titles : iterable of str of length `Nfilt`, optional
        Titles for each of the subplots corresponding to each band.
        If not provided `Band #` will be used.

    show_off : bool, optional
        Whether to include the offsets in the titles. Default is `True`.

    xlabel : str, optional
        Label for the x-axis of each subplot. If not provided,
        this will default to `X`.

    ylabel : str, optional
        Label for the y-axis of each subplot. If not provided,
        this will default to `Y`.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.axes.Axes.imshow`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    postpredplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        The associated figure and axes for the photometric offsets.

    """

    # Initialize values.
    nmodels, nfilt, ncoeff = models.shape
    nobj, nsamps = idxs.shape
    # Coerce to boolean: an integer 0/1 mask would otherwise fancy-index
    # (selecting rows 0/1) and make `~mask` a bitwise complement (-1/-2).
    mask = np.asarray(mask, dtype=bool)
    if plot_kwargs is None:
        plot_kwargs = dict()
    weights = _sample_weights(weights, nobj, nsamps)
    if np.ndim(bins) == 0 or len(bins) != nfilt:
        bins = [bins] * nfilt
    if titles is None:
        titles = ["Band {0}".format(i) for i in range(nfilt)]
    if show_off and offset is not None:
        titles = [
            t + " ({:+.1f}% offset)".format(100.0 * (off - 1.0))
            for t, off in zip(titles, offset)
        ]
    if xlabel is None:
        xlabel = "X"
    if ylabel is None:
        ylabel = "Y"
    if offset is None:
        offset = np.ones(nfilt)

    # Compute posterior predictive SED magnitudes.
    mpred = get_seds(models[idxs.flatten()], av=reds.flatten(), rv=dreds.flatten())
    mpred += 5.0 * np.log10(dists.flatten())[:, None]
    mpred = mpred.reshape(nobj, nsamps, nfilt)

    # Convert observed data to magnitudes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        # The leave-one-out weights use the data of the fit; `offset` only
        # shifts the plotted residuals.
        if flux:
            fobs, feobs = phot, err
            magobs = magnitude(phot * offset, err * offset)[0]
        else:
            # `offset` is a multiplicative *flux* correction; in magnitude
            # space it is applied as -2.5 log10(offset).
            fobs, feobs = inv_magnitude(phot, err)
            magobs = phot - 2.5 * np.log10(offset)

        # Magnitude offsets.
        dm = mpred - magobs[:, None]
        for i in range(nfilt):
            dm[~mask[:, i], :, i] = np.nan

    # Generate figure.
    if fig is None:
        ncols = 5
        nrows = (nfilt - 1) // ncols + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 15, nrows * 12))
    else:
        fig, axes = fig
        nrows, ncols = axes.shape
    ax = axes.flatten()

    # Plot offsets.
    for i in range(nfilt):
        # Bin in 2-D.
        nx, ny = np.broadcast_to(bins[i], (2,))
        xbins = np.histogram_bin_edges(x, nx, None if xspan is None else xspan[i])
        ybins = np.histogram_bin_edges(y, ny, None if yspan is None else yspan[i])
        # Digitize values. The last bin includes its right edge, as in
        # `np.histogram`.
        xloc, yloc = np.digitize(x, xbins) - 1, np.digitize(y, ybins) - 1
        xloc[x == xbins[-1]] = nx - 1
        yloc[y == ybins[-1]] = ny - 1
        # Require at least 4 observed bands besides band i, and finite
        # magnitudes only in *observed* bands: unobserved
        # (masked) bands routinely hold nonpositive/NaN fluxes that map to
        # non-finite magnitudes and must not disqualify the object.
        s = (
            mask[:, i]
            & (np.sum(mask, axis=1) > 4)
            & (np.all(np.isfinite(magobs) | ~mask, axis=1))
            & (weights.sum(axis=1) > 0)
        )
        wt = np.zeros((nobj, nsamps))
        wt_s = _loo_weights(
            fobs[s, i], feobs[s, i], 10 ** (-0.4 * mpred[s, :, i]), weights[s]
        )
        wt[s] = wt_s / wt_s.sum(axis=1, keepdims=True)
        # Compute weighted median offsets.
        offset2d = np.zeros((len(xbins) - 1, len(ybins) - 1))
        for xidx in range(len(xbins) - 1):
            for yidx in range(len(ybins) - 1):
                bsel = np.where((xloc == xidx) & (yloc == yidx) & s)[0]
                off, w = dm[bsel, :, i], wt[bsel]
                if len(bsel) >= plot_thresh and w.sum() > 0:
                    # If we have enough weighted objects, compute the median.
                    off_med = quantile(off.flatten(), [0.5], w.flatten())[0]
                    offset2d[xidx, yidx] = off_med
                else:
                    # Otherwise mask the bin.
                    offset2d[xidx, yidx] = np.nan
        # Plot offsets over 2-D histogram.
        img = ax[i].imshow(
            offset2d.T,
            origin="lower",
            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
            vmin=clims[0],
            vmax=clims[1],
            aspect="auto",
            cmap=cmap,
            **plot_kwargs,
        )
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(titles[i])
        plt.colorbar(img, ax=ax[i], label=r"$\Delta\,$mag")
    # Clear other axes.
    for i in range(nfilt, nrows * ncols):
        ax[i].set_frame_on(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()

    return fig, axes


def _sample_weights(weights, nobj, nsamps):
    """Return `weights` of shape `(nobj,)` or `(nobj, nsamps)` as `(nobj, nsamps)`."""
    if weights is None:
        return np.ones((nobj, nsamps))
    weights = np.asarray(weights, dtype=float)
    if weights.shape == (nobj,):
        return np.broadcast_to(weights[:, None], (nobj, nsamps))
    if weights.shape != (nobj, nsamps):
        raise ValueError(
            f"`weights` must have shape ({nobj},) or ({nobj}, {nsamps}); "
            f"got {weights.shape}."
        )
    return weights


__all__ = ["photometric_offsets", "photometric_offsets_2d"]
