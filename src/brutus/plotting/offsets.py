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
from scipy.special import logsumexp

from ..core.sed_utils import get_seds
from ..utils.photometry import magnitude, phot_loglike
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
    dim_prior=True,
    plot_thresh=0.0,
    cmap="viridis",
    xspan=None,
    yspan=None,
    titles=None,
    xlabel=None,
    plot_kwargs=None,
    fig=None,
):
    """
    Plot photometric offsets (`mag_pred - mag_obs`).

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

    dreds : `~numpy.ndarray` of shape `(Nsamps)`
        "Differential" reddening samples (in Rv) associated with
        the model indices.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Distance samples (in kpc) associated with the model indices.

    x : `~numpy.ndarray` with shape `(Nobj)` or `(Nobj, Nsamps)`, optional
        Corresponding values to be plotted on the `x` axis. In not provided,
        the default behavior is to plot as a function of observed magnitude.

    flux : bool, optional
        Whether the photometry provided is in fluxes (instead of magnitudes).
        Default is `True`.

    weights : `~numpy.ndarray` of shape `(Nobj)` or `(Nobj, Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    bins : single value or iterable of length `Nfilt`, optional
        The number of bins to use. Passed to `~matplotlib.pyplot.hist2d`.
        Default is `100`.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that will be applied to
        the data (i.e. `data_new = data * phot_offsets`) and errors
        when provided.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood when reweighting the data while cycling through each
        band. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 3` degrees of freedom. Default is `True`.

    plot_thresh : float, optional
        The threshold used to threshold the colormap when plotting.
        Default is `0.`.

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
        Labels for the x-axis of each subplot. If not provided,
        these will default to the titles.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.imshow`.

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
    if plot_kwargs is None:
        plot_kwargs = dict()
    if weights is None:
        weights = np.ones((nobj, nsamps))
    elif weights.shape != (nobj, nsamps):
        weights = np.repeat(weights, nsamps).reshape(nobj, nsamps)
    try:
        nbins = len(bins)
        if nbins != 2:
            bins = [b for b in bins]
        else:
            bins = [bins for i in range(nfilt)]
    except TypeError:
        bins = [bins for i in range(nfilt)]
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
        if flux:
            magobs, mageobs = magnitude(phot * offset, err * offset)
        else:
            magobs, mageobs = phot + offset, err

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
        # Compute selection ignoring current band.
        mtemp = np.array(mask)
        mtemp[:, i] = False
        s = (
            mask[:, i]
            & (np.sum(mtemp, axis=1) > 3)
            & (np.all(np.isfinite(magobs), axis=1))
        )
        # Compute weights from ignoring current band.
        # TODO: vectorize this loop over objects for performance
        lnl = np.array(
            [
                phot_loglike(
                    mo[None, :],
                    me[None, :],
                    mp[None, :, :],
                    mask=mt[None, :],
                    dim_prior=dim_prior,
                ).squeeze(0)
                for mo, me, mt, mp in zip(magobs[s], mageobs[s], mtemp[s], mpred[s])
            ]
        )
        levid = logsumexp(lnl, axis=1)
        logwt = lnl - levid[:, None]
        wt = np.exp(logwt)
        wt /= wt.sum(axis=1)[:, None]
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
        w = weights[s].flatten() * wt.flatten()
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
        ax[i].set_xlabel(xlabel[i])
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(r"$\Delta\,$mag")
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
    dim_prior=True,
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
    """
    Plot photometric offsets (`mag_pred - mag_obs`).

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

    dreds : `~numpy.ndarray` of shape `(Nsamps)`
        "Differential" reddening samples (in Rv) associated with
        the model indices.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Distance samples (in kpc) associated with the model indices.

    x : `~numpy.ndarray` with shape `(Nobj)` or `(Nobj, Nsamps)`
        Corresponding values to be plotted on the `x` axis. In not provided,
        the default behavior is to plot as a function of observed magnitude.

    y : `~numpy.ndarray` with shape `(Nobj)` or `(Nobj, Nsamps)`
        Corresponding values to be plotted on the `y` axis.

    flux : bool, optional
        Whether the photometry provided is in fluxes (instead of magnitudes).
        Default is `True`.

    weights : `~numpy.ndarray` of shape `(Nobj)` or `(Nobj, Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    bins : single value or iterable of length `Nfilt`, optional
        The number of bins to use. Passed to `~matplotlib.pyplot.hist2d`.
        Default is `100`.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that will be applied to
        the data (i.e. `data_new = data * phot_offsets`) and errors
        when provided.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood when reweighting the data while cycling through each
        band. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 3` degrees of freedom. Default is `True`.

    plot_thresh : float, optional
        The threshold used to threshold the colormap when plotting.
        Default is `10.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'coolwarm'`.

    clims : 2-tuple, optional
        Plotting bounds for the colorbar. Default is `(-0.05, 0.05)`.

    xspan : iterable with shape `(nfilt, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds for the x-axis for each plot.

    yspan : iterable with shape `(nfilt, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds for the y-axis for each plot.

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
        Keyword arguments to be passed to `~matplotlib.pyplot.imshow`.

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
    if plot_kwargs is None:
        plot_kwargs = dict()
    if weights is None:
        weights = np.ones((nobj, nsamps))
    elif weights.shape != (nobj, nsamps):
        weights = np.repeat(weights, nsamps).reshape(nobj, nsamps)
    try:
        nbins = len(bins)
        if nbins != 2:
            bins = [b for b in bins]
        else:
            bins = [bins for i in range(nfilt)]
    except TypeError:
        bins = [bins for i in range(nfilt)]
    if titles is None:
        titles = ["Band {0}".format(i) for i in range(nfilt)]
    if show_off and offset is not None:
        titles = [
            t + " ({:2.2}% offset)".format(100.0 * (off - 1.0))
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
        if flux:
            magobs, mageobs = magnitude(phot * offset, err * offset)
        else:
            magobs, mageobs = phot + offset, err

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
        n, xbins, ybins = np.histogram2d(x, y, bins=bins[i])
        xcent = 0.5 * (xbins[1:] + xbins[:-1])
        ycent = 0.5 * (ybins[1:] + ybins[:-1])
        bounds = [xcent[0], xcent[-1], ycent[0], ycent[-1]]  # default size
        # Digitize values.
        xloc, yloc = np.digitize(x, xbins) - 1, np.digitize(y, ybins) - 1
        # Compute selection ignoring current band.
        mtemp = np.array(mask)
        mtemp[:, i] = False
        s = (
            mask[:, i]
            & (np.sum(mtemp, axis=1) > 3)
            & (np.all(np.isfinite(magobs), axis=1))
        )
        # Compute weights from ignoring current band.
        # TODO: vectorize this loop over objects for performance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            lnl = np.array(
                [
                    phot_loglike(
                        mo[None, :],
                        me[None, :],
                        mp[None, :, :],
                        mask=mt[None, :],
                        dim_prior=dim_prior,
                    ).squeeze(0)
                    for mo, me, mt, mp in zip(magobs, mageobs, mtemp, mpred)
                ]
            )
            levid = logsumexp(lnl, axis=1)
            logwt = lnl - levid[:, None]
            wt = np.exp(logwt)
            wt /= wt.sum(axis=1)[:, None]
        # Compute weighted median offsets.
        offset2d = np.zeros((len(xbins) - 1, len(ybins) - 1))
        for xidx in range(len(xbins) - 1):
            for yidx in range(len(ybins) - 1):
                bsel = np.where((xloc == xidx) & (yloc == yidx) & s)[0]
                if len(bsel) >= plot_thresh:
                    # If we have enough objects, compute weighted median.
                    off, w = dm[bsel, :, i], wt[bsel] * weights[bsel]
                    off_med = quantile(off.flatten(), [0.5], w.flatten())[0]
                    offset2d[xidx, yidx] = off_med
                else:
                    # If we don't have enough objects, mask bin.
                    offset2d[xidx, yidx] = np.nan
        # Plot offsets over 2-D histogram.
        if xspan is not None:
            bounds[:2] = xspan[i]
        if yspan is not None:
            bounds[2:] = yspan[i]
        img = ax[i].imshow(
            offset2d.T,
            origin="lower",
            extent=bounds,
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


__all__ = ["photometric_offsets", "photometric_offsets_2d"]
