#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SED (Spectral Energy Distribution) plotting utilities.

This module contains functions for visualizing stellar SEDs and related
posterior predictive distributions.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..core.sed_utils import get_seds
from ..utils.photometry import magnitude

__all__ = ["posterior_predictive"]


def posterior_predictive(
    models,
    idxs,
    reds,
    dreds,
    dists,
    weights=None,
    flux=False,
    data=None,
    data_err=None,
    data_mask=None,
    offset=None,
    vcolor="black",
    pcolor="black",
    labels=None,
    rstate=None,
    psig=2.0,
    fig=None,
):
    """
    Plot the posterior predictive SED.

    Parameters
    ----------
    models : `~numpy.ndarray` of shape `(Nmodels, Nfilts, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    idxs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of resampled indices corresponding to the set of models used
        to fit the data.

    reds : `~numpy.ndarray` of shape `(Nsamps)`
        Reddening samples (in Av) associated with the model indices.

    dreds : `~numpy.ndarray` of shape `(Nsamps)`
        "Differential" reddening samples (in Rv) associated with
        the model indices.

    dists : `~numpy.ndarray` of shape `(Nsamps)`
        Distance samples (in kpc) associated with the model indices.

    weights : `~numpy.ndarray` of shape `(Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    flux : bool, optional
        Whether to plot the SEDs in flux space rather than magniude space.
        Default is `False`.

    data : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Observed data values (fluxes). If provided, these will be overplotted.

    data_err : `~numpy.ndarray` of shape `(Nfilt)`
        Associated 1-sigma errors on the data values. If provided,
        these will be overplotted as `psig`-sigma error bars (default: 2-sigma).

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`
        Binary mask (0/1) indicating whether the data value was observed.
        If provided, these will be used to mask missing/bad data values.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that will be applied to
        the data (i.e. `data_new = data * phot_offsets`) and errors
        when provided.

    vcolor : str, optional
        Color used when plotting the violin plots that comprise the
        SED posterior predictive distribution. Default is `'black'`.

    pcolor : str, optional
        Color used when plotting the provided data values.
        Default is `'black'`.

    labels : iterable with shape `(ndim,)`, optional
        A list of names corresponding to each filter. If not provided,
        an ascending set of integers `(0, 1, 2, ...)` will be used.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    psig : float, optional
        The number of sigma to plot when showcasing the error bars
        from any provided `data_err`. Default is `2.`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    postpredplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`, dict)
        The associated figure, axes, and violinplot dictionary for the
        posterior predictive distribution.

    """

    # Initialize values.
    nmodels, nfilt, ncoeff = models.shape
    nsamps = len(idxs)
    if rstate is None:
        rstate = np.random
    if weights is None:
        weights = np.ones_like(idxs, dtype="float")
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")
    if data_err is None:
        data_err = np.zeros(nfilt)
    if data_mask is None:
        data_mask = np.ones(nfilt, dtype="bool")
    if offset is None:
        offset = np.ones(nfilt)

    # Generate SEDs.
    seds = get_seds(models[idxs], av=reds, rv=dreds, return_flux=flux)
    if flux:
        # SEDs are in flux space.
        seds /= dists[:, None] ** 2
    else:
        # SEDs are in magnitude space.
        seds += 5.0 * np.log10(dists)[:, None]

    # Generate figure.
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(nfilt * 1.5, 10))
    else:
        fig, ax = fig

    # Plot posterior predictive SED distribution.
    if np.any(weights != weights[0]):
        # If weights are non-uniform, sample indices proportional to weights.
        idxs = rstate.choice(nsamps, p=weights / weights.sum(), size=nsamps * 10)
    else:
        idxs = np.arange(nsamps)
    parts = ax.violinplot(seds[idxs], positions=np.arange(nfilt), showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(vcolor)
        pc.set_edgecolor("none")
        pc.set_alpha(0.4)
    # Plot photometry.
    if data is not None:
        if flux:
            m = data[data_mask] * offset[data_mask]
            e = data_err[data_mask] * offset[data_mask]
        else:
            m, e = magnitude(
                data[data_mask] * offset[data_mask],
                data_err[data_mask] * offset[data_mask],
            )
        ax.errorbar(
            np.arange(nfilt)[data_mask],
            m,
            yerr=psig * e,
            marker="o",
            color=pcolor,
            linestyle="none",
            ms=7,
            lw=3,
        )
    # Label axes.
    ax.set_xticks(np.arange(nfilt))
    if labels is not None:
        ax.set_xticklabels(labels, rotation="vertical")
    if flux:
        ax.set_ylabel("Flux")
    else:
        ax.set_ylabel("Magnitude")
        ax.set_ylim(ax.get_ylim()[::-1])  # flip axis
    plt.tight_layout()

    return fig, ax, parts
