"""
Posterior predictive spectral energy distribution (SED) plots.
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
        Whether to plot SEDs in linear flux densities (maggies) rather than
        magnitudes. Default is `False`.

    data : `~numpy.ndarray` of shape `(Nfilt,)`, optional
        Observed flux densities in maggies to overplot.

    data_err : `~numpy.ndarray` of shape `(Nfilt,)`, optional
        Observed 1-sigma flux density uncertainties in maggies, overplotted as
        ``psig``-sigma error bars.

    data_mask : `~numpy.ndarray` of shape `(Nfilt,)`, optional
        Boolean mask (or 0/1 values) indicating observed bands.

    offset : `~numpy.ndarray` of shape `(Nfilt,)`, optional
        Multiplicative photometric flux offsets applied to ``data`` and
        ``data_err``.

    vcolor : str, optional
        Color for the posterior predictive violin plots. Default is ``'black'``.

    pcolor : str, optional
        Color for the observed photometric points. Default is ``'black'``.

    labels : sequence of str of length `Nfilt`, optional
        Filter names for the x-axis tick labels. If not provided, integer
        indices ``(0, 1, ..., Nfilt - 1)`` are used.

    rstate : `~numpy.random.RandomState` or `~numpy.random.Generator`, optional
        Random number generator used to resample weighted draws.

    psig : float, optional
        Error-bar multiple applied to ``data_err``. Default is ``2.0``.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, draw on the given axes and leave the figure layout
        unchanged. Otherwise, a new figure is created and laid out with
        `~matplotlib.figure.Figure.tight_layout`.

    Returns
    -------
    postpredplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`, dict)
        Figure, axes, and violinplot artist dictionary.

    Raises
    ------
    ValueError
        If ``weights`` is not 1-D of length ``Nsamps`` or ``data_mask`` does
        not have shape ``(Nfilt,)``.

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
    data_mask = np.asarray(data_mask, dtype=bool)
    if data_mask.shape != (nfilt,):
        raise ValueError(f"data_mask must have shape ({nfilt},), got {data_mask.shape}")
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
    own_fig = fig is None
    if own_fig:
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
    if own_fig:
        fig.tight_layout()

    return fig, ax, parts
