#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Corner plot visualization functions.

This module provides functions for creating corner plots of multi-dimensional
posterior distributions.
"""

import copy
import warnings
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.special import logsumexp

from ..priors import logp_galactic_structure as gal_lnprior
from ..priors import logp_parallax
from ..utils.sampling import draw_sar, quantile
from .utils import hist2d

__all__ = ["cornerplot"]


def cornerplot(
    idxs,
    data,
    params,
    lndistprior=None,
    coord=None,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    weights=None,
    parallax=None,
    parallax_err=None,
    Nr=500,
    applied_parallax=True,
    pcolor="blue",
    parallax_kwargs=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=10,
    hist_kwargs=None,
    hist2d_kwargs=None,
    labels=None,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    title_quantiles=[0.025, 0.5, 0.975],
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
    fig=None,
    rstate=None,
    R_solar=8.2,
    Z_solar=0.025,
):
    """
    Generate a corner plot of the 1-D and 2-D marginalized posteriors.

    Parameters
    ----------
    idxs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of resampled indices corresponding to the set of models used
        to fit the data.

    data : 3-tuple or 4-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of
        `(dists, reds, dreds)` that were saved, or a collection of
        `(scales, avs, rvs, covs_sar)` that will be used to regenerate
        `(dists, reds, dreds)` in conjunction with any applied distance
        and/or parallax priors.

    params : structured `~numpy.ndarray` with shape `(Nmodels,)`
        Set of parameters corresponding to the input set of models. Note that
        `'agewt'` will always be ignored.

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

    weights : `~numpy.ndarray` of shape `(Nsamps)`, optional
        An optional set of importance weights used to reweight the samples.

    parallax : float, optional
        The parallax estimate for the source.

    parallax_err : float, optional
        The parallax error.

    Nr : int, optional
        The number of Monte Carlo realizations used when sampling using the
        provided parallax prior. Default is `500`.

    applied_parallax : bool, optional
        Whether the parallax was applied when initially computing the fits.
        Default is `True`.

    pcolor : str, optional
        Color used when plotting the parallax prior. Default is `'blue'`.

    parallax_kwargs : kwargs, optional
        Keyword arguments used when plotting the parallax prior passed to
        `fill_between`.

    span : iterable with shape `(ndim,)`, optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.99` (99% credible interval).

    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (spanning the 95%/2-sigma credible interval).

    color : str or iterable with shape `(ndim,)`, optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'black'`.

    smooth : float or iterable with shape `(ndim,)`, optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
        marginalized posteriors, expressed as a fraction of the span.
        If an integer is provided instead, this will instead default
        to a simple (weighted) histogram with `bins=smooth`.
        Default is `10` (10 bins).

    hist_kwargs : dict, optional
        Extra keyword arguments to send to the 1-D (smoothed) histograms.

    hist2d_kwargs : dict, optional
        Extra keyword arguments to send to the 2-D (smoothed) histograms.

    labels : iterable with shape `(ndim,)`, optional
        A list of names for each parameter. If not provided, the names will
        be taken from `params.dtype.names`.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the quantiles specified by `title_quantiles`. By default,
        This will show the median (0.5 quantile) along with the upper/lower
        bounds associated with the 0.025 and 0.975 (95%/2-sigma credible
        interval) quantiles.
        Default is `False`.

    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.

    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.

    title_quantiles : iterable, optional
        A list of 3 fractional quantiles displayed in the title, ordered
        from lowest to highest. Default is `[0.025, 0.5, 0.975]`
        (spanning the 95%/2-sigma credible interval).

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

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    top_ticks : bool, optional
        Whether to label the top (rather than bottom) ticks. Default is
        `False`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    cornerplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output corner plot.

    """

    # Initialize values.
    if quantiles is None:
        quantiles = []
    if truth_kwargs is None:
        truth_kwargs = dict()
    else:
        truth_kwargs = dict(truth_kwargs)
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    # Copy caller-supplied kwargs dicts: defaults are written into them below,
    # which would otherwise leak state back into the caller and across calls.
    if hist_kwargs is None:
        hist_kwargs = dict()
    else:
        hist_kwargs = dict(hist_kwargs)
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()
    else:
        hist2d_kwargs = dict(hist2d_kwargs)
    if weights is None:
        weights = np.ones_like(idxs, dtype="float")
    if rstate is None:
        rstate = np.random
    if applied_parallax:
        if parallax is None or parallax_err is None:
            raise ValueError(
                "`parallax` and `parallax_err` must be provided " "together."
            )
    if parallax_kwargs is None:
        parallax_kwargs = dict()
    else:
        parallax_kwargs = dict(parallax_kwargs)
    if lndistprior is None:
        lndistprior = partial(gal_lnprior, R_solar=R_solar, Z_solar=Z_solar)

    # Set defaults.
    hist_kwargs["alpha"] = hist_kwargs.get("alpha", 0.6)
    hist2d_kwargs["alpha"] = hist2d_kwargs.get("alpha", 0.6)
    truth_kwargs["linestyle"] = truth_kwargs.get("linestyle", "solid")
    truth_kwargs["linewidth"] = truth_kwargs.get("linewidth", 2)
    truth_kwargs["alpha"] = truth_kwargs.get("alpha", 0.7)
    parallax_kwargs["alpha"] = parallax_kwargs.get("alpha", 0.3)

    # Ignore age weights.
    if labels is None:
        labels = [x for x in params.dtype.names if x != "agewt"]
    else:
        # Copy any caller-supplied labels: the Av/Rv/Parallax/Distance appends
        # below mutate this list in place, which would otherwise corrupt the
        # caller's list and break reuse across repeated calls.
        labels = list(labels)

    # Deal with 1D results.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        samples = params[idxs]
        samples = np.array([samples[lbl] for lbl in labels]).T
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], (
        "There are more " "dimensions than samples!"
    )

    try:
        # Grab distance and reddening samples.
        ddraws, adraws, rdraws = copy.deepcopy(data)
        pdraws = 1.0 / ddraws
    except (ValueError, TypeError):
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = copy.deepcopy(data)

        _is_default_prior = lndistprior is gal_lnprior or (
            hasattr(lndistprior, "func") and lndistprior.func is gal_lnprior
        )
        if _is_default_prior and coord is None:
            raise ValueError(
                "`coord` must be passed if the default distance " "prior was used."
            )

        # Add in scale/parallax/distance, Av, and Rv realizations.
        nsamps = len(idxs)
        sdraws, adraws, rdraws = draw_sar(
            scales,
            avs,
            rvs,
            covs_sar,
            ndraws=Nr,
            avlim=avlim,
            rvlim=rvlim,
            rstate=rstate,
        )
        pdraws = np.sqrt(sdraws)
        ddraws = 1.0 / pdraws

        # Re-apply distance and parallax priors to realizations.
        # Draws come from a Gaussian proposal in *scale* space (s = d^-2),
        # while `lndistprior` is a density over distance, so the importance
        # weight needs the change-of-variables Jacobian |dd/ds| = d^3 / 2
        # (constant factor irrelevant after normalization). Matches the
        # ln-distance Jacobian handling in `analysis.individual.logpost_grid`.
        lnp_draws = lndistprior(ddraws, coord) + 3.0 * np.log(ddraws)
        if applied_parallax:
            lnp_draws += logp_parallax(pdraws, parallax, parallax_err)

        # Resample draws.
        lnp = logsumexp(lnp_draws, axis=1)
        pwt = np.exp(lnp_draws - lnp[:, None])
        pwt /= pwt.sum(axis=1)[:, None]
        ridx = [rstate.choice(Nr, p=pwt[i]) for i in range(nsamps)]
        pdraws = pdraws[np.arange(nsamps), ridx]
        ddraws = ddraws[np.arange(nsamps), ridx]
        adraws = adraws[np.arange(nsamps), ridx]
        rdraws = rdraws[np.arange(nsamps), ridx]

    # Append to samples.
    samples = np.c_[samples.T, adraws, rdraws, pdraws, ddraws].T
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds.
    if span is None:
        span = [0.99 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except (TypeError, ValueError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = quantile(samples[i], q, weights=weights)

    # Append additional labels for extra dimensions
    labels.append("Av")
    labels.append("Rv")
    labels.append("Parallax")
    labels.append("Distance")

    # Setting up smoothing.
    if isinstance(smooth, int) or isinstance(smooth, float):
        smooth = [smooth for i in range(ndim)]

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * ndim + factor * (ndim - 1.0) * whspace  # plot size
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim, ndim))
        except (ValueError, TypeError):
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    # Plotting.
    for i, x in enumerate(samples):
        if np.shape(samples)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]

        # Plot the 1-D marginalized posteriors.

        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        if i < ndim - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
            else:
                ax.set_xticklabels([])
        else:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.set_xlabel(labels[i], **label_kwargs)
            ax.xaxis.set_label_coords(0.5, -0.3)
        # Generate distribution.
        sx = smooth[i]
        if isinstance(sx, int):
            # If `sx` is an integer, plot a weighted histogram with
            # `sx` bins within the provided bounds.
            n, b, _ = ax.hist(
                x,
                bins=sx,
                weights=weights,
                color=color,
                range=np.sort(span[i]),
                **hist_kwargs,
            )
        else:
            # If `sx` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10.0 / sx))
            n, b = np.histogram(x, bins=bins, weights=weights, range=np.sort(span[i]))
            n = norm_kde(n, 10.0)
            b0 = 0.5 * (b[1:] + b[:-1])
            n, b, _ = ax.hist(
                b0,
                bins=b,
                weights=n,
                range=np.sort(span[i]),
                color=color,
                **hist_kwargs,
            )
        ax.set_ylim([0.0, max(n) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                for t in truths[i]:
                    ax.axvline(t, color=truth_color, **truth_kwargs)
            except TypeError:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = quantile(x, title_quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = f"{{0:{title_fmt}}}".format
                title = rf"${{{fmt(qm)}}}_{{-{fmt(q_minus)}}}^{{+{fmt(q_plus)}}}$"
                title = f"{labels[i]} = {title}"
                ax.set_title(title, **title_kwargs)
        # Add parallax prior.
        if i == ndim - 2 and parallax is not None and parallax_err is not None:
            parallax_logpdf = logp_parallax(b, parallax, parallax_err)
            parallax_pdf = np.exp(parallax_logpdf - max(parallax_logpdf))
            parallax_pdf *= max(n) / max(parallax_pdf)
            ax.fill_between(b, parallax_pdf, color=pcolor, **parallax_kwargs)

        for j, y in enumerate(samples):
            if np.shape(samples)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]

            # Plot the 2-D marginalized posteriors.

            # Setup axes.
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                for tick in ax.get_yticklabels():
                    tick.set_rotation(45)
                ax.set_ylabel(labels[i], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Generate distribution.
            sy = smooth[j]
            check_ix = isinstance(sx, int)
            check_iy = isinstance(sy, int)
            if check_ix and check_iy:
                fill_contours = False
                plot_contours = False
            else:
                fill_contours = True
                plot_contours = True
            # Resolve contour flags per panel (caller overrides win) without
            # writing back into `hist2d_kwargs`: storing them there would
            # freeze the first panel's smooth-based choice for all later
            # panels with different int/float smoothing.
            hist2d_extra = {
                k: v
                for k, v in hist2d_kwargs.items()
                if k not in ("fill_contours", "plot_contours")
            }
            hist2d(
                y,
                x,
                ax=ax,
                span=[span[j], span[i]],
                weights=weights,
                color=color,
                smooth=[sy, sx],
                fill_contours=hist2d_kwargs.get("fill_contours", fill_contours),
                plot_contours=hist2d_kwargs.get("plot_contours", plot_contours),
                **hist2d_extra,
            )

            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        for t in truths[j]:
                            ax.axvline(t, color=truth_color, **truth_kwargs)
                    except TypeError:
                        ax.axvline(truths[j], color=truth_color, **truth_kwargs)
                if truths[i] is not None:
                    try:
                        for t in truths[i]:
                            ax.axhline(t, color=truth_color, **truth_kwargs)
                    except TypeError:
                        ax.axhline(truths[i], color=truth_color, **truth_kwargs)

    return (fig, axes)
