#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting utilities.

"""

from __future__ import (print_function, division)
from six.moves import range

import warnings
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter as norm_kde
import copy

from .pdf import gal_lnprior, parallax_lnprior
from .utils import quantile, draw_sar, get_seds, magnitude, phot_loglike

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

str_type = str
float_type = float
int_type = int

__all__ = ["cornerplot", "dist_vs_red", "posterior_predictive",
           "photometric_offsets", "photometric_offsets_2d", "_hist2d"]


def cornerplot(idxs, data, params, lndistprior=None, coord=None,
               avlim=(0., 6.), rvlim=(1., 8.), weights=None, parallax=None,
               parallax_err=None, Nr=500, applied_parallax=True,
               pcolor='blue', parallax_kwargs=None, span=None,
               quantiles=[0.025, 0.5, 0.975], color='black', smooth=10,
               hist_kwargs=None, hist2d_kwargs=None, labels=None,
               label_kwargs=None, show_titles=False, title_fmt=".2f",
               title_kwargs=None, title_quantiles=[0.025, 0.5, 0.975],
               truths=None, truth_color='red',
               truth_kwargs=None, max_n_ticks=5, top_ticks=False,
               use_math_text=False, verbose=False, fig=None, rstate=None):
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
        Default is `True`.

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
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist_kwargs is None:
        hist_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()
    if weights is None:
        weights = np.ones_like(idxs, dtype='float')
    if rstate is None:
        rstate = np.random
    if applied_parallax:
        if parallax is None or parallax_err is None:
            raise ValueError("`parallax` and `parallax_err` must be provided "
                             "together.")
    if parallax_kwargs is None:
        parallax_kwargs = dict()
    if lndistprior is None:
        lndistprior = gal_lnprior

    # Set defaults.
    hist_kwargs['alpha'] = hist_kwargs.get('alpha', 0.6)
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.6)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)
    parallax_kwargs['alpha'] = parallax_kwargs.get('alpha', 0.3)

    # Ignore age weights.
    labels = [x for x in params.dtype.names if x != 'agewt']

    # Deal with 1D results.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        samples = params[idxs]
        samples = np.array([samples[l] for l in labels]).T
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    try:
        # Grab distance and reddening samples.
        ddraws, adraws, rdraws = copy.deepcopy(data)
        pdraws = 1. / ddraws
    except:
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = copy.deepcopy(data)

        if lndistprior == gal_lnprior and coord is None:
            raise ValueError("`coord` must be passed if the default distance "
                             "prior was used.")

        # Add in scale/parallax/distance, Av, and Rv realizations.
        nsamps = len(idxs)
        sdraws, adraws, rdraws = draw_sar(scales, avs, rvs, covs_sar,
                                          ndraws=Nr, avlim=avlim, rvlim=rvlim,
                                          rstate=rstate)
        pdraws = np.sqrt(sdraws)
        ddraws = 1. / pdraws

        # Re-apply distance and parallax priors to realizations.
        lnp_draws = lndistprior(ddraws, coord)
        if applied_parallax:
            lnp_draws += parallax_lnprior(pdraws, parallax, parallax_err)

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
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = quantile(samples[i], q, weights=weights)

    # Set labels
    if labels is None:
        labels = list(params.dtype.names)
    labels.append('Av')
    labels.append('Rv')
    labels.append('Parallax')
    labels.append('Distance')

    # Setting up smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim, ndim))
        except:
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

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
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                   prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        if i < ndim - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xlabel(labels[i], **label_kwargs)
            ax.xaxis.set_label_coords(0.5, -0.3)
        # Generate distribution.
        sx = smooth[i]
        if isinstance(sx, int_type):
            # If `sx` is an integer, plot a weighted histogram with
            # `sx` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=sx, weights=weights, color=color,
                              range=np.sort(span[i]), **hist_kwargs)
        else:
            # If `sx` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / sx))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            b0 = 0.5 * (b[1:] + b[:-1])
            n, b, _ = ax.hist(b0, bins=b, weights=n,
                              range=np.sort(span[i]), color=color,
                              **hist_kwargs)
        ax.set_ylim([0., max(n) * 1.05])
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
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = quantile(x, title_quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)
        # Add parallax prior.
        if i == ndim - 2 and parallax is not None and parallax_err is not None:
            parallax_logpdf = parallax_lnprior(b, parallax, parallax_err)
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
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(labels[i], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Generate distribution.
            sy = smooth[j]
            check_ix = isinstance(sx, int_type)
            check_iy = isinstance(sy, int_type)
            if check_ix and check_iy:
                fill_contours = False
                plot_contours = False
            else:
                fill_contours = True
                plot_contours = True
            hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                               fill_contours)
            hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                               plot_contours)
            _hist2d(y, x, ax=ax, span=[span[j], span[i]],
                    weights=weights, color=color, smooth=[sy, sx],
                    **hist2d_kwargs)

            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, color=truth_color, **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], color=truth_color,
                                   **truth_kwargs)
                if truths[i] is not None:
                    try:
                        [ax.axhline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], color=truth_color,
                                   **truth_kwargs)

    return (fig, axes)


def dist_vs_red(data, ebv=None, dist_type='distance_modulus',
                lndistprior=None, coord=None, avlim=(0., 6.), rvlim=(1., 8.),
                weights=None, parallax=None, parallax_err=None, Nr=300,
                cmap='Blues', bins=300, span=None, smooth=0.01,
                plot_kwargs=None, truths=None, truth_color='red',
                truth_kwargs=None, rstate=None):
    """
    Generate a 2-D plot of distance vs reddening.

    Parameters
    ----------
    data : 3-tuple or 4-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of
        `(dists, reds, dreds)` that were saved, or a collection of
        `(scales, avs, rvs, covs_sar)` that will be used to regenerate
        `(dists, reds)` in conjunction with any applied distance
        and/or parallax priors.

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

    weights : `~numpy.ndarray` of shape `(Nsamps)`, optional
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
        Default is `0.01` (1% smoothing).
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
    if weights is None:
        weights = np.ones_like(data[0], dtype='float')
    if rstate is None:
        rstate = np.random
    if lndistprior is None:
        lndistprior = gal_lnprior
    if parallax is None or parallax_err is None:
        parallax, parallax_err = np.nan, np.nan

    # Establish minimum smoothing in distance.
    p1sig = np.array([parallax + parallax_err,
                      max(parallax - parallax_err, 1e-10)])
    p_min_smooth = abs(np.diff(p1sig)) / 2.
    s_min_smooth = abs(np.diff(p1sig**2)) / 2.
    d_min_smooth = abs(np.diff(1. / p1sig)) / 2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        dm_min_smooth = abs(np.diff(5. * np.log10(1. / p1sig) + 10.)) / 2.

    # Set up axes and labels.
    if dist_type not in ['parallax', 'scale', 'distance', 'distance_modulus']:
        raise ValueError("The provided `dist_type` is not valid.")
    if span is None:
        avlims = avlim
        dlims = 10**(np.array([4., 19.]) / 5. - 2.)
    else:
        avlims, dlims = span
    try:
        xbin, ybin = bins
    except:
        xbin = ybin = bins
    if ebv:
        ylabel = r'$E(B-V)$ [mag]'
        ylims = avlims  # default Rv goes from [1., 8.] -> min(Rv) = 1.
    else:
        ylabel = r'$A_v$ [mag]'
        ylims = avlims
    if dist_type == 'scale':
        xlabel = r'$s$'
        xlims = (1. / dlims[::-1])**2
        x_min_smooth = s_min_smooth
    elif dist_type == 'parallax':
        xlabel = r'$\pi$ [mas]'
        xlims = 1. / dlims[::-1]
        x_min_smooth = p_min_smooth
    elif dist_type == 'distance':
        xlabel = r'$d$ [kpc]'
        xlims = dlims
        x_min_smooth = d_min_smooth
    elif dist_type == 'distance_modulus':
        xlabel = r'$\mu$'
        xlims = 5. * np.log10(dlims) + 10.
        x_min_smooth = dm_min_smooth
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
    except:
        if smooth < 1:
            xsmooth, ysmooth = smooth * xspan, smooth * yspan
        else:
            xsmooth, ysmooth = smooth * dx, smooth * dy
    if np.isfinite(x_min_smooth):
        xsmooth = min(x_min_smooth, xsmooth)
    try:
        xsmooth = xsmooth[0]  # catch possible list
    except:
        pass

    # Set defaults.
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)

    try:
        # Grab distance and reddening samples.
        ddraws, adraws, rdraws = copy.deepcopy(data)
        pdraws = 1. / ddraws
        sdraws = pdraws**2
        dmdraws = 5. * np.log10(ddraws) + 10.
    except:
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = copy.deepcopy(data)

        if lndistprior is None and coord is None:
            raise ValueError("`coord` must be passed if the default distance "
                             "prior was used.")

        # Generate parallax and Av realizations.
        sdraws, adraws, rdraws = draw_sar(scales, avs, rvs, covs_sar,
                                          ndraws=Nr, avlim=avlim, rvlim=rvlim,
                                          rstate=rstate)
        pdraws = np.sqrt(sdraws)
        ddraws = 1. / pdraws
        dmdraws = 5. * np.log10(ddraws) + 10.

        # Re-apply distance and parallax priors to realizations.
        lnp_draws = lndistprior(ddraws, coord)
        if parallax is not None and parallax_err is not None:
            lnp_draws += parallax_lnprior(pdraws, parallax, parallax_err)
        lnp = logsumexp(lnp_draws, axis=1)
        pwt = np.exp(lnp_draws - lnp[:, None])
        pwt /= pwt.sum(axis=1)[:, None]
        weights = np.repeat(weights, Nr)
        weights *= pwt.flatten()

    # Grab draws.
    ydraws = adraws.flatten()
    if ebv:
        ydraws /= rdraws.flatten()
    if dist_type == 'scale':
        xdraws = sdraws.flatten()
    elif dist_type == 'parallax':
        xdraws = pdraws.flatten()
    elif dist_type == 'distance':
        xdraws = ddraws.flatten()
    elif dist_type == 'distance_modulus':
        xdraws = dmdraws.flatten()

    # Generate 2-D histogram.
    H, xedges, yedges = np.histogram2d(xdraws, ydraws, bins=(xbins, ybins),
                                       weights=weights)

    # Apply smoothing.
    H = norm_kde(H, (xsmooth / dx, ysmooth / dy))

    # Generate 2-D histogram.
    img = plt.imshow(H.T, cmap=cmap, aspect='auto',
                     interpolation=None, origin='lower',
                     extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                     **plot_kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return H, xedges, yedges, img


def posterior_predictive(models, idxs, reds, dreds, dists, weights=None,
                         flux=False, data=None, data_err=None, data_mask=None,
                         offset=None, vcolor='blue', pcolor='black',
                         labels=None, rstate=None, psig=2., fig=None):
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
        these will be overplotted as **3-sigma** error bars.

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`
        Binary mask (0/1) indicating whether the data value was observed.
        If provided, these will be used to mask missing/bad data values.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that will be applied to
        the data (i.e. `data_new = data * phot_offsets`) and errors
        when provided.

    vcolor : str, optional
        Color used when plotting the violin plots that comprise the
        SED posterior predictive distribution. Default is `'blue'`.

    pcolor : str, optional
        Color used when plotting the provided data values.
        Default is `'black'`.

    labels : iterable with shape `(ndim,)`, optional
        A list of names corresponding to each filter. If not provided,
        an ascending set of integers `(0, 1, 2, ...)` will be used.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    top_ticks : bool, optional
        Whether to label the top (rather than bottom) ticks. Default is
        `False`.

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
        weights = np.ones_like(idxs, dtype='float')
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")
    if data_err is None:
        data_err = np.zeros(nfilt)
    if data_mask is None:
        data_mask = np.ones(nfilt, dtype='bool')
    if offset is None:
        offset = np.ones(nfilt)

    # Generate SEDs.
    seds = get_seds(models[idxs], av=reds, rv=dreds, return_flux=flux)
    if flux:
        # SEDs are in flux space.
        seds /= dists[:, None]**2
    else:
        # SEDs are in magnitude space.
        seds += 5. * np.log10(dists)[:, None]

    # Generate figure.
    if fig is None:
        fig, ax = fig, axes = plt.subplots(1, 1, figsize=(nfilt * 1.5, 10))
    else:
        fig, ax = fig

    # Plot posterior predictive SED distribution.
    if np.any(weights != weights[0]):
        # If weights are non-uniform, sample indices proportional to weights.
        idxs = rstate.choice(nsamps, p=weights / weights.sum(),
                             size=nsamps * 10)
    else:
        idxs = np.arange(nsamps)
    parts = ax.violinplot(seds, positions=np.arange(nfilt),
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(vcolor)
        pc.set_edgecolor('none')
        pc.set_alpha(0.4)
    # Plot photometry.
    if data is not None:
        if flux:
            m = data[data_mask] * offset[data_mask]
            e = data_err[data_mask] * offset[data_mask]
        else:
            m, e = magnitude(data[data_mask] * offset[data_mask],
                             data_err[data_mask] * offset[data_mask])
        ax.errorbar(np.arange(nfilt)[data_mask], m, yerr=psig * e,
                    marker='o', color=pcolor, linestyle='none',
                    ms=7, lw=3)
    # Label axes.
    ax.set_xticks(np.arange(nfilt))
    if labels is not None:
        ax.set_xticklabels(labels, rotation='vertical')
    if flux:
        ax.set_ylabel('Flux')
    else:
        ax.set_ylabel('Magnitude')
        ax.set_ylim(ax.get_ylim()[::-1])  # flip axis
    plt.tight_layout()

    return fig, ax, parts


def photometric_offsets(phot, err, mask, models, idxs, reds, dreds, dists,
                        x=None, flux=True, weights=None, bins=100,
                        offset=None, dim_prior=True,
                        plot_thresh=0., cmap='viridis', xspan=None, yspan=None,
                        titles=None, xlabel=None, plot_kwargs=None, fig=None):
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
    except:
        bins = [bins for i in range(nfilt)]
        pass
    if titles is None:
        titles = ['Band {0}'.format(i) for i in range(nfilt)]
    if xlabel is None:
        if x is None:
            xlabel = titles
        else:
            xlabel = ['Label' for i in range(nfilt)]
    else:
        xlabel = [xlabel for i in range(nfilt)]
    if offset is None:
        offset = np.ones(nfilt)

    # Compute posterior predictive SED magnitudes.
    mpred = get_seds(models[idxs.flatten()],
                     av=reds.flatten(), rv=dreds.flatten())
    mpred += 5. * np.log10(dists.flatten())[:, None]
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
        fig, axes = fig, axes = plt.subplots(nrows, ncols,
                                             figsize=(ncols * 6, nrows * 5))
    else:
        fig, axes = fig
        nrows, ncols = axes.shape
    ax = axes.flatten()
    # Plot offsets.
    for i in range(nfilt):
        # Compute selection ignoring current band.
        mtemp = np.array(mask)
        mtemp[:, i] = False
        s = (mask[:, i] & (np.sum(mtemp, axis=1) > 3) &
             (np.all(np.isfinite(magobs), axis=1)))
        # Compute weights from ignoring current band.
        lnl = np.array([phot_loglike(mo, me, mt, mp, dim_prior=dim_prior)
                        for mo, me, mt, mp in zip(magobs[s], mageobs[s],
                                                  mtemp[s], mpred[s])])
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
        ax[i].hist2d(xp, mp - mobs, bins=(bx, by), weights=w,
                     cmin=plot_thresh, cmap=cmap, **plot_kwargs)
        ax[i].set_xlabel(xlabel[i])
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(r'$\Delta\,$mag')
    # Clear other axes.
    for i in range(nfilt, nrows * ncols):
        ax[i].set_frame_on(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()

    return fig, axes


def photometric_offsets_2d(phot, err, mask, models, idxs, reds, dreds, dists,
                           x, y, flux=True, weights=None, bins=100,
                           offset=None, dim_prior=True, plot_thresh=10.,
                           cmap='coolwarm', clims=(-0.05, 0.05),
                           xspan=None, yspan=None, titles=None, show_off=True,
                           xlabel=None, ylabel=None, plot_kwargs=None,
                           fig=None):
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
    except:
        bins = [bins for i in range(nfilt)]
        pass
    if titles is None:
        titles = ['Band {0}'.format(i) for i in range(nfilt)]
    if show_off and offset is not None:
        titles = [t + ' ({:2.2}% offset)'.format(100. * (off - 1.))
                  for t, off in zip(titles, offset)]
    if xlabel is None:
        xlabel = 'X'
    if ylabel is None:
        ylabel = 'Y'
    if offset is None:
        offset = np.ones(nfilt)

    # Compute posterior predictive SED magnitudes.
    mpred = get_seds(models[idxs.flatten()],
                     av=reds.flatten(), rv=dreds.flatten())
    mpred += 5. * np.log10(dists.flatten())[:, None]
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
        fig, axes = fig, axes = plt.subplots(nrows, ncols,
                                             figsize=(ncols * 15, nrows * 12))
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
        bounds = (xcent[0], xcent[-1], ycent[0], ycent[-1])  # default size
        # Digitize values.
        xloc, yloc = np.digitize(x, xbins), np.digitize(y, ybins)
        # Compute selection ignoring current band.
        mtemp = np.array(mask)
        mtemp[:, i] = False
        s = (mask[:, i] & (np.sum(mtemp, axis=1) > 3) &
             (np.all(np.isfinite(magobs), axis=1)))
        # Compute weights from ignoring current band.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            lnl = np.array([phot_loglike(mo, me, mt, mp, dim_prior=dim_prior)
                            for mo, me, mt, mp in zip(magobs, mageobs,
                                                      mtemp, mpred)])
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
        img = ax[i].imshow(offset2d.T, origin='lower', extent=bounds,
                           vmin=clims[0], vmax=clims[1], aspect='auto',
                           cmap=cmap, **plot_kwargs)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(titles[i])
        plt.colorbar(img, ax=ax[i], label=r'$\Delta\,$mag')
    # Clear other axes.
    for i in range(nfilt, nrows * ncols):
        ax[i].set_frame_on(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()

    return fig, axes


def _hist2d(x, y, smooth=0.02, span=None, weights=None, levels=None,
            ax=None, color='gray', plot_datapoints=False, plot_density=True,
            plot_contours=True, no_fill_contours=False, fill_contours=True,
            contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
            **kwargs):
    """
    Internal function called by :meth:`cornerplot` used to generate a
    a 2-D histogram/contour of samples.

    Parameters
    ----------
    x : interable with shape `(nsamps,)`
       Sample positions in the first dimension.

    y : iterable with shape `(nsamps,)`
       Sample positions in the second dimension.

    span : iterable with shape `(ndim,)`, optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.99` (99% credible interval).

    weights : iterable with shape `(nsamps,)`
        Weights associated with the samples. Default is `None` (no weights).

    levels : iterable, optional
        The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

    ax : `~matplotlib.axes.Axes`, optional
        An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
        If not provided, a figure will be generated.

    color : str, optional
        The `~matplotlib`-style color used to draw lines and color cells
        and contours. Default is `'gray'`.

    plot_datapoints : bool, optional
        Whether to plot the individual data points. Default is `False`.

    plot_density : bool, optional
        Whether to draw the density colormap. Default is `True`.

    plot_contours : bool, optional
        Whether to draw the contours. Default is `True`.

    no_fill_contours : bool, optional
        Whether to add absolutely no filling to the contours. This differs
        from `fill_contours=False`, which still adds a white fill at the
        densest points. Default is `False`.

    fill_contours : bool, optional
        Whether to fill the contours. Default is `True`.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """

    if ax is None:
        ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.99 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # Initialize smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth, smooth]
    bins = []
    svalues = []
    for s in smooth:
        if isinstance(s, int_type):
            # If `s` is an integer, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(s)
            svalues.append(0.)
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results.
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T,
                    np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])
