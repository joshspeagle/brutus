"""
Combined summary plot: corner plot + posterior predictive SED in one figure.

Places the posterior predictive SED in the upper-right triangle of the
corner plot to eliminate white space and provide a single diagnostic view.
"""

import numpy as np

from .corner import cornerplot
from .sed import posterior_predictive


def summary_plot(
    models,
    idxs,
    reds,
    dreds,
    dists,
    params,
    data=None,
    data_err=None,
    data_mask=None,
    offset=None,
    coord=None,
    parallax=None,
    parallax_err=None,
    labels=None,
    weights=None,
    show_titles=True,
    color="black",
    vcolor="black",
    pcolor="black",
    R_solar=8.2,
    Z_solar=0.025,
    figsize=None,
    **corner_kwargs,
):
    """
    Generate a combined corner plot with posterior predictive SED inset.

    Creates a corner plot of the stellar parameter posteriors with the
    posterior predictive SED placed in the upper-right triangle, making
    efficient use of the figure space.

    Parameters
    ----------
    models : `~numpy.ndarray` of shape `(Nmodels, Nfilts, Ncoeffs)`
        Model magnitude polynomial coefficients.

    idxs : `~numpy.ndarray` of shape `(Nsamps)`
        Resampled model indices.

    reds : `~numpy.ndarray` of shape `(Nsamps)`
        A(V) samples.

    dreds : `~numpy.ndarray` of shape `(Nsamps)`
        R(V) samples.

    dists : `~numpy.ndarray` of shape `(Nsamps)`
        Distance samples in kpc.

    params : structured `~numpy.ndarray` with shape `(Nmodels,)`
        Model parameters for the corner plot axes.

    data : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Observed flux data for the posterior predictive.

    data_err : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Observed flux errors.

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Binary mask for valid bands.

    offset : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets.

    coord : tuple, optional
        Galactic (l, b) coordinates in degrees.

    parallax : float, optional
        Parallax in mas.

    parallax_err : float, optional
        Parallax error in mas.

    labels : list, optional
        Filter names for the SED plot x-axis.

    weights : `~numpy.ndarray`, optional
        Importance weights for the samples.

    show_titles : bool, optional
        Whether to show parameter titles on the corner plot.

    color : str, optional
        Color for the corner plot. Default is ``'black'``.

    vcolor : str, optional
        Color for the SED violin plots. Default is ``'black'``.

    pcolor : str, optional
        Color for the SED data points. Default is ``'black'``.

    R_solar : float, optional
        Solar galactocentric radius in kpc. Default is 8.2.

    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.

    figsize : tuple, optional
        Figure size. If None, determined automatically.

    **corner_kwargs
        Additional keyword arguments passed to ``cornerplot``.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The combined figure.
    axes : `~numpy.ndarray`
        The corner plot axes array.
    ax_sed : `~matplotlib.axes.Axes`
        The posterior predictive SED axes.
    """

    # Build parallax kwargs
    plx_kwargs = {}
    applied_parallax = False
    if parallax is not None and parallax_err is not None:
        if np.isfinite(parallax) and np.isfinite(parallax_err):
            plx_kwargs["parallax"] = parallax
            plx_kwargs["parallax_err"] = parallax_err
            applied_parallax = True

    # Generate the corner plot
    fig, axes = cornerplot(
        idxs,
        (dists, reds, dreds),
        params,
        coord=coord,
        applied_parallax=applied_parallax,
        weights=weights,
        show_titles=show_titles,
        color=color,
        R_solar=R_solar,
        Z_solar=Z_solar,
        **plx_kwargs,
        **corner_kwargs,
    )

    # Find the upper-right triangle region for the SED inset.
    # The corner plot has ndim × ndim axes; upper-right is empty.
    ndim = axes.shape[0]

    # Place the SED in the upper-right block.
    # Use gridspec coordinates spanning the top rows and right columns.
    # The upper-right quadrant spans rows 0:ndim//2, cols ndim//2:ndim
    # For a nice layout, span from row 0 to row ndim//2-1, col ndim//2 to ndim-1
    r_start = 0
    r_end = max(ndim // 2, 2)
    c_start = r_end
    c_end = ndim

    # Get the position of the corner subplot grid to place the SED
    # by reading the bounding box of the relevant axes.
    # Add padding to avoid label collisions with the corner plot.
    pos_tl = axes[r_start, c_start].get_position()
    pos_br = axes[r_end - 1, c_end - 1].get_position()

    # Shrink and offset the SED inset to add padding on the left and
    # bottom edges (where corner plot labels live).
    pad_left = 0.06
    pad_bottom = 0.18
    shrink = 0.80  # shrink to 80% of available space

    full_left = pos_tl.x0
    full_bottom = pos_br.y0
    full_width = pos_br.x1 - pos_tl.x0
    full_height = pos_tl.y1 - pos_br.y0

    left = full_left + pad_left * full_width
    bottom = full_bottom + pad_bottom * full_height
    width = full_width * shrink
    height = full_height * shrink

    ax_sed = fig.add_axes([left, bottom, width, height])

    # Save the corner plot's subplot params before calling posterior_predictive,
    # which calls plt.tight_layout() and would destroy the spacing.
    saved_params = fig.subplotpars
    saved_left = saved_params.left
    saved_right = saved_params.right
    saved_top = saved_params.top
    saved_bottom = saved_params.bottom
    saved_wspace = saved_params.wspace
    saved_hspace = saved_params.hspace

    # Generate the posterior predictive SED on the inset axes
    posterior_predictive(
        models,
        idxs=idxs,
        reds=reds,
        dreds=dreds,
        dists=dists,
        weights=weights,
        data=data,
        data_err=data_err,
        data_mask=data_mask,
        offset=offset,
        vcolor=vcolor,
        pcolor=pcolor,
        labels=labels,
        fig=(fig, ax_sed),
    )

    ax_sed.set_title("Posterior Predictive SED", fontsize=9)

    # Restore the corner plot's subplot params (undoes plt.tight_layout())
    fig.subplots_adjust(
        left=saved_left,
        right=saved_right,
        top=saved_top,
        bottom=saved_bottom,
        wspace=saved_wspace,
        hspace=saved_hspace,
    )

    # Re-set the SED axes position (it was computed before tight_layout)
    ax_sed.set_position([left, bottom, width, height])

    return fig, axes, ax_sed
