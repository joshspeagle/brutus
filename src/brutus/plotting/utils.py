#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common plotting utilities for brutus visualization functions.

This module provides low-level plotting utilities shared across
multiple plotting functions.
"""

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from scipy.ndimage import gaussian_filter as norm_kde

from ..utils.sampling import quantile

str_type = str
float_type = float
int_type = int

__all__ = ["hist2d"]


def hist2d(
    x,
    y,
    smooth=0.02,
    span=None,
    weights=None,
    levels=None,
    ax=None,
    color="gray",
    plot_datapoints=False,
    plot_density=True,
    plot_contours=True,
    no_fill_contours=False,
    fill_contours=True,
    contour_kwargs=None,
    contourf_kwargs=None,
    data_kwargs=None,
    **kwargs,
):
    """
    Internal function used to generate a 2-D histogram/contour of samples.

    This is the refactored version of _hist2d from the original plotting.py.

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
        except (TypeError, ValueError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)]
    )

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for level in levels] + [rgba_color]
    for i, level in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # Initialize smoothing.
    if isinstance(smooth, int_type) or isinstance(smooth, float_type):
        smooth = [smooth, smooth]
    bins = []
    svalues = []
    for s in smooth:
        if isinstance(s, int_type):
            # If `s` is an integer, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(s)
            svalues.append(0.0)
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2.0 / s)))
            svalues.append(2.0)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            range=list(map(np.sort, span)),
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range."
        )

    # Smooth the results.
    if not np.all(svalues == 0.0):
        H = norm_kde(H, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)

    # Check for degenerate case (all zeros or no variation)
    if sm[-1] == 0 or H.max() == 0:
        logging.warning("Data has no density variation; skipping contour plots.")
        V = np.array([0])  # Minimal valid contour level
        degenerate_data = True
    else:
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except IndexError:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        if np.any(m) and plot_contours:
            logging.warning("Too few points to create valid contours.")

        # Fix infinite loop when all values are zero or identical
        safety_counter = 0
        max_iterations = 100  # Prevent infinite loop
        while np.any(m) and safety_counter < max_iterations:
            idx = np.where(m)[0][0]
            if V[idx] == 0:
                # If value is zero, add small increment instead of multiplying
                V[idx] = 1e-10 * (safety_counter + 1)
            else:
                V[idx] *= 1.0 - 1e-4
            m = np.diff(V) == 0
            safety_counter += 1

        if safety_counter >= max_iterations:
            logging.warning(
                "Could not resolve duplicate contour levels after %d iterations.",
                max_iterations,
            )

        V.sort()
        degenerate_data = False

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
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )

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
    if (plot_contours or plot_density) and not no_fill_contours and not degenerate_data:
        # Ensure contour levels are valid and increasing
        base_levels = [V.min(), H.max()]
        if base_levels[0] >= base_levels[1]:
            # If levels are not increasing, create a minimal valid range
            if H.max() > 0:
                base_levels = [H.max() * 0.9, H.max()]
            else:
                base_levels = [0, 1e-10]
        ax.contourf(X2, Y2, H2.T, base_levels, cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours and not degenerate_data:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
        ax.contourf(
            X2,
            Y2,
            H2.T,
            np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
            **contourf_kwargs,
        )

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills. Use sqrt scaling to make low-density bins more visible.
    elif plot_density:
        H_scaled = np.sqrt(np.maximum(H, 0))
        ax.pcolor(X, Y, H_scaled.max() - H_scaled.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours and not degenerate_data:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])
