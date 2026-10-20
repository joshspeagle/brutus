#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests of hist2d in brutus.plotting.utils."""

import logging
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet

from brutus.plotting.utils import hist2d
from brutus.utils.sampling import quantile


@pytest.fixture
def sample_data():
    """200 correlated 2-D Gaussian samples."""
    rng = np.random.RandomState(42)
    return rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 200).T


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _artists(ax):
    """Count the artists hist2d draws: point markers, filled contour sets,
    line contour sets and density meshes."""
    filled = [c for c in ax.collections if isinstance(c, ContourSet) and c.filled]
    lines = [c for c in ax.collections if isinstance(c, ContourSet) and not c.filled]
    mesh = [c for c in ax.collections if not isinstance(c, ContourSet)]
    return len(ax.lines), len(filled), len(lines), len(mesh)


@pytest.mark.parametrize(
    "options, expected",
    [
        # (points, filled contour sets, line contour sets, density meshes).
        # The first filled set is the white base fill under contours or density.
        ({}, (0, 2, 1, 0)),
        (dict(plot_datapoints=True), (1, 2, 1, 0)),
        (dict(plot_density=False), (0, 2, 1, 0)),
        (dict(plot_contours=False), (0, 1, 0, 1)),
        (dict(fill_contours=False), (0, 1, 1, 1)),
        (dict(no_fill_contours=True), (0, 1, 1, 0)),
        (dict(plot_density=False, plot_contours=False), (0, 0, 0, 0)),
        (
            dict(plot_datapoints=True, plot_density=False, plot_contours=False),
            (1, 0, 0, 0),
        ),
    ],
)
def test_plot_options_draw_the_matching_artists(sample_data, ax, options, expected):
    x, y = sample_data
    hist2d(x, y, ax=ax, **options)
    assert _artists(ax) == expected


def test_datapoints_are_the_samples_with_their_style(sample_data, ax):
    x, y = sample_data
    hist2d(x, y, ax=ax, plot_datapoints=True, data_kwargs={"marker": "s", "alpha": 0.3})
    (line,) = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    assert line.get_marker() == "s"
    assert line.get_alpha() == 0.3


def test_drawing_goes_to_the_given_axes(sample_data):
    x, y = sample_data
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    hist2d(x, y, ax=ax2)
    assert not ax1.collections
    assert ax2.collections
    plt.close(fig)


def test_explicit_span_sets_the_axis_limits(sample_data, ax):
    x, y = sample_data
    hist2d(x, y, ax=ax, span=[(-2.0, 2.0), (-1.0, 3.0)])
    assert ax.get_xlim() == (-2.0, 2.0)
    assert ax.get_ylim() == (-1.0, 3.0)


def test_fractional_span_sets_limits_to_central_weighted_quantiles(sample_data, ax):
    x, y = sample_data
    weights = np.random.RandomState(1).exponential(1.0, len(x))
    hist2d(x, y, ax=ax, weights=weights, span=[(-1.0, 1.0), 0.8])
    assert ax.get_xlim() == (-1.0, 1.0)
    np.testing.assert_allclose(
        ax.get_ylim(), quantile(y, [0.1, 0.9], weights=weights), rtol=1e-12
    )


def test_default_span_is_the_central_99_percent(sample_data, ax):
    x, y = sample_data
    hist2d(x, y, ax=ax)
    np.testing.assert_allclose(ax.get_xlim(), quantile(x, [0.005, 0.995]), rtol=1e-12)
    np.testing.assert_allclose(ax.get_ylim(), quantile(y, [0.005, 0.995]), rtol=1e-12)


@pytest.mark.parametrize("levels", [[0.5, 0.9], [0.1, 0.5, 0.9]])
def test_contour_levels_enclose_the_requested_mass(levels, ax):
    """With an integer `smooth` (bins, no smoothing), each contour level V is
    the density at which the mass in bins above it reaches the requested
    fraction: mass(H > V) <= level < mass(H >= V) + V."""
    x, y = np.random.RandomState(3).normal(size=(2, 5000))
    span = [(-3.0, 3.0), (-3.0, 3.0)]
    hist2d(x, y, ax=ax, smooth=10, span=span, levels=levels)
    H = np.histogram2d(x, y, bins=10, range=span)[0]
    (contours,) = [
        c for c in ax.collections if isinstance(c, ContourSet) and not c.filled
    ]
    V = np.asarray(contours.levels) / H.sum()
    H = H / H.sum()
    assert len(V) == len(levels)
    # A larger mass fraction encloses a lower density.
    for v, level in zip(V, sorted(levels, reverse=True)):
        assert H[H > v].sum() <= level < H[H >= v].sum() + v


def test_constant_data_warns_about_identical_limits(ax):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hist2d(np.ones(10), np.ones(10), ax=ax)
    messages = [str(warning.message) for warning in w]
    assert any("identical" in msg and "lims" in msg for msg in messages)


def test_too_few_points_logs_a_contour_warning(ax, caplog):
    with caplog.at_level(logging.WARNING):
        # Three points in three of 100 bins: every level maps to one count.
        hist2d(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), ax=ax, smooth=10)
    assert "Too few points to create valid contours." in caplog.messages


def test_span_with_wrong_length_raises(sample_data, ax):
    x, y = sample_data
    with pytest.raises(ValueError, match="Dimension mismatch"):
        hist2d(x, y, ax=ax, span=[0.95])


@pytest.mark.parametrize(
    "options",
    [
        dict(smooth=[0.02, 0.03]),
        dict(smooth=[15, 25]),
        dict(contour_kwargs={"linewidths": 2}, contourf_kwargs={"alpha": 0.6}),
    ],
)
def test_hist2d_options_run(sample_data, ax, options):
    """Smoke test: these options draw without raising."""
    assert hist2d(*sample_data, ax=ax, **options) is None


@pytest.mark.parametrize("n", [1, 2])
def test_hist2d_tiny_inputs_run(ax, n):
    """Smoke test: one or two points draw without raising."""
    x = np.arange(float(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert hist2d(x, x, ax=ax) is None
