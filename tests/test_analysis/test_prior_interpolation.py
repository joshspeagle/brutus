"""Tests of the interpolation of the Galactic prior across a model's draws."""

import numpy as np
import pytest

from brutus.analysis.individual import (
    PRIOR_INTERP_TOL,
    _galactic_prior_at_draws,
    _interpolate_prior,
)
from brutus.priors import logp_galactic_structure
from brutus.utils.math import LOG_ZERO


def test_quartic_in_ln_distance_is_interpolated_exactly_and_flagged_by_its_error():
    rng = np.random.default_rng(1)
    eta = rng.normal(0.0, 0.1, (50, 3))
    lo, hi = eta.min(0), eta.max(0)
    nodes = (
        0.5 * (hi + lo) + 0.5 * (hi - lo) * np.cos(np.pi * np.arange(5) / 4)[:, None]
    )
    poly = np.array(
        [[1.0, -2.0, 0.5, 3.0, 4.0], [0.0, 0.0, 1.0, 2.0, 0.0], [0.0, 0, 0, 0, 5.0]]
    )

    def f(x, k):
        return np.polyval(poly[k], x)

    values = np.array([f(nodes[:, k], k) for k in range(3)]).T
    out, err = _interpolate_prior(eta, lo, hi, values)
    expect = np.array([f(eta[:, k], k) for k in range(3)]).T
    np.testing.assert_allclose(out, expect, rtol=1e-10, atol=1e-12)
    # A quadratic and a constant are exact in the 3-node interpolant too,
    # a quartic is not. np.polyval takes the highest power first.
    assert err[2] == pytest.approx(0.0, abs=1e-12)
    assert err[1] == pytest.approx(0.0, abs=1e-12)
    assert err[0] > 1e-6


def test_node_at_zero_probability_sends_the_model_to_the_exact_prior():
    eta = np.linspace(-0.1, 0.1, 50)[:, None]
    values = np.zeros((5, 1))
    values[3] = LOG_ZERO
    _, err = _interpolate_prior(eta, eta.min(0), eta.max(0), values)
    assert np.isinf(err[0])


@pytest.mark.parametrize(
    "coord, center, width",
    [
        ((0.0, 4.0), 8.3, 0.4),  # across the bar and its taper
        ((90.0, 60.0), 6.5, 0.2),  # across the thin-disk taper at |z| 5-7 kpc
        ((180.0, 20.0), 0.8, 0.02),  # a typical narrow spread
    ],
)
@pytest.mark.parametrize(
    "motion", [{}, {"pm": np.array([-3.0, 2.0]), "pm_cov": np.eye(2) * 1e-3}]
)
def test_interpolated_prior_matches_the_exact_one_at_the_draws(
    coord, center, width, motion
):
    rng = np.random.default_rng(4)
    n = 40
    centers = center * np.exp(rng.normal(0, 3 * width, n))
    dists = centers * np.exp(rng.normal(0, width, (50, n)))
    labels = {"feh": rng.uniform(-1.5, 0.3, n), "loga": rng.uniform(9.0, 10.1, n)}
    exact = logp_galactic_structure(dists, coord, **labels, **motion)
    approx = _galactic_prior_at_draws(
        logp_galactic_structure, dists, coord, labels, motion
    )
    finite = exact > LOG_ZERO / 2
    np.testing.assert_array_equal(approx[~finite], exact[~finite])
    # The 3-node estimate bounds the 5-node error, which is far smaller.
    assert np.max(np.abs(approx - exact)[finite]) < PRIOR_INTERP_TOL


def test_few_draws_use_the_exact_prior():
    dists = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    labels = {"feh": np.zeros(2), "loga": np.full(2, 9.5)}
    exact = logp_galactic_structure(dists, (90.0, 30.0), **labels)
    approx = _galactic_prior_at_draws(
        logp_galactic_structure, dists, (90.0, 30.0), labels, {}
    )
    np.testing.assert_array_equal(approx, exact)
