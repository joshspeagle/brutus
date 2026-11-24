"""Tests of the per-model summary of the Monte Carlo prior draws."""

import numpy as np
from scipy.special import logsumexp

from brutus.analysis.individual import _mc_summary
from brutus.utils.math import LOG_ZERO


def test_mc_summary_matches_logsumexp_and_weight_concentration():
    rng = np.random.default_rng(2)
    # More models than one block, with spreads from flat to one dominant draw.
    lnp_mc = rng.normal(-50.0, 1.0, (40, 600)) * rng.uniform(0.0, 30.0, 600)
    lnp_mc[:, 5] = LOG_ZERO
    lnp_mc[3:, 6] = LOG_ZERO
    lnp, ess = _mc_summary(lnp_mc)
    w = np.exp(lnp_mc - lnp_mc.max(0))
    np.testing.assert_allclose(lnp, logsumexp(lnp_mc, 0) - np.log(40), rtol=1e-13)
    ok = np.arange(600) != 5
    np.testing.assert_allclose(ess[ok], (w.sum(0) ** 2 / (w**2).sum(0))[ok], rtol=1e-12)
    assert lnp[5] == LOG_ZERO and ess[5] == 0.0
    assert 1.0 <= ess[6] <= 3.0


def test_model_with_a_nan_draw_gets_zero_weight():
    lnp_mc = np.zeros((4, 3))
    lnp_mc[1, 1] = np.nan
    lnp_mc[:, 2] = np.inf
    lnp, ess = _mc_summary(lnp_mc)
    assert lnp[0] == 0.0 and ess[0] == 4.0
    assert lnp[1] == LOG_ZERO and ess[1] == 0.0
    assert lnp[2] == LOG_ZERO and ess[2] == 0.0
