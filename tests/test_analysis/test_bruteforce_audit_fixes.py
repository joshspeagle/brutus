#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for verified audit fixes in brutus.analysis.individual.

Covers:
1. _setup must not mutate a user-supplied lnprior array in place.
2. The parallax model-selection gate must use the MARGINAL scale
   uncertainty sqrt(cov[0,0]) rather than the conditional 1/sqrt(icov[0,0]).
3. logpost_grid must honor apply_av_prior=False.
4. A dust-map prior (dustfile) without a sky position must raise a clear
   ValueError instead of crashing deep inside the map query.
5. A pathlib.Path dustfile must be auto-loaded exactly like a str path
   rather than being mistaken for a pre-loaded dust-map object.
"""

import numpy as np
import pytest

from brutus.analysis.individual import BruteForce
from brutus.core.individual import StarGrid

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mock_grid():
    """Small mock grid (27 models, 5 filters) mirroring test_bruteforce.py."""
    nmodels = 27
    nfilters = 5

    models = np.zeros((nmodels, nfilters, 3))
    labels = np.zeros(nmodels, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])
    params = np.zeros(
        nmodels,
        dtype=[
            ("mass", "f4"),
            ("radius", "f4"),
            ("logg", "f4"),
            ("Teff", "f4"),
            ("Mr", "f4"),
        ],
    )

    idx = 0
    for m in [0.5, 1.0, 2.0]:
        for j, e in enumerate([200, 350, 450]):
            for z in [-1.0, 0.0, 0.5]:
                base_mag = 15.0 - 2.5 * np.log10(m)
                models[idx, :, 0] = base_mag + np.array([0, -0.2, -0.3, -0.4, -0.45])
                models[idx, :, 1] = np.array([1.5, 1.2, 1.0, 0.8, 0.7])
                models[idx, :, 2] = np.array([0.15, 0.12, 0.10, 0.08, 0.07])
                labels[idx] = (m, e, z)
                params[idx] = (
                    m * (1 - 0.05 * j),
                    m**0.8 * (1 + 0.05 * j),
                    4.4 - 0.1 * j,
                    5777 * m**0.4,
                    5.0 - 2.5 * np.log10(m),
                )
                idx += 1

    return StarGrid(models, labels, params, filters=["g", "r", "i", "z", "y"])


@pytest.fixture(scope="module")
def fitter(mock_grid):
    return BruteForce(mock_grid, verbose=False)


@pytest.fixture
def observation(mock_grid):
    """Noiseless synthetic observation from mock grid model 13."""
    true_flux = 10 ** (-0.4 * mock_grid.models[13, :, 0])
    flux_err = true_flux * 0.05
    mask = np.ones(len(true_flux), dtype=bool)
    return true_flux, flux_err, mask


def _make_loglike_results(scales, cov_list, ndim=5):
    """Build a synthetic loglike_grid(return_vals=True) tuple."""
    n = len(scales)
    lnlike = np.zeros(n)
    chi2 = np.zeros(n)
    avs = np.full(n, 0.5)
    rvs = np.full(n, 3.3)
    icovs = np.array([np.linalg.inv(c) for c in cov_list])
    return (lnlike, ndim, chi2, np.asarray(scales, dtype=float), avs, rvs, icovs)


# ============================================================================
# Fix 1: _setup must not mutate a caller-supplied lnprior
# ============================================================================


class TestSetupLnpriorNotMutated:
    def test_user_lnprior_array_unchanged(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)
        user_lnprior = np.zeros(fitter.nmodels)

        res = fitter._setup(
            data,
            data_err,
            data_mask,
            lnprior=user_lnprior,
            data_coords=(120.0, 45.0),
            apply_agewt=True,
            apply_grad=True,
        )
        lnprior_out = res[3]

        # Grid-gradient corrections were applied (output differs from input)
        assert not np.allclose(lnprior_out, 0.0)
        # ... but never in place on the caller's array
        assert lnprior_out is not user_lnprior
        np.testing.assert_array_equal(user_lnprior, np.zeros(fitter.nmodels))

    def test_repeated_setup_calls_do_not_compound(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)
        user_lnprior = np.zeros(fitter.nmodels)

        kwargs = dict(
            lnprior=user_lnprior,
            data_coords=(120.0, 45.0),
            apply_agewt=True,
            apply_grad=True,
        )
        out1 = fitter._setup(data, data_err, data_mask, **kwargs)[3]
        out2 = fitter._setup(data, data_err, data_mask, **kwargs)[3]

        # Corrections must be applied once per call, not accumulated across
        # calls (double-strength age/grid weighting on the second run).
        np.testing.assert_allclose(out1, out2)


# ============================================================================
# Fix 2: parallax model selection uses marginal scale uncertainty
# ============================================================================


class TestParallaxSelectionMarginalError:
    def test_degenerate_model_kept_with_marginal_error(self, fitter):
        """A model 2 marginal-sigma from the parallax scale must be kept.

        With a strong scale-A(V) degeneracy (rho=0.99) the conditional std
        is ~7x smaller than the marginal std. The old conditional gate gave
        this model a selection penalty of ~98 (pruned at log(1e-3)=-6.9);
        the marginal gate gives ~2 (kept).
        """
        sig_s, sig_a, sig_r, rho = 1.0, 1.0, 0.1, 0.99
        cov = np.array(
            [
                [sig_s**2, rho * sig_s * sig_a, 0.0],
                [rho * sig_s * sig_a, sig_a**2, 0.0],
                [0.0, 0.0, sig_r**2],
            ]
        )
        # Model 0: ML scale exactly at the parallax-implied scale.
        # Model 1: ML scale 2 marginal-sigma away.
        results = _make_loglike_results([1.0, 3.0], [cov, cov])

        out = fitter.logpost_grid(
            results,
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert 0 in sel
        assert 1 in sel, (
            "Model 2 marginal-sigma from the parallax scale was pruned: "
            "selection gate is using the conditional (not marginal) "
            "scale uncertainty"
        )

    def test_inconsistent_model_still_pruned(self, fitter):
        """With uncorrelated errors the gate must still prune bad models."""
        cov = np.diag([0.02**2, 0.1**2, 0.1**2])
        results = _make_loglike_results([1.0, 3.0], [cov, cov])

        out = fitter.logpost_grid(
            results,
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert 0 in sel
        assert 1 not in sel

    def test_diagonal_precision_matches_conditional(self, fitter):
        """For a diagonal precision matrix marginal == conditional, so the
        selection must be identical to the old-code reference computation."""
        cov = np.diag([0.05**2, 0.1**2, 0.1**2])
        icov = np.linalg.inv(cov)
        # Reference (old code): conditional error from icov[0,0].
        cond_err = 1.0 / np.sqrt(icov[0, 0])
        # New code marginal error via cofactor formula.
        minor00 = icov[1, 1] * icov[2, 2] - icov[1, 2] ** 2
        marg_err = np.sqrt(minor00 / np.linalg.det(icov))
        np.testing.assert_allclose(marg_err, cond_err, rtol=1e-12)

    def test_singular_precision_does_not_crash(self, fitter):
        """Degenerate (all-zero) precision matrices fall back gracefully."""
        cov = np.diag([0.05**2, 0.1**2, 0.1**2])
        results = list(_make_loglike_results([1.0, 1.0], [cov, cov]))
        icovs = results[6]
        icovs[1] = 0.0  # singular: uninformative error is used
        results[6] = icovs

        out = fitter.logpost_grid(
            tuple(results),
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert len(sel) > 0


# ============================================================================
# Fixes 3+4: apply_av_prior flag and dustfile-requires-coord guard
# ============================================================================


class TestDustPriorGating:
    def _mock_dust(self, calls):
        def lndust(avs, dustmap, coord, distance=None):
            calls.append(np.size(avs))
            return np.zeros_like(avs)

        return lndust

    def test_apply_av_prior_false_skips_dust_prior(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=False,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert calls == [], "dust prior was applied despite apply_av_prior=False"

    def test_apply_av_prior_true_applies_dust_prior(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=True,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert len(calls) == 1

    def test_dustfile_without_coord_raises(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        with pytest.raises(ValueError, match="coord"):
            fitter.logpost_grid(
                like,
                coord=None,
                dustfile=object(),
                lndustprior=self._mock_dust([]),
                Nmc_prior=5,
                wt_thresh=0.1,
            )

    def test_dustfile_without_coord_ok_when_prior_disabled(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=None,
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=False,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert calls == []

    def test_dustfile_accepts_pathlib_path(
        self, fitter, observation, monkeypatch, tmp_path
    ):
        """A pathlib.Path dustfile must trigger the same auto-load as a str
        path; previously the isinstance(dustfile, str) gate let a Path fall
        through as a "pre-loaded map" and crash downstream."""
        import brutus.dust as dust_mod

        loaded = []

        class FakeBayestar:
            def __init__(self, dustfile=None, **kwargs):
                loaded.append(dustfile)

        monkeypatch.setattr(dust_mod, "Bayestar", FakeBayestar)

        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        dust_path = tmp_path / "bayestar_fake.h5"
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=dust_path,
            lndustprior=self._mock_dust(calls),
            apply_av_prior=True,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert loaded == [dust_path], "Path dustfile was not auto-loaded"
        assert len(calls) == 1

    def test_setup_dustfile_without_data_coords_raises(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        def custom_gal(distance, coord, labels=None):
            return np.zeros_like(distance)

        with pytest.raises(ValueError, match="data_coords"):
            fitter._setup(
                data,
                data_err,
                data_mask,
                lngalprior=custom_gal,  # bypass the galactic-prior coord check
                dustfile=object(),
                data_coords=None,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
