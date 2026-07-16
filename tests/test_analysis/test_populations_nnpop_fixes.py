#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for the 2026-07 audit fixes to analysis.populations
(group: nnpop).

Covers:
- corr_params plumbing through isochrone_population_loglike
- apply_isochrone_mixture_model via np.logaddexp (equivalence + edge cases)
- generate_isochrone_population_grid reusing SMF-invariant primary SEDs
  (equivalence with the old per-slice implementation)
- duck-typed StellarPop fallback (objects without get_seds_smf_grid)
"""

import warnings

import numpy as np
import pytest

from brutus.analysis.populations import (
    apply_isochrone_mixture_model,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
)


@pytest.fixture(scope="module")
def real_stellarpop():
    from conftest import find_brutus_data_file

    from brutus.core.populations import Isochrone, StellarPop

    iso_file = find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")
    nn_file = find_brutus_data_file("nnMIST_BC.h5")
    if nn_file is None:
        nn_file = find_brutus_data_file("nn_c3k.h5")
    if iso_file is None or nn_file is None:
        pytest.skip("MIST isochrone / neural network data files not available")

    iso = Isochrone(mistfile=iso_file, verbose=False)
    return StellarPop(
        isochrone=iso,
        filters=["SDSS_g", "SDSS_r", "SDSS_i"],
        nnfile=nn_file,
        verbose=False,
    )


class RecordingStellarPop:
    """Duck-typed StellarPop (get_seds only) that records call kwargs."""

    def __init__(self, n_filters=3):
        self.n_filters = n_filters
        self.calls = []

    def get_seds(
        self,
        feh=0.0,
        loga=8.5,
        av=0.0,
        rv=3.3,
        eep=None,
        binary_fraction=0.0,
        dist=1000.0,
        mini_bound=0.08,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
    ):
        self.calls.append({"corr_params": corr_params, "smf": binary_fraction})
        eep = np.asarray(eep)
        n = len(eep)
        masses = 0.5 + 0.001 * (eep - eep.min())
        sed = np.full((n, self.n_filters), 10.0)
        if binary_fraction > 0:
            is_ms = eep <= eep_binary_max
            sed[is_ms] -= 2.5 * np.log10(1.0 + binary_fraction**3.5)
        params = {"mini": masses}
        params2 = {"mini": np.where(eep <= eep_binary_max, masses, np.nan)}
        return sed, params, params2


class TestCorrParamsPlumbing:
    """corr_params must flow through the MCMC entry point to get_seds."""

    def test_corr_params_forwarded(self):
        pop = RecordingStellarPop()
        rng = np.random.default_rng(0)
        flux = rng.uniform(0.5, 1.5, (4, 3))
        err = np.full((4, 3), 0.1)
        corr = (0.05, -0.05, 25.0, 0.4)
        lnl = isochrone_population_loglike(
            [0.0, 9.0, 0.1, 3.1, 1000.0, 0.1],
            pop,
            flux,
            err,
            smf_grid=np.array([0.0, 0.5]),
            eep_grid=np.linspace(300, 400, 10),
            corr_params=corr,
        )
        assert np.isfinite(lnl)
        assert len(pop.calls) == 2
        for call in pop.calls:
            assert call["corr_params"] == corr

    def test_grid_generation_forwards_corr_params(self, real_stellarpop):
        """Different corr_params must actually change the generated grid."""
        kw = dict(
            feh=-0.3,
            loga=9.5,
            av=0.1,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0]),
            eep_grid=np.linspace(250, 400, 20),
        )
        g_def = generate_isochrone_population_grid(
            real_stellarpop, corr_params=None, **kw
        )
        g_alt = generate_isochrone_population_grid(
            real_stellarpop, corr_params=(0.3, -0.3, 30.0, 0.0), **kw
        )
        assert not np.allclose(g_def["photometry"], g_alt["photometry"], equal_nan=True)


def _reference_mixture(lnl_cluster, lnl_outlier, cluster_prob, field_fraction):
    """Old hand-rolled two-term log-sum-exp (pre-audit implementation)."""
    ln_cluster_weight = np.log(cluster_prob * (1.0 - field_fraction))
    ln_outlier_weight = np.log(1.0 - cluster_prob * (1.0 - field_fraction))
    cluster_term = lnl_cluster + ln_cluster_weight
    outlier_term = lnl_outlier + ln_outlier_weight
    max_term = np.maximum(cluster_term, outlier_term)
    with np.errstate(invalid="ignore"):
        return max_term + np.log(
            np.exp(cluster_term - max_term) + np.exp(outlier_term - max_term)
        )


class TestMixtureLogaddexp:
    """np.logaddexp path must match the old expression on finite values."""

    def test_equivalence_random(self):
        rng = np.random.default_rng(1)
        a = rng.normal(-50, 30, (500, 40))
        b = rng.normal(-40, 5, (500, 40))
        old = _reference_mixture(a, b, 0.95, 0.1)
        new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-13)

    def test_nan_rows_preserved(self):
        """NaN cluster likelihoods (invalid models) must stay non-finite so
        marginalization can drop them."""
        a = np.array([[np.nan, -10.0]])
        b = np.array([[-5.0, -5.0]])
        with np.errstate(invalid="ignore"):
            new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        assert np.isnan(new[0, 0])
        assert np.isfinite(new[0, 1])

    def test_both_neginf_gives_neginf(self):
        """logaddexp(-inf, -inf) = -inf (the old expression produced NaN);
        either way the point is dropped, but -inf is the correct value."""
        a = np.full((1, 2), -np.inf)
        b = np.full((1, 2), -np.inf)
        new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        assert np.all(np.isneginf(new))


def _reference_population_grid(
    stellarpop,
    feh,
    loga,
    av,
    rv,
    dist,
    smf_grid,
    eep_grid,
    mini_bound=0.08,
    eep_binary_max=480.0,
    corr_params=None,
):
    """Pre-audit generate_isochrone_population_grid: one full get_seds call
    per SMF slice (primary recomputed every time)."""
    smf_grid = np.asarray(smf_grid)
    eep_grid = np.asarray(eep_grid)
    if len(smf_grid) > 1:
        smf_jacobians = np.gradient(smf_grid)
    else:
        smf_jacobians = np.array([1.0])
    all_photometry, all_masses, all_smf_values = [], [], []
    all_mass_jacobians, all_smf_jacobians = [], []
    post_ms_stored = False
    total_smf_measure = np.sum(smf_jacobians)
    for i, smf in enumerate(smf_grid):
        sed, params1, params2 = stellarpop.get_seds(
            feh=feh,
            loga=loga,
            av=av,
            rv=rv,
            eep=eep_grid,
            binary_fraction=smf,
            dist=dist,
            mini_bound=mini_bound,
            eep_binary_max=eep_binary_max,
            corr_params=corr_params,
        )
        masses = params1["mini"]
        mass_jacobians = np.gradient(masses) if len(masses) > 1 else np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_valid = mass_jacobians > 0.0
            slice_mask = base_valid & (eep_grid <= eep_binary_max)
            post_ms_mask = base_valid & (eep_grid > eep_binary_max)
        blocks = [(slice_mask, smf_jacobians[i])]
        if not post_ms_stored and np.any(post_ms_mask):
            blocks.append((post_ms_mask, total_smf_measure))
            post_ms_stored = True
        for block_mask, smf_jac in blocks:
            valid_indices = np.where(block_mask)[0]
            if len(valid_indices) == 0:
                continue
            all_photometry.append(10 ** (-0.4 * sed[valid_indices]))
            all_masses.append(masses[valid_indices])
            all_smf_values.append(np.full(len(valid_indices), smf))
            all_mass_jacobians.append(mass_jacobians[valid_indices])
            all_smf_jacobians.append(np.full(len(valid_indices), smf_jac))
    return {
        "photometry": np.vstack(all_photometry),
        "masses": np.concatenate(all_masses),
        "smf_values": np.concatenate(all_smf_values),
        "mass_jacobians": np.concatenate(all_mass_jacobians),
        "smf_jacobians": np.concatenate(all_smf_jacobians),
    }


class TestPrimaryCaching:
    """The cached-primary grid must equal the per-slice reference exactly."""

    def test_equivalence_with_reference(self, real_stellarpop):
        kw = dict(
            feh=0.05,
            loga=9.4,
            av=0.15,
            rv=3.2,
            dist=1200.0,
            smf_grid=np.linspace(0.0, 1.0, 11),
            eep_grid=np.linspace(202.0, 808.0, 120),
        )
        ref = _reference_population_grid(real_stellarpop, **kw)
        new = generate_isochrone_population_grid(real_stellarpop, **kw)
        for key in ref:
            np.testing.assert_array_equal(new[key], ref[key], err_msg=key)

    def test_ducktyped_fallback_without_smf_grid_method(self):
        """Population objects exposing only get_seds must keep working."""
        pop = RecordingStellarPop()
        assert not hasattr(pop, "get_seds_smf_grid")
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 1.0]),
            eep_grid=np.linspace(300, 400, 10),
        )
        assert grid["photometry"].shape[0] > 0
        assert len(pop.calls) == 2  # one get_seds call per slice

    def test_primary_computed_once(self, real_stellarpop):
        """The isochrone must be interpolated once, not once per SMF value."""
        calls = {"n": 0}
        iso = real_stellarpop.isochrone
        orig = iso.get_predictions

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        iso.get_predictions = counting
        try:
            generate_isochrone_population_grid(
                real_stellarpop,
                feh=0.0,
                loga=9.5,
                av=0.1,
                rv=3.1,
                dist=1000.0,
                smf_grid=np.linspace(0.0, 1.0, 5),
                eep_grid=np.linspace(300.0, 500.0, 20),
            )
        finally:
            iso.get_predictions = orig
        # 1 primary call + one secondary call per 0 < smf < 1 slice (3 of 5).
        # The old implementation used 5 primary + 3 secondary = 8 calls.
        assert calls["n"] == 4
