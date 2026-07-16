#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for the 2026-07 audit fixes to isochrone population fitting.

Covers:
- SMF grid actually reaching StellarPop (binary_fraction plumbing)
- primary-only SEDs preserved when a binary companion is invalid
- equal-mass binaries restricted to eep <= eep_binary_max
- post-main-sequence models carrying the full SMF measure
- NaN flux / zero error auto-masking in the cluster likelihood
- normalization of the (mass, SMF) integration measure
- dimensionally consistent uniform outlier model
"""

import numpy as np
import pytest

from brutus.analysis.populations import (
    compute_isochrone_cluster_loglike,
    generate_isochrone_population_grid,
    marginalize_isochrone_grid,
)
from brutus.utils.photometry import phot_loglike, uniform_outlier_loglike


@pytest.fixture(scope="module")
def real_stellarpop():
    """Load real MIST StellarPop with neural network once for module tests."""
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


class MockStellarPop:
    """Duck-typed StellarPop whose SED brightens with binary_fraction on the
    main sequence only, mirroring the real model's structure."""

    def __init__(self, n_filters=3):
        self.n_filters = n_filters

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
        eep = np.asarray(eep)
        n = len(eep)
        # Monotone mass-EEP relation
        masses = 0.5 + 0.001 * (eep - eep.min())
        base = np.full((n, self.n_filters), 10.0)
        # Binaries brighten MS stars only
        is_ms = eep <= eep_binary_max
        sed = base.copy()
        if binary_fraction > 0:
            sed[is_ms] -= 2.5 * np.log10(1.0 + binary_fraction**3.5)
        params = {"mini": masses}
        params2 = {"mini": np.where(is_ms, masses * binary_fraction, np.nan)}
        return sed, params, params2


class TestSMFPlumbing:
    """The smf grid must actually change the generated photometry."""

    def test_smf_slices_differ(self):
        pop = MockStellarPop()
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
        phot0 = grid["photometry"][grid["smf_values"] == 0.0]
        phot1 = grid["photometry"][grid["smf_values"] == 1.0]
        assert phot0.shape == phot1.shape
        # equal-mass binaries should be exactly twice as bright in flux
        np.testing.assert_allclose(phot1 / phot0, 2.0, rtol=1e-10)

    def test_real_smf_slices_differ(self, real_stellarpop):
        """End-to-end through the real StellarPop: smf=1 doubles MS flux."""
        eep_grid = np.linspace(300, 400, 8)
        grid = generate_isochrone_population_grid(
            real_stellarpop,
            feh=0.0,
            loga=9.5,
            av=0.1,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 1.0]),
            eep_grid=eep_grid,
        )
        phot0 = grid["photometry"][grid["smf_values"] == 0.0]
        phot1 = grid["photometry"][grid["smf_values"] == 1.0]
        # Compare rows for matching masses (both slices share the mass grid)
        m0 = grid["masses"][grid["smf_values"] == 0.0]
        m1 = grid["masses"][grid["smf_values"] == 1.0]
        shared = np.intersect1d(m0, m1)
        assert shared.size > 0, "expected overlapping MS masses across slices"
        f0 = phot0[np.isin(m0, shared)]
        f1 = phot1[np.isin(m1, shared)]
        finite = np.isfinite(f0) & np.isfinite(f1)
        assert finite.any()
        np.testing.assert_allclose(f1[finite] / f0[finite], 2.0, rtol=1e-6)


class TestPostMSMeasure:
    """Post-MS (giant) models must carry the full SMF measure."""

    def test_giant_weight_matches_ms_weight(self):
        pop = MockStellarPop()
        eep_grid = np.linspace(400, 560, 17)  # spans eep_binary_max=480
        smf_grid = np.linspace(0.0, 1.0, 21)
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=smf_grid,
            eep_grid=eep_grid,
        )
        # Total integration weight accumulated at a fixed mass, MS vs post-MS
        w = grid["mass_jacobians"] * grid["smf_jacobians"]
        masses = grid["masses"]
        ms_mass = 0.5 + 0.001 * (440.0 - 400.0)  # eep=440 (MS)
        pms_mass = 0.5 + 0.001 * (520.0 - 400.0)  # eep=520 (post-MS)
        w_ms = w[np.isclose(masses, ms_mass)].sum()
        w_pms = w[np.isclose(masses, pms_mass)].sum()
        assert w_pms > 0
        # Equal to the MS total weight (both integrate the full SMF axis)
        np.testing.assert_allclose(w_pms, w_ms, rtol=1e-12)

    def test_post_ms_stored_once(self):
        pop = MockStellarPop()
        eep_grid = np.array([450.0, 500.0])
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 0.5, 1.0]),
            eep_grid=eep_grid,
        )
        pms_mass = 0.5 + 0.001 * (500.0 - 450.0)
        n_pms_rows = np.sum(np.isclose(grid["masses"], pms_mass))
        assert n_pms_rows == 1, "post-MS model duplicated across SMF slices"


class TestNaNAutoMask:
    """A stray NaN flux or zero error must only drop that band."""

    def _tiny_grid(self):
        return {
            "photometry": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            "masses": np.array([0.5, 0.6]),
            "smf_values": np.zeros(2),
            "mass_jacobians": np.ones(2),
            "smf_jacobians": np.ones(2),
        }

    def test_nan_flux_does_not_poison_object(self):
        grid = self._tiny_grid()
        flux = np.array([[1.0, np.nan, 0.9], [1.1, 1.0, 0.9]])
        err = np.full_like(flux, 0.1)
        lnl = compute_isochrone_cluster_loglike(flux, err, grid, dim_prior=True)
        assert np.isfinite(lnl).all()
        # Equivalent to explicitly masking the NaN band
        mask = np.ones_like(flux)
        mask[0, 1] = 0
        flux_clean = np.nan_to_num(flux, nan=1.0)
        lnl_masked = compute_isochrone_cluster_loglike(
            flux_clean, err, grid, dim_prior=True, mask=mask
        )
        np.testing.assert_allclose(lnl, lnl_masked)

    def test_zero_error_does_not_poison_object(self):
        grid = self._tiny_grid()
        flux = np.array([[1.0, 1.0, 0.9]])
        err = np.array([[0.1, 0.0, 0.1]])
        lnl = compute_isochrone_cluster_loglike(flux, err, grid, dim_prior=False)
        assert np.isfinite(lnl).all()


class TestMeasureNormalization:
    """The marginalization must be invariant to the measure's overall scale."""

    def test_marginalization_invariant_to_jacobian_scale(self):
        rng = np.random.default_rng(0)
        lnl_mixture = rng.normal(-10, 2, size=(50, 4))
        mass_jac = rng.uniform(0.5, 1.5, 50)
        smf_jac = rng.uniform(0.5, 1.5, 50)
        base = marginalize_isochrone_grid(lnl_mixture, mass_jac, smf_jac)
        scaled = marginalize_isochrone_grid(lnl_mixture, mass_jac * 7.3, smf_jac)
        np.testing.assert_allclose(base, scaled, rtol=1e-12)

    def test_invalid_rows_do_not_change_outlier_mass(self):
        # A grid with extra invalid (NaN) rows must marginalize identically
        # to the same grid without them.
        rng = np.random.default_rng(1)
        lnl_valid = rng.normal(-10, 2, size=(30, 4))
        jac = rng.uniform(0.5, 1.5, 30)
        base = marginalize_isochrone_grid(lnl_valid, jac, np.ones(30))
        lnl_padded = np.vstack([lnl_valid, np.full((10, 4), np.nan)])
        jac_padded = np.concatenate([jac, rng.uniform(0.5, 1.5, 10)])
        padded = marginalize_isochrone_grid(lnl_padded, jac_padded, np.ones(40))
        np.testing.assert_allclose(base, padded, rtol=1e-12)


class TestParallaxMeasure:
    """Parallax must enter the chi-square (not a Gaussian density) when
    dim_prior=True, matching phot_loglike's extra_chi2 mechanism."""

    def test_parallax_enters_chi2_when_dim_prior(self):
        grid = {
            "photometry": np.array([[1.0, 1.0, 1.0]]),
            "masses": np.array([0.5]),
            "smf_values": np.zeros(1),
            "mass_jacobians": np.ones(1),
            "smf_jacobians": np.ones(1),
        }
        flux = np.array([[1.0, 1.05, 0.95]])
        err = np.full_like(flux, 0.1)
        parallax = np.array([1.0])  # mas
        parallax_err = np.array([0.1])
        dist = 1200.0  # pc -> predicted parallax 0.833 mas
        lnl = compute_isochrone_cluster_loglike(
            flux,
            err,
            grid,
            parallax=parallax,
            parallax_err=parallax_err,
            distance=dist,
            dim_prior=True,
        )
        chi2_par = (1.0 - 1000.0 / dist) ** 2 / 0.1**2
        expected = phot_loglike(
            flux,
            err,
            grid["photometry"],
            dim_prior=True,
            extra_chi2=np.array([chi2_par]),
            extra_dims=np.array([1]),
        ).T
        np.testing.assert_allclose(lnl, expected, rtol=1e-12)


class TestUniformOutlierDensity:
    """The uniform outlier model must be a proper flux-space density."""

    def test_unit_scaling(self):
        rng = np.random.default_rng(3)
        flux = rng.uniform(0.5, 2.0, (6, 4))
        err = rng.uniform(0.02, 0.1, (6, 4))
        base = uniform_outlier_loglike(flux, err)
        halved = uniform_outlier_loglike(flux * 0.5, err * 0.5)
        # density transforms as 1/volume: lnP shifts by +Nfilt*log(2)
        np.testing.assert_allclose(halved - base, 4 * np.log(2.0), rtol=1e-10)

    def test_small_errors_do_not_blow_up(self):
        rng = np.random.default_rng(4)
        flux = rng.uniform(0.5, 2.0, (6, 4))
        err = rng.uniform(0.02, 0.1, (6, 4))
        base = uniform_outlier_loglike(flux, err)
        tight = uniform_outlier_loglike(flux, err * 1e-3)
        # bounded by the +/- sigma_clip*err range padding (< ~log(2)/band);
        # the old dimensionless form scaled as -log(err), i.e. +4*log(1e3)
        # here (~27.6)
        assert np.all(tight - base < 4 * np.log(2.0))


class TestGetSedsKwargs:
    """StellarPop.get_seds must reject unknown keyword arguments."""

    def test_unknown_kwarg_raises(self, real_stellarpop):
        with pytest.raises(TypeError):
            real_stellarpop.get_seds(
                feh=0.0, loga=9.0, smf=0.5, eep=np.linspace(300, 400, 5)
            )


class TestBinaryPreservesPrimary:
    """Stars without a valid companion keep their primary-only SED."""

    def test_partial_binary_keeps_primaries(self, real_stellarpop):
        eep_grid = np.linspace(300, 550, 12)  # spans eep_binary_max
        sed_single, p1, _ = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.0
        )
        sed_binary, p1b, p2 = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.3
        )
        valid_single = np.isfinite(sed_single).all(axis=1)
        valid_binary = np.isfinite(sed_binary).all(axis=1)
        # A binary run must never lose models that were valid as singles
        assert np.all(
            valid_binary[valid_single]
        ), "binary companion invalidity poisoned valid primary SEDs"
        # Post-MS rows (no companion possible) must be identical
        post_ms = (eep_grid > 480.0) & valid_single
        if post_ms.any():
            np.testing.assert_allclose(
                sed_binary[post_ms], sed_single[post_ms], rtol=1e-12
            )

    def test_equal_mass_binary_respects_eep_max(self, real_stellarpop):
        eep_grid = np.linspace(300, 550, 12)
        sed_single, _, _ = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.0
        )
        sed_eq, _, p2 = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=1.0
        )
        valid = np.isfinite(sed_single).all(axis=1)
        ms = (eep_grid <= 480.0) & valid
        post_ms = (eep_grid > 480.0) & valid
        # MS stars brightened by exactly 2.5*log10(2)
        if ms.any():
            np.testing.assert_allclose(
                sed_single[ms] - sed_eq[ms], 2.5 * np.log10(2.0), rtol=1e-10
            )
        # post-MS stars unchanged, with no secondary parameters
        if post_ms.any():
            np.testing.assert_allclose(sed_eq[post_ms], sed_single[post_ms], rtol=1e-12)
            assert np.all(np.isnan(p2["mini"][post_ms]))
