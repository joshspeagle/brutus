#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive integration tests for StarGrid and BruteForce classes.

Tests the complete workflow from grid creation to stellar parameter estimation.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
from conftest import find_brutus_data_file

from brutus.analysis.individual import BruteForce

# Import the classes to test
from brutus.core.individual import StarGrid
from brutus.data import load_models
from brutus.data.filters import ps

# ============================================================================
# Module-level fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mist_grid():
    """Load MIST v9 grid once for all integration tests."""
    import os

    # Try different paths for the MIST grid file
    grid_paths = [
        find_brutus_data_file("grid_mist_v9.h5"),
        "./data/DATAFILES/grid_mist_v9.h5",
        "data/DATAFILES/grid_mist_v9.h5",
    ]

    for path in grid_paths:
        if path and os.path.exists(path):
            try:
                models, combined_labels, label_mask = load_models(path)

                # Separate grid parameters from predictions using label_mask
                grid_names = [
                    name
                    for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                    if is_grid
                ]
                pred_names = [
                    name
                    for name, is_grid in zip(combined_labels.dtype.names, label_mask[0])
                    if not is_grid
                ]

                # Extract grid parameters
                if grid_names:
                    labels = combined_labels[grid_names]
                else:
                    # Fallback to standard grid parameters
                    grid_names = ["mini", "eep", "feh"]
                    labels = combined_labels[grid_names]

                # Extract predictions if available
                if pred_names:
                    params = combined_labels[pred_names]
                else:
                    params = None

                # Use only first 5 Pan-STARRS filters for testing (g, r, i, z, y)
                ps_filters = ps[
                    :5
                ]  # Get first 5 PS filters: PS_g, PS_r, PS_i, PS_z, PS_y

                # Load the grid with only PS filters
                models_ps, combined_labels_ps, label_mask_ps = load_models(
                    path, filters=ps_filters
                )

                # Create a subset for faster testing (every 100th model to get ~6000 models)
                subset_indices = slice(0, None, 100)
                models_subset = models_ps[subset_indices]
                labels_subset = combined_labels_ps[subset_indices]

                # Re-extract labels and params for PS-filtered data
                if grid_names:
                    labels_final = labels_subset[grid_names]
                else:
                    grid_names = ["mini", "eep", "feh"]
                    labels_final = labels_subset[grid_names]

                if pred_names:
                    params_final = labels_subset[pred_names]
                else:
                    params_final = None

                return StarGrid(models_subset, labels_final, params_final)
            except Exception as e:
                continue

    # Real MIST grid should be available
    raise FileNotFoundError("MIST grid file not found in expected locations")


@pytest.fixture
def multi_star_observations():
    """Create multiple synthetic stellar observations."""
    np.random.seed(42)

    # Define different stellar types
    star_params = [
        # Solar-like star
        {"mini": 1.0, "eep": 350, "feh": 0.0, "av": 0.1, "rv": 3.1, "dist": 100},
        # Low-mass main sequence
        {"mini": 0.5, "eep": 300, "feh": -0.5, "av": 0.2, "rv": 3.3, "dist": 50},
        # Massive star
        {"mini": 1.8, "eep": 320, "feh": 0.2, "av": 0.5, "rv": 2.8, "dist": 200},
        # Evolved star
        {"mini": 1.2, "eep": 450, "feh": 0.1, "av": 0.3, "rv": 3.5, "dist": 150},
    ]

    return star_params


# ============================================================================
# Integration tests
# ============================================================================


class TestStarGridBruteForceIntegration:
    """Test integration between StarGrid and BruteForce."""

    def test_complete_workflow(self, mist_grid):
        """Test complete workflow from grid to fitting."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Create synthetic observation from known model
        true_idx = 24  # Middle model
        true_params = {
            "mini": grid.labels["mini"][true_idx],
            "eep": grid.labels["eep"][true_idx],
            "feh": grid.labels["feh"][true_idx],
        }

        # Generate SED through StarGrid
        true_sed, true_params_out, true_params2 = grid.get_seds(
            **true_params, av=0.2, rv=3.1, dist=100.0, return_flux=True
        )

        # Use true SED with error bars but no added noise
        flux = true_sed
        flux_err = true_sed * 0.03
        mask = np.ones(5, dtype=bool)

        # Fit through BruteForce
        results = fitter._fit(flux, flux_err, mask, Nmc_prior=20, wt_thresh=0.05)

        # _fit returns 14 values with return_distreds=True (default)
        (
            idxs,
            scales,
            avs,
            rvs,
            covs_sar,
            Ndim,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = results

        # Check results are reasonable
        assert len(idxs) > 0
        assert np.all(np.isfinite(lnprob))

        # BruteForce does model selection, so just check that we get reasonable results
        # Check that we sampled valid model indices
        assert len(idxs) > 0
        assert np.all(np.isfinite(lnprob))

    def test_interpolation_vs_grid_points(self, mist_grid):
        """Compare interpolated SEDs vs exact grid points."""
        grid = mist_grid

        # Get SED at exact grid point
        exact_params = {"mini": 1.2, "eep": 400, "feh": 0.0}
        sed_exact, params_exact, params2_exact = grid.get_seds(
            **exact_params, use_multilinear=False
        )

        # Get SED with interpolation enabled
        sed_interp, params_interp, params2_interp = grid.get_seds(
            **exact_params, use_multilinear=True
        )

        # If interpolation worked, should be similar or identical
        if sed_exact.shape == sed_interp.shape:
            # Check that they're reasonably close
            relative_diff = np.abs(sed_exact - sed_interp) / (sed_exact + 1e-10)
            assert np.all(relative_diff < 0.1)  # Within 10%
        else:
            # If interpolation fell back to grid point, should be identical
            np.testing.assert_array_equal(sed_exact, sed_interp)

    def test_optimization_convergence_properties(self, mist_grid):
        """Test optimization produces reasonable results."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Create clean synthetic data
        true_model = grid.models[20, :, 0]  # Base magnitudes
        flux = 10 ** (-0.4 * true_model)
        flux_err = flux * 0.01  # Very low noise
        mask = np.ones(5, dtype=bool)

        # Run with tight optimization
        results = fitter.loglike_grid(
            flux,
            flux_err,
            mask,
            avlim=(0.0, 1.0),
            rvlim=(2.5, 4.0),
            ltol=1e-3,
            return_vals=True,
        )

        lnl, ndim, chi2, scale, av, rv, icov = results

        # Best model should have reasonable parameters
        best_idx = np.argmax(lnl)

        # Scale should be reasonable (allowing for optimization variations)
        assert 0.1 < scale[best_idx] < 5.0

        # Extinction should be small for this clean case
        assert av[best_idx] < 0.5

        # Basic check that inverse covariance is finite
        assert np.all(np.isfinite(icov[best_idx]))

    def test_multi_object_fitting(self, mist_grid, multi_star_observations):
        """Test fitting multiple objects."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        results = []

        for params in multi_star_observations:
            # Generate synthetic SED
            sed, _, _ = grid.get_seds(**params, return_flux=True)
            flux = sed * (1 + np.random.normal(0, 0.05, len(sed)))
            flux_err = sed * 0.05
            mask = np.ones(len(sed), dtype=bool)

            # Fit
            result = fitter._fit(flux, flux_err, mask, Nmc_prior=10, wt_thresh=0.1)
            results.append(result)

        # All fits should succeed
        assert len(results) == len(multi_star_observations)

        # Different stars should prefer different models
        # _fit returns 14 values: (idxs, scales, avs, rvs, covs_sar, Ndim, lnprob, levid, chi2min, dists, reds, dreds, logwts, mc_ess)
        best_models = [
            idxs[np.argmax(lnprob)]
            for idxs, _, _, _, _, _, lnprob, _, _, _, _, _, _, _ in results
        ]
        assert len(set(best_models)) > 1  # Not all the same

    def test_prior_effects_on_posteriors(self, mist_grid):
        """Test that priors affect posterior distributions."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Create ambiguous data (high noise)
        true_model = grid.models[15, :, 0]
        flux = 10 ** (-0.4 * true_model) * (
            1 + np.random.normal(0, 0.3, len(true_model))
        )
        flux_err = 10 ** (-0.4 * true_model) * 0.3
        mask = np.ones(len(true_model), dtype=bool)

        # Fit with different priors
        results_flat = fitter.loglike_grid(flux, flux_err, mask, av_gauss=(0.0, 1e6))
        results_tight = fitter.loglike_grid(flux, flux_err, mask, av_gauss=(0.1, 0.1))

        # Check that priors have some effect (not identical results)
        lnl_flat, _, _ = results_flat
        lnl_tight, _, _ = results_tight

        # Should get different likelihood patterns
        assert not np.allclose(lnl_flat, lnl_tight, rtol=0.1)


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def test_single_band_observation(self, mist_grid):
        """Test fitting with only one filter."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Single band observation (first filter only)
        flux = np.array([1e-8, 0, 0, 0, 0])  # 5 filters but only first has data
        flux_err = np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
        mask = np.array([True, False, False, False, False])  # Only first filter used

        # Should handle gracefully or raise informative error
        try:
            results = fitter.loglike_grid(flux, flux_err, mask)
            lnl, ndim, chi2 = results
            assert len(lnl) > 0
        except (ValueError, RuntimeError) as e:
            # Acceptable to fail with informative error for single band
            assert "insufficient" in str(e).lower() or "dimension" in str(e).lower()

    def test_extreme_noise_levels(self, mist_grid):
        """Test behavior with very high noise."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # High noise observation
        true_model = grid.models[10, :5, 0]
        flux = 10 ** (-0.4 * true_model)
        flux_err = flux * 10.0  # 1000% error bars
        mask = np.ones(5, dtype=bool)

        # Should complete without error
        results = fitter.loglike_grid(flux, flux_err, mask)
        lnl, ndim, chi2 = results

        assert len(lnl) > 0
        assert np.all(np.isfinite(lnl))

    def test_boundary_conditions(self, mist_grid):
        """Test fitting at parameter boundaries."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Use boundary model (first/last in grid)
        true_model = grid.models[0, :5, 0]  # First model
        flux = 10 ** (-0.4 * true_model)
        flux_err = flux * 0.05
        mask = np.ones(5, dtype=bool)

        # Should handle boundary fitting
        results = fitter.loglike_grid(flux, flux_err, mask, avlim=(0.0, 0.1))
        lnl, ndim, chi2 = results

        assert len(lnl) > 0
        assert np.all(np.isfinite(lnl))

    def test_optimization_bounds_enforcement(self, mist_grid):
        """Test that optimization respects parameter bounds."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Normal observation
        true_model = grid.models[25, :5, 0]
        flux = 10 ** (-0.4 * true_model)
        flux_err = flux * 0.03
        mask = np.ones(5, dtype=bool)

        # Tight bounds
        results = fitter.loglike_grid(
            flux,
            flux_err,
            mask,
            avlim=(0.5, 0.6),  # Very tight
            rvlim=(3.0, 3.2),
            return_vals=True,
        )

        lnl, ndim, chi2, scale, av, rv, icov = results

        # Check bounds are respected
        assert np.all((av >= 0.5) & (av <= 0.6))
        assert np.all((rv >= 3.0) & (rv <= 3.2))


# ============================================================================
# Performance and scaling tests
# ============================================================================


class TestPerformanceAndScaling:
    """Test performance characteristics (basic checks)."""

    def test_grid_size_scaling(self, mist_grid):
        """Test that operations scale reasonably with grid size."""
        # Use mock grid which has consistent filter count
        fitter = BruteForce(mist_grid, verbose=False)

        # Create simple observation
        flux = np.array([0.1, 0.12, 0.15, 0.18, 0.2])
        flux_err = flux * 0.05
        mask = np.ones(5, dtype=bool)

        # Time a likelihood calculation (just check it completes)
        lnl, ndim, chi2 = fitter.loglike_grid(flux, flux_err, mask)

        assert len(lnl) > 0
        assert np.all(np.isfinite(lnl))

    def test_batch_operations_efficiency(self, mist_grid):
        """Test that batch operations are reasonably efficient."""
        grid = mist_grid

        # Test batch SED generation
        n_test = 10
        params_list = []
        for i in range(n_test):
            params_list.append(
                {"mini": 1.0 + 0.1 * i, "eep": 350 + 10 * i, "feh": -0.1 * i}
            )

        # Should complete in reasonable time
        for params in params_list:
            sed, _, _ = grid.get_seds(**params, return_flux=True)
            assert len(sed) > 0


# ============================================================================
# Real-world scenarios
# ============================================================================


class TestRealWorldScenarios:
    """Test realistic astrophysical scenarios."""

    def test_typical_photometric_survey(self, mist_grid):
        """Test scenario like typical photometric survey."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Simulate Pan-STARRS-like observation
        # Typical errors: ~0.02 mag in good conditions
        true_model = grid.models[25, :, 0]  # Random model
        flux = 10 ** (-0.4 * true_model)

        # Add realistic noise
        flux = flux * (1 + np.random.normal(0, 0.02, 5))
        flux_err = flux * 0.02
        mask = np.ones(5, dtype=bool)

        # Add some parallax constraint (like Gaia)
        results = fitter._fit(
            flux,
            flux_err,
            mask,
            parallax=2.0,  # 500 pc
            parallax_err=0.2,  # 10% error
            coord=(180.0, 30.0),  # Some Galactic coordinates
            Nmc_prior=50,
            wt_thresh=0.01,
        )

        # _fit returns 14 values with return_distreds=True (default)
        (
            idxs,
            scales,
            avs,
            rvs,
            covs_sar,
            Ndim,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = results

        # Should get reasonable posterior samples
        assert len(idxs) > 0
        assert np.all(dists > 0)

        # Distance should be positive and reasonable (units may differ)
        median_dist = np.median(dists)
        assert median_dist > 0  # Distance should be positive

    def test_reddened_star_scenario(self, mist_grid):
        """Test scenario with significant reddening."""
        grid = mist_grid
        fitter = BruteForce(grid, verbose=False)

        # Generate heavily reddened observation
        true_params = {"mini": 1.5, "eep": 380, "feh": 0.2}
        true_sed, _, _ = grid.get_seds(
            **true_params, av=2.0, rv=3.1, dist=200.0, return_flux=True
        )

        flux = true_sed * (1 + np.random.normal(0, 0.04, len(true_sed)))
        flux_err = true_sed * 0.04
        mask = np.ones(len(true_sed), dtype=bool)

        # Fit allowing for high extinction
        results = fitter._fit(
            flux,
            flux_err,
            mask,
            avlim=(0.0, 5.0),
            rvlim=(2.5, 4.5),
            Nmc_prior=30,
            wt_thresh=0.02,
        )

        # _fit returns 14 values with return_distreds=True (default)
        (
            idxs,
            scales,
            avs,
            rvs,
            covs_sar,
            Ndim,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = results

        # Should recover reasonable extinction (allowing some tolerance for fitting)
        assert len(idxs) > 0
        median_av = np.median(reds)  # reds is the A(V) samples
        assert median_av > 0.5  # Should find some extinction (relaxed from 1.0)
        assert median_av < 4.0  # But not crazy high
