#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean integration tests for StarGrid and BruteForce classes.

Tests the essential workflow from grid creation to stellar parameter estimation.
"""

import numpy as np
import pytest
from conftest import split_labels

from brutus.analysis.individual import BruteForce

# Import the classes to test
from brutus.core.individual import StarGrid
from brutus.data.filters import ps
from brutus.utils import inv_magnitude


@pytest.fixture(scope="module")
def mist_integration_setup(mist_ps_grid):
    """100 models of the MIST v9 grid, with the five PS1 filters."""
    models, combined_labels, label_mask = mist_ps_grid
    subset = np.arange(0, len(models), 1000)[:100]
    labels, params = split_labels(combined_labels[subset], label_mask)
    grid = StarGrid(models[subset], labels, params, filters=ps[:5])
    return {
        "grid": grid,
        "fitter": BruteForce(grid, verbose=False),
        "ps_indices": list(range(5)),
        "labels": labels,
    }


class TestCoreIntegration:
    """Test the core StarGrid + BruteForce workflow."""

    def test_end_to_end_workflow(self, mist_integration_setup):
        """Test complete workflow from observations to parameter estimates."""
        setup = mist_integration_setup
        grid = setup["grid"]
        fitter = setup["fitter"]
        ps_indices = setup["ps_indices"]

        # Create synthetic observations of a solar-like star
        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # PanSTARRS observations: g, r, i, z, y
        solar_mags = [15.0, 14.6, 14.3, 14.1, 14.0]
        for i, idx in enumerate(ps_indices):
            obs[idx] = solar_mags[i]
            obs_err[idx] = 0.05  # Realistic error

        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        # Step 1: Generate SED using StarGrid for a known model
        reference_idx = len(grid.labels) // 2  # Middle model
        ref_labels = grid.labels[reference_idx]

        sed, params_pred, _ = grid.get_seds(
            mini=float(ref_labels["mini"]),
            eep=float(ref_labels["eep"]),
            feh=float(ref_labels["feh"]),
            return_dict=True,
        )

        # Verify SED generation works
        assert sed is not None
        assert len(sed) == nfilters
        assert np.all(np.isfinite(sed))

        # Step 2: Use BruteForce to fit observations.
        lnl, ndim, chi2 = fitter.loglike_grid(
            obs,
            obs_err,
            obs_mask,
            ltol=5e-2,  # Moderate clipping for speed
            verbose=False,
        )

        # Step 3: Verify results are sensible
        assert lnl.shape == (fitter.nmodels,)
        assert np.all(np.isfinite(lnl))
        assert np.all(ndim > 0)
        assert np.all(chi2 >= 0)

        # Find best-fit model
        best_idx = np.argmax(lnl)
        best_labels = fitter.models_labels[best_idx]

        # Should find a reasonable stellar model
        # (not testing exact match due to noise and limited grid)
        assert 0.1 < best_labels["mini"] < 5.0  # Reasonable mass range
        assert 100 < best_labels["eep"] < 1000  # Reasonable evolutionary phase
        assert (
            -3.0 < best_labels["feh"] < 1.0
        )  # Reasonable metallicity (MIST goes to -3)

        print(
            f"Best-fit parameters: mini={best_labels['mini']:.2f}, eep={best_labels['eep']:.0f}, feh={best_labels['feh']:.2f}"
        )
        print(f"Best loglike: {lnl[best_idx]:.2f}")

    def test_loglike_grid_indices_select_models(self, mist_integration_setup):
        """`indices` restricts loglike_grid to the requested models."""
        setup = mist_integration_setup
        fitter = setup["fitter"]
        ps_indices = setup["ps_indices"]

        # Create simple observations
        nfilters = setup["grid"].nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Use fewer filters for speed
        for i, idx in enumerate(ps_indices[:3]):
            obs[idx] = 15.0
            obs_err[idx] = 0.05
        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        import warnings

        for n_models in [10, 50, 100]:
            indices = np.arange(n_models)
            # 3 bands trigger a RuntimeWarning about the 4-band minimum
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lnl, _, _ = fitter.loglike_grid(
                    obs,
                    obs_err,
                    obs_mask,
                    indices=indices,
                    ltol=5e-2,
                    verbose=False,
                )
            assert lnl.shape == (n_models,)
            assert np.all(np.isfinite(lnl))

    def test_grid_interpolation_accuracy(self, mist_integration_setup):
        """Test that StarGrid interpolation gives reasonable results."""
        grid = mist_integration_setup["grid"]
        labels = mist_integration_setup["labels"]

        # Find a model in the middle of the grid
        mini_vals = np.unique(labels["mini"])
        eep_vals = np.unique(labels["eep"])
        feh_vals = np.unique(labels["feh"])

        if len(mini_vals) > 2 and len(eep_vals) > 2 and len(feh_vals) > 2:
            # Test interpolation at intermediate point
            mini_mid = (mini_vals[1] + mini_vals[2]) / 2
            eep_mid = (eep_vals[1] + eep_vals[2]) / 2
            feh_mid = (feh_vals[1] + feh_vals[2]) / 2

            # Get SED with interpolation
            sed_interp, params_interp, _ = grid.get_seds(
                mini=mini_mid, eep=eep_mid, feh=feh_mid, use_multilinear=True
            )

            # Get SED with nearest neighbor
            sed_nn, params_nn, _ = grid.get_seds(
                mini=mini_mid, eep=eep_mid, feh=feh_mid, use_multilinear=False
            )

            # Both should give valid results
            assert np.all(np.isfinite(sed_interp))
            assert np.all(np.isfinite(sed_nn))

            # Interpolated parameters should be reasonable
            if isinstance(params_interp, dict) and "mass" in params_interp:
                assert 0.1 < params_interp["mass"] < 10.0

            print(
                f"Interpolation test: mini={mini_mid:.2f}, eep={eep_mid:.0f}, feh={feh_mid:.2f}"
            )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_inputs(self, mist_integration_setup):
        """Test handling of invalid inputs."""
        grid = mist_integration_setup["grid"]
        fitter = mist_integration_setup["fitter"]
        ps_indices = mist_integration_setup["ps_indices"]

        nfilters = grid.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Valid setup
        obs[ps_indices[0]] = 15.0
        obs_err[ps_indices[0]] = 0.05
        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        # Test with mismatched array sizes
        with pytest.raises((ValueError, IndexError)):
            fitter.loglike_grid(
                obs[:-1], obs_err, obs_mask, verbose=False
            )  # Wrong size

        # Test with all NaN observations
        obs_bad = np.full(nfilters, np.nan)
        obs_err_bad = np.full(nfilters, np.nan)
        obs_mask_bad = ~np.isnan(obs_bad)

        # Should handle gracefully (may error or return low likelihood)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = fitter.loglike_grid(
                    obs_bad, obs_err_bad, obs_mask_bad, verbose=False
                )
                # If it doesn't error, result should be valid
                assert isinstance(result, tuple)
                assert len(result) >= 3
            except (ValueError, RuntimeError, ZeroDivisionError):
                # Acceptable to raise an error for invalid input
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
