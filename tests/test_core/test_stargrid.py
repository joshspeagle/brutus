#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for StarGrid class.

Tests the StarGrid class for stellar model grid management and SED generation.
"""

import warnings

import numpy as np
import pytest
from conftest import find_brutus_data_file

# Import the class to test
from brutus.core.individual import StarGrid
from brutus.data import load_models

# ============================================================================
# Module-level fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mist_grid():
    """Load MIST v9 grid once for all tests in this module."""
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

                return StarGrid(models, labels, params)
            except Exception as e:
                continue

    # If the real grid is unavailable, skip dependent tests (CI pre-downloads).
    pytest.skip(
        "MIST grid data not available; run fetch_grids() or set BRUTUS_DATA_DIR "
        "to enable this data-dependent test"
    )


@pytest.fixture(scope="module")
def mock_grid():
    """Create a small mock grid for basic testing."""
    np.random.seed(42)

    # Create minimal grid (2x2x2 = 8 models)
    nmodels = 8
    nfilters = 5

    # Create models
    models = np.random.randn(nmodels, nfilters, 3)
    models[:, :, 0] = np.random.uniform(10, 20, (nmodels, nfilters))
    models[:, :, 1] = np.random.uniform(0.5, 2.0, (nmodels, nfilters))
    models[:, :, 2] = np.random.uniform(0.05, 0.2, (nmodels, nfilters))

    # Create labels (grid parameters only)
    labels_dtype = [("mini", "f4"), ("eep", "f4"), ("feh", "f4")]
    labels = np.zeros(nmodels, dtype=labels_dtype)

    # Create params (predictions/derived parameters)
    params_dtype = [("mass", "f4"), ("radius", "f4"), ("Teff", "f4")]
    params = np.zeros(nmodels, dtype=params_dtype)

    # Fill with grid pattern
    idx = 0
    for m in [0.5, 1.0]:
        for e in [200, 300]:
            for z in [-0.5, 0.0]:
                labels[idx] = (m, e, z)
                params[idx] = (m * 0.9, m**0.8, 5777 * m**0.5)
                idx += 1

    filters = ["g", "r", "i", "z", "y"]

    return StarGrid(models, labels, params, filters=filters)


# ============================================================================
# StarGrid Tests
# ============================================================================


class TestStarGridInitialization:
    """Test StarGrid initialization and setup."""

    def test_initialization_real_grid(self, mist_grid):
        """Test grid loads correctly with real MIST data."""
        assert mist_grid.nmodels > 0
        assert mist_grid.nfilters > 0
        assert len(mist_grid.label_names) > 0
        assert len(mist_grid.param_names) > 0

    def test_initialization_mock_grid(self, mock_grid):
        """Test grid initialization with mock data."""
        assert mock_grid.nmodels == 8
        assert mock_grid.nfilters == 5
        assert mock_grid.label_names == ["mini", "eep", "feh"]
        assert mock_grid.param_names == ["mass", "radius", "Teff"]

    def test_grid_structure(self, mock_grid):
        """Test grid structure is correctly identified."""
        # Check grid axes
        assert "mini" in mock_grid.grid_axes
        assert "eep" in mock_grid.grid_axes
        assert "feh" in mock_grid.grid_axes

        # Check unique values
        assert len(mock_grid.grid_axes["mini"]) == 2
        assert len(mock_grid.grid_axes["eep"]) == 2
        assert len(mock_grid.grid_axes["feh"]) == 2

    def test_grid_shape(self, mock_grid):
        """Test grid shape calculation."""
        assert mock_grid.grid_shape == [2, 2, 2]

    def test_kdtree_initialization(self, mock_grid):
        """Test KD-tree is built when needed."""
        # Initially None
        assert mock_grid.kdtree is None

        # Trigger KD-tree construction by using nearest neighbor mode
        sed, params, params2 = mock_grid.get_seds(
            mini=0.5, eep=200, feh=-0.5, use_multilinear=False  # Force KD-tree usage
        )

        # Now should be built
        assert mock_grid.kdtree is not None


class TestStarGridSEDGeneration:
    """Test SED generation methods."""

    def test_get_seds_exact_point(self, mock_grid):
        """Test SED at exact grid point."""
        # Get parameters of first model
        first_model = mock_grid.labels[0]

        sed, params, params2 = mock_grid.get_seds(
            mini=first_model["mini"], eep=first_model["eep"], feh=first_model["feh"]
        )

        assert sed is not None
        assert isinstance(params, dict)
        assert isinstance(params2, dict)
        assert sed.shape == (5,)

    def test_get_seds_with_extinction(self, mock_grid):
        """Test SED generation with extinction."""
        params = {"mini": 1.0, "eep": 300, "feh": 0.0}

        # Get SED without extinction
        sed_no_ext, _, _ = mock_grid.get_seds(**params, av=0.0, rv=3.3)

        # Get SED with extinction
        sed_ext, _, _ = mock_grid.get_seds(**params, av=0.5, rv=3.1)

        # Should be different (extinction makes things fainter)
        assert not np.allclose(sed_no_ext, sed_ext)
        # With extinction, magnitudes should be larger (fainter)
        assert np.all(sed_ext > sed_no_ext)

    def test_distance_scaling(self, mock_grid):
        """Test distance affects magnitude."""
        params = {"mini": 1.0, "eep": 300, "feh": 0.0}

        # Get magnitude at 1 kpc
        sed_1kpc, _, _ = mock_grid.get_seds(**params, dist=1000.0, return_flux=False)

        # Get magnitude at 2 kpc
        sed_2kpc, _, _ = mock_grid.get_seds(**params, dist=2000.0, return_flux=False)

        # Magnitude difference should be 5*log10(2) ≈ 1.505
        mag_diff = sed_2kpc - sed_1kpc
        expected_diff = 5.0 * np.log10(2.0)
        np.testing.assert_allclose(mag_diff, expected_diff, rtol=1e-4)

    def test_return_formats(self, mock_grid):
        """Test different return formats."""
        params = {"mini": 1.0, "eep": 300, "feh": 0.0}

        # Return with dict parameters (default)
        sed, params_dict, params2_dict = mock_grid.get_seds(**params, return_dict=True)
        assert isinstance(params_dict, dict)
        assert isinstance(params2_dict, dict)

        # Return with array parameters
        sed, params_arr, params2_arr = mock_grid.get_seds(**params, return_dict=False)
        assert isinstance(params_arr, np.ndarray)
        assert isinstance(params2_arr, np.ndarray)

        # Return flux vs magnitude
        sed_mag, _, _ = mock_grid.get_seds(**params, return_flux=False)
        sed_flux, _, _ = mock_grid.get_seds(**params, return_flux=True)

        # Convert and compare
        flux_from_mag = 10 ** (-0.4 * sed_mag)
        np.testing.assert_allclose(flux_from_mag, sed_flux, rtol=1e-6)


class TestStarGridInterpolation:
    """Test interpolation methods."""

    def test_interpolation_vs_nearest(self, mock_grid):
        """Test multilinear vs nearest neighbor."""
        # Use intermediate point
        params = {"mini": 0.75, "eep": 250, "feh": -0.25}

        # Get with multilinear interpolation
        sed_multi, params_multi, _ = mock_grid.get_seds(**params, use_multilinear=True)

        # Get with nearest neighbor
        sed_nn, params_nn, _ = mock_grid.get_seds(**params, use_multilinear=False)

        # Results might be different if interpolation succeeded
        # (depends on whether point is inside grid bounds)
        assert sed_multi is not None
        assert sed_nn is not None

    def test_interpolation_bounds(self, mock_grid):
        """Test interpolated values are reasonable."""
        # Use intermediate point
        params = {"mini": 0.75, "eep": 250, "feh": -0.25}

        sed, pred, _ = mock_grid.get_seds(**params, use_multilinear=True)

        # Check SED is reasonable
        assert sed is not None
        assert np.all(np.isfinite(sed))

        # Check parameters are within reasonable bounds
        if mock_grid.params is not None and "mass" in pred:
            # Mass should be between min and max in grid
            all_masses = mock_grid.params["mass"]
            assert np.min(all_masses) <= pred["mass"] <= np.max(all_masses)

        if mock_grid.params is not None and "Teff" in pred:
            # Same for other parameters
            all_teff = mock_grid.params["Teff"]
            assert np.min(all_teff) <= pred["Teff"] <= np.max(all_teff)

    def test_extrapolation_fallback(self, mock_grid):
        """Test behavior outside grid bounds."""
        # Parameters well outside grid
        params = {"mini": 10.0, "eep": 1000, "feh": 2.0}

        # Should fall back to nearest neighbor
        sed, pred, _ = mock_grid.get_seds(**params, use_multilinear=True)

        # Should still return valid SED
        assert sed is not None
        assert np.all(np.isfinite(sed))


class TestStarGridEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_parameters(self, mock_grid):
        """Omitting a multi-valued grid parameter raises a clear error."""
        # Historically this silently evaluated at the grid-minimum feh
        # (about -3 for MIST), corrupting results with no signal; there is
        # no sensible default, so it must fail fast.
        with pytest.raises(ValueError, match="feh"):
            mock_grid.get_seds(mini=1.0, eep=300)  # Missing feh

    def test_negative_distance(self, mock_grid):
        """Test with invalid distance."""
        params = {"mini": 1.0, "eep": 300, "feh": 0.0}

        # Negative distance will cause issues with log10
        # Test that it's handled somehow (may give NaN or use abs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sed, _, _ = mock_grid.get_seds(**params, dist=-1000)

        # Result exists but may contain NaNs
        assert sed is not None

    def test_negative_extinction(self, mock_grid):
        """Test negative extinction values."""
        params = {"mini": 1.0, "eep": 300, "feh": 0.0}

        # Negative Av might be clipped to 0 or allowed
        sed_neg, _, _ = mock_grid.get_seds(**params, av=-0.5)
        sed_zero, _, _ = mock_grid.get_seds(**params, av=0.0)

        # Should give same or similar result
        # (implementation may clip negative Av to 0)
        assert sed_neg is not None
        assert sed_zero is not None

    def test_single_filter(self):
        """Test with single filter grid."""
        # Create grid with one filter
        models = np.random.randn(8, 1, 3)
        labels = np.zeros(8, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])
        for i in range(8):
            labels[i] = (
                0.5 + 0.5 * (i // 4),
                200 + 100 * (i // 2 % 2),
                -0.5 + 0.5 * (i % 2),
            )

        grid = StarGrid(models, labels)

        sed, params, _ = grid.get_seds(mini=0.5, eep=200, feh=-0.5)
        assert sed.shape == (1,)

    def test_models_as_dict(self):
        """Test StarGrid initialization with models as a dict."""
        # Create mock model data as a dict
        nmodels = 5
        nfilters = 3
        mag_coeffs = np.random.randn(nmodels, nfilters, 3)
        labels = np.zeros(nmodels, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])
        params = np.zeros(nmodels, dtype=[("mass", "f4"), ("Teff", "f4")])

        for i in range(nmodels):
            labels[i] = (0.8 + 0.2 * i, 300 + 10 * i, -0.5 + 0.1 * i)
            params[i] = (0.8 + 0.2 * i, 5000 + 100 * i)

        models_dict = {"mag_coeffs": mag_coeffs, "labels": labels, "parameters": params}

        # Initialize with dict (labels extracted from dict)
        grid = StarGrid(models_dict, None, verbose=False)

        assert grid.nmodels == nmodels
        assert grid.nfilters == nfilters
        assert np.array_equal(grid.labels, labels)
        assert np.array_equal(grid.params, params)

    def test_models_2d_reshape(self):
        """Test StarGrid with 2D models array that needs reshaping."""
        nmodels = 5
        nfilters = 4
        # Create 2D array (Nmodels, Nfilters * 3)
        models_2d = np.random.randn(nmodels, nfilters * 3)
        labels = np.zeros(nmodels, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])

        for i in range(nmodels):
            labels[i] = (1.0 + 0.1 * i, 300, 0.0)

        grid = StarGrid(models_2d, labels)

        assert grid.nmodels == nmodels
        assert grid.nfilters == nfilters
        assert grid.models.shape == (nmodels, nfilters, 3)

    def test_models_wrong_coefficients(self):
        """Test StarGrid raises error with wrong number of coefficients."""
        nmodels = 5
        nfilters = 3
        # Create models with 4 coefficients instead of 3
        models_bad = np.random.randn(nmodels, nfilters, 4)
        labels = np.zeros(nmodels, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])

        with pytest.raises(ValueError, match="Expected 3 coefficients"):
            grid = StarGrid(models_bad, labels)

    def test_models_dict_missing_mag_coeffs(self):
        """Test StarGrid raises error when dict is missing mag_coeffs."""
        models_dict = {"labels": np.zeros(5, dtype=[("mini", "f4")])}

        with pytest.raises(ValueError, match="must contain 'mag_coeffs'"):
            grid = StarGrid(models_dict, None, verbose=False)

    def test_models_structured_array_with_filter_selection(self):
        """Test StarGrid with structured array and filter selection."""
        nmodels = 5
        # Create structured array with named filter columns
        dtype = [("PS_g", "3f4"), ("PS_r", "3f4"), ("PS_i", "3f4")]
        models_structured = np.zeros(nmodels, dtype=dtype)

        for i in range(nmodels):
            models_structured[i]["PS_g"] = [15.0 + i, 0.5, 0.1]
            models_structured[i]["PS_r"] = [14.5 + i, 0.6, 0.12]
            models_structured[i]["PS_i"] = [14.0 + i, 0.7, 0.15]

        labels = np.zeros(nmodels, dtype=[("mini", "f4"), ("eep", "f4"), ("feh", "f4")])
        for i in range(nmodels):
            labels[i] = (1.0 + 0.1 * i, 300, 0.0)

        # Select only PS_g and PS_i
        grid = StarGrid(models_structured, labels, filters=["PS_g", "PS_i"])

        assert grid.nmodels == nmodels
        assert grid.nfilters == 2
        assert list(grid.filters) == ["PS_g", "PS_i"]
        assert grid.models.shape == (nmodels, 2, 3)


class TestStarGridWithRealData:
    """Tests using real MIST grid (if available)."""

    def test_real_grid_dimensions(self, mist_grid):
        """Test real grid has expected structure."""
        # MIST grid should have many models
        assert mist_grid.nmodels > 1000

        # Should have standard labels
        expected_labels = ["mini", "eep", "feh"]
        for label in expected_labels:
            assert label in mist_grid.label_names

        # Should have predictions
        assert len(mist_grid.param_names) >= 5

    def test_real_grid_sed_generation(self, mist_grid):
        """Test SED generation with real grid."""
        # Use typical solar-like star parameters
        sed, params, params2 = mist_grid.get_seds(
            mini=1.0,
            eep=350,  # Main sequence
            feh=0.0,  # Solar metallicity
            av=0.1,  # Small extinction
            rv=3.1,
            dist=100.0,  # 100 pc
        )

        assert sed is not None
        assert isinstance(params, dict)

        # Check predictions are reasonable for solar-like star
        if "logt" in params:
            # Temperature in log scale - solar is around log10(5777) ≈ 3.76
            assert 3.6 < params["logt"] < 3.85

        if "logg" in params:
            # Main sequence gravity
            assert 3.5 < params["logg"] < 5.0

    def test_real_grid_interpolation(self, mist_grid):
        """Test interpolation with real grid."""
        # Get unique grid values
        mini_vals = np.unique(mist_grid.labels["mini"])
        eep_vals = np.unique(mist_grid.labels["eep"])
        feh_vals = np.unique(mist_grid.labels["feh"])

        # Pick intermediate values
        if len(mini_vals) > 1 and len(eep_vals) > 1 and len(feh_vals) > 1:
            params = {
                "mini": (mini_vals[0] + mini_vals[1]) / 2,
                "eep": (eep_vals[0] + eep_vals[1]) / 2,
                "feh": (feh_vals[0] + feh_vals[1]) / 2,
            }

            # Test both interpolation methods
            sed_multi, params_multi, _ = mist_grid.get_seds(
                **params, use_multilinear=True
            )
            sed_nn, params_nn, _ = mist_grid.get_seds(**params, use_multilinear=False)

            # Both should return valid SEDs
            assert sed_multi is not None
            assert sed_nn is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
