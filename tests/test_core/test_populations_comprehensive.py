#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus stellar populations module.

This test suite covers:
1. Isochrone class initialization and methods
2. StellarPop class initialization and methods
3. Parameter validation and error handling
4. Edge cases and boundary conditions
5. Integration between Isochrone and StellarPop
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest

# Try importing from the reorganized structure first, fall back to original
try:
    from brutus.core.populations import Isochrone, StellarPop
except ImportError:
    raise  # Module should be available


@pytest.fixture(scope="module")
def real_isochrone():
    """Load real MIST isochrone once for module tests."""
    from conftest import find_brutus_data_file

    iso_file = find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")

    assert iso_file is not None, "iso_file should be found after downloading all data"

    return Isochrone(mistfile=iso_file, verbose=False)


class TestIsochroneInitialization:
    """Test Isochrone class initialization."""

    def test_isochrone_default_initialization(self):
        """Test Isochrone initialization with real MIST data file."""
        from conftest import find_brutus_data_file

        # Try to find the real MIST isochrone file
        iso_file = find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")

        assert (
            iso_file is not None
        ), "iso_file should be found after downloading all data"

        # Test with real MIST data
        iso = Isochrone(mistfile=iso_file, verbose=False)

        # Check that basic attributes are set
        assert hasattr(iso, "predictions")
        assert hasattr(iso, "feh_grid")
        assert hasattr(iso, "afe_grid")
        assert hasattr(iso, "loga_grid")
        assert hasattr(iso, "eep_grid")

        # Verify the grids have reasonable values
        assert len(iso.feh_grid) > 0
        assert len(iso.afe_grid) > 0
        assert len(iso.loga_grid) > 0
        assert len(iso.eep_grid) > 0

        # Check that we can get predictions
        preds = iso.get_predictions(feh=0.0, afe=0.0, loga=9.5)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] > 0  # Should have stellar models
        assert preds.shape[1] == len(iso.predictions)  # Should match prediction labels

    def test_isochrone_custom_predictions(self):
        """Test Isochrone initialization with custom predictions using real data."""
        from conftest import find_brutus_data_file

        custom_preds = ["mini", "mass", "logt", "logg"]
        iso_file = find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")

        assert (
            iso_file is not None
        ), "iso_file should be found after downloading all data"

        iso = Isochrone(mistfile=iso_file, predictions=custom_preds, verbose=False)
        assert iso.predictions == custom_preds

        # Test that we can get predictions
        preds = iso.get_predictions(feh=0.0, afe=0.0, loga=9.5)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] > 0  # Should have stellar models

    def test_isochrone_file_not_found_error(self):
        """Test Isochrone initialization with non-existent file."""
        with patch("h5py.File", side_effect=OSError("File not found")):
            with pytest.raises(RuntimeError):
                Isochrone(mistfile="nonexistent.h5", verbose=False)

    def test_isochrone_verbose_output(self, capsys):
        """Test that verbose output is produced when requested."""
        with patch("h5py.File") as mock_file:
            mock_context = MagicMock()
            mock_file.return_value.__enter__.return_value = mock_context
            feh_grid = np.array([0.0])
            afe_grid = np.array([0.0])
            loga_grid = np.array([9.0])
            eep_grid = np.arange(202, 808)
            predictions = np.random.random((1, 1, 1, len(eep_grid), 8))

            mock_predictions = MagicMock()
            mock_predictions.__getitem__.return_value = predictions
            mock_predictions.attrs = {
                "labels": [
                    "mini",
                    "mass",
                    "logl",
                    "logt",
                    "logr",
                    "logg",
                    "feh_surf",
                    "afe_surf",
                ]
            }

            mock_context.__getitem__.side_effect = lambda key: {
                "feh": feh_grid,
                "afe": afe_grid,
                "loga": loga_grid,
                "eep": eep_grid,
                "predictions": mock_predictions,
            }[key]

            iso = Isochrone(verbose=True)
            captured = capsys.readouterr()
            assert "Constructing MIST isochrones" in captured.err


class TestIsochroneMethods:
    """Test Isochrone class methods."""

    def test_get_predictions_default_parameters(self, real_isochrone):
        """Test get_predictions with default parameters using real data."""
        preds = real_isochrone.get_predictions()

        # Check output shape and type
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 2
        assert preds.shape[1] == len(real_isochrone.predictions)
        assert preds.shape[0] > 0  # Should have stellar models

        # Should have at least some finite values (not all NaN)
        assert np.any(np.isfinite(preds))

    def test_get_predictions_custom_parameters(self, real_isochrone):
        """Test get_predictions with custom stellar parameters using real data."""
        # Use parameters that should give finite results
        preds = real_isochrone.get_predictions(feh=0.0, afe=0.0, loga=9.5)

        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 2
        assert preds.shape[0] > 0

        # Should have at least some finite values
        assert np.any(np.isfinite(preds))

    def test_get_predictions_specific_eep_range(self, real_isochrone):
        """Test get_predictions with specific EEP range using real data."""
        eep_range = np.arange(300, 500, 50)
        preds = real_isochrone.get_predictions(eep=eep_range)

        assert preds.shape[0] == len(eep_range)
        assert isinstance(preds, np.ndarray)
        assert np.all(np.isfinite(preds))

    def test_get_predictions_corrections(self, real_isochrone):
        """Test get_predictions with and without corrections using real data."""
        preds_no_corr = real_isochrone.get_predictions(apply_corr=False)
        preds_with_corr = real_isochrone.get_predictions(apply_corr=True)

        # Should get results (corrections may or may not change values depending on isochrone)
        assert isinstance(preds_no_corr, np.ndarray)
        assert isinstance(preds_with_corr, np.ndarray)
        assert preds_no_corr.shape == preds_with_corr.shape


class TestStellarPopInitialization:
    """Test StellarPop class initialization."""

    @pytest.fixture
    def mock_isochrone(self):
        """Create a mock isochrone for StellarPop testing."""
        iso = MagicMock()
        iso.get_predictions.return_value = np.random.random((100, 8))
        return iso

    def test_stellarpop_basic_initialization(self, mock_isochrone):
        """Test StellarPop initialization with basic parameters."""
        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_nn.return_value = MagicMock()

            pop = StellarPop(isochrone=mock_isochrone, verbose=False)

            assert pop.isochrone is mock_isochrone
            assert hasattr(pop, "filters")
            assert hasattr(pop, "predictor")

    def test_stellarpop_custom_filters(self, mock_isochrone):
        """Test StellarPop initialization with custom filters."""
        custom_filters = ["U", "B", "V", "R", "I"]

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_nn.return_value = MagicMock()

            pop = StellarPop(
                isochrone=mock_isochrone, filters=custom_filters, verbose=False
            )

            assert pop.filters == custom_filters

    def test_stellarpop_verbose_output(self, mock_isochrone, capsys):
        """Test that verbose output is produced when requested."""
        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_nn.return_value = MagicMock()

            pop = StellarPop(isochrone=mock_isochrone, verbose=True)

            captured = capsys.readouterr()
            # Should contain some verbose output about initialization
            assert len(captured.err) > 0


class TestStellarPopMethods:
    """Test StellarPop class methods."""

    @pytest.fixture
    def mock_stellar_pop(self):
        """Create a mock StellarPop for testing."""
        iso = MagicMock()
        # Create realistic stellar parameters with reasonable values
        stellar_params = np.array(
            [
                [0.8, 1.2, 1.5, 2.0, 2.5],  # mini - initial masses in solar masses
                [0.8, 1.2, 1.5, 2.0, 2.5],  # mass - current masses
                [0.5, 1.0, 1.5, 2.0, 2.5],  # logl - log luminosity
                [3.7, 3.8, 3.9, 4.0, 4.1],  # logt - log temperature
                [0.2, 0.5, 0.8, 1.1, 1.4],  # logr - log radius
                [4.5, 4.3, 4.1, 3.9, 3.7],  # logg - log surface gravity
                [0.0, 0.0, 0.0, 0.0, 0.0],  # feh_surf - surface metallicity
                [0.0, 0.0, 0.0, 0.0, 0.0],  # afe_surf - surface alpha enhancement
            ]
        ).T  # Transpose to get (Nstar, Nparam)

        iso.get_predictions.return_value = stellar_params
        iso.predictions = [
            "mini",
            "mass",
            "logl",
            "logt",
            "logr",
            "logg",
            "feh_surf",
            "afe_surf",
        ]
        iso.eep_u = np.array(
            [300, 350, 400, 450, 500]
        )  # Mock EEP values for binary calculations

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            # Mock neural network predictor with correct method
            mock_predictor = MagicMock()
            mock_predictor.sed.return_value = np.array(
                [15.0, 14.5, 14.0, 13.8, 13.5]
            )  # Realistic magnitudes
            mock_predictor.NFILT = 5  # Number of filters
            # sed_batch should return (N, Nfilt) array of magnitudes
            mock_predictor.sed_batch.return_value = np.tile(
                np.array([15.0, 14.5, 14.0, 13.8, 13.5]), (5, 1)
            )
            mock_nn.return_value = mock_predictor

            pop = StellarPop(isochrone=iso, verbose=False)
            pop.filters = ["u", "g", "r", "i", "z"]  # Set filters for testing
            pop.predictor = mock_predictor  # Explicitly set predictor

            return pop

    def test_get_seds_basic_functionality(self, mock_stellar_pop):
        """Test basic get_seds functionality."""
        seds, params, params2 = mock_stellar_pop.get_seds()

        # Check output types and shapes
        assert isinstance(seds, np.ndarray)
        assert isinstance(params, dict)  # Default return_dict=True
        assert isinstance(params2, dict)
        assert seds.ndim == 2  # N_stars x N_filters
        assert np.all(np.isfinite(seds))

    def test_get_seds_with_extinction(self, mock_stellar_pop):
        """Test get_seds with extinction parameters."""
        seds, params, params2 = mock_stellar_pop.get_seds(av=0.5, rv=3.1)

        assert isinstance(seds, np.ndarray)
        assert np.all(np.isfinite(seds))

    def test_get_seds_with_distance(self, mock_stellar_pop):
        """Test get_seds with distance modulus."""
        seds, params, params2 = mock_stellar_pop.get_seds(dist=1000.0)

        assert isinstance(seds, np.ndarray)
        assert np.all(np.isfinite(seds))

    def test_get_seds_binary_fraction(self, mock_stellar_pop):
        """Test get_seds with binary star fraction."""
        seds, params, params2 = mock_stellar_pop.get_seds(binary_fraction=0.3)

        assert isinstance(seds, np.ndarray)
        # Binary computation is complex - just check we get arrays back
        assert isinstance(params2, dict)  # Default return_dict=True
        # Should have attempted binary calculation
        assert len(params2) > 0


class TestPopulationsIntegration:
    """Integration tests between Isochrone and StellarPop classes."""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated Isochrone + StellarPop system."""
        # Mock the file loading and neural network for integration testing
        with (
            patch("h5py.File") as mock_file,
            patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn,
        ):

            # Mock isochrone data
            mock_context = MagicMock()
            mock_file.return_value.__enter__.return_value = mock_context
            feh_grid = np.array([-1.0, 0.0, 0.5])
            afe_grid = np.array([0.0, 0.4])
            loga_grid = np.array([8.5, 9.0, 9.5, 10.0])
            eep_grid = np.arange(202, 808)
            predictions = np.random.random((3, 2, 4, len(eep_grid), 8))

            mock_predictions = MagicMock()
            mock_predictions.__getitem__.return_value = predictions
            mock_predictions.attrs = {
                "labels": [
                    "mini",
                    "mass",
                    "logl",
                    "logt",
                    "logr",
                    "logg",
                    "feh_surf",
                    "afe_surf",
                ]
            }

            mock_context.__getitem__.side_effect = lambda key: {
                "feh": feh_grid,
                "afe": afe_grid,
                "loga": loga_grid,
                "eep": eep_grid,
                "predictions": mock_predictions,
            }[key]

            # Mock neural network predictor
            mock_predictor = MagicMock()
            mock_predictor.sed.return_value = np.array([15.0, 14.5, 14.0, 13.8, 13.5])
            mock_predictor.NFILT = 5
            # sed_batch must return (N, Nfilt) for N input stars
            mock_predictor.sed_batch.side_effect = lambda **kw: np.tile(
                np.array([15.0, 14.5, 14.0, 13.8, 13.5]),
                (len(kw["logt"]), 1),
            )
            mock_nn.return_value = mock_predictor

            # Create integrated system
            iso = Isochrone(verbose=False)
            pop = StellarPop(
                isochrone=iso, filters=["g", "r", "i", "z", "y"], verbose=False
            )
            pop.predictor = mock_predictor  # Explicitly set predictor

            return iso, pop

    def test_isochrone_to_stellarpop_workflow(self, integrated_system):
        """Test complete workflow from isochrone to stellar population."""
        iso, pop = integrated_system

        # Get stellar parameters from isochrone
        params = iso.get_predictions(feh=0.0, afe=0.0, loga=9.0)
        assert isinstance(params, np.ndarray)

        # Use population synthesizer to get photometry
        seds, params_out, params2 = pop.get_seds(feh=0.0, afe=0.0, loga=9.0)

        # Check that we get consistent results
        assert isinstance(seds, np.ndarray)
        assert isinstance(params_out, dict)  # Default return_dict=True
        assert seds.shape[1] == len(pop.filters)
        # Some stars may be filtered out due to mass bounds - check we have some finite values
        assert np.any(np.isfinite(seds))  # At least some stars should have valid SEDs

    def test_parameter_consistency(self, integrated_system):
        """Test that parameters are consistent between classes."""
        iso, pop = integrated_system

        # Same stellar parameters should give same base predictions
        feh, afe, loga = -0.5, 0.2, 9.5

        iso_params = iso.get_predictions(feh=feh, afe=afe, loga=loga)
        _, pop_params, _ = pop.get_seds(feh=feh, afe=afe, loga=loga)

        # Should have compatible shapes and finite values
        assert isinstance(iso_params, np.ndarray)
        assert isinstance(pop_params, dict)  # Default return_dict=True
        assert np.all(np.isfinite(iso_params))
        assert all(
            np.any(np.isfinite(v))
            for v in pop_params.values()
            if isinstance(v, np.ndarray)
        )


class TestPopulationsEdgeCases:
    """Test edge cases and error conditions."""

    def test_isochrone_extreme_parameters(self):
        """Test isochrone behavior with extreme parameters."""
        with patch("h5py.File") as mock_file:
            mock_context = MagicMock()
            mock_file.return_value.__enter__.return_value = mock_context
            feh_grid = np.array([-2.0, 0.0, 0.5])
            afe_grid = np.array([0.0, 0.4])
            loga_grid = np.array([8.0, 10.0])
            eep_grid = np.arange(202, 808)
            predictions = np.random.random((3, 2, 2, len(eep_grid), 8))

            mock_predictions = MagicMock()
            mock_predictions.__getitem__.return_value = predictions
            mock_predictions.attrs = {
                "labels": [
                    "mini",
                    "mass",
                    "logl",
                    "logt",
                    "logr",
                    "logg",
                    "feh_surf",
                    "afe_surf",
                ]
            }

            mock_context.__getitem__.side_effect = lambda key: {
                "feh": feh_grid,
                "afe": afe_grid,
                "loga": loga_grid,
                "eep": eep_grid,
                "predictions": mock_predictions,
            }[key]

            iso = Isochrone(verbose=False)

            # Test extreme values - mocked interpolator handles all values
            preds = iso.get_predictions(feh=-10.0)  # Extreme metallicity
            assert isinstance(preds, np.ndarray)  # Should return something from mock

    def test_stellarpop_binary_functionality(self):
        """Test StellarPop binary star functionality with realistic constraints."""
        # Create realistic mock data that supports binary star calculations
        iso = MagicMock()

        # Create internally consistent stellar evolution data
        # Primary stars: masses 0.8, 1.0, 1.2, 1.5, 2.0 solar masses
        # EEPs: corresponding evolutionary points on main sequence
        primary_masses = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 0.6, 0.7, 1.8, 2.2, 2.5])
        primary_eeps = np.array([350, 400, 450, 500, 550, 320, 330, 520, 580, 600])

        primary_params = np.array(
            [
                primary_masses,  # mini
                primary_masses
                * 0.95,  # mass (current mass, slightly less than initial)
                np.array([0.2, 0.5, 0.8, 1.2, 1.6, 0.0, 0.1, 1.4, 1.8, 2.0]),  # logl
                np.array(
                    [3.7, 3.75, 3.8, 3.85, 3.9, 3.65, 3.68, 3.88, 3.92, 3.95]
                ),  # logt
                np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.05, 0.08, 0.6, 0.8, 0.9]),  # logr
                np.array([4.6, 4.5, 4.4, 4.2, 4.0, 4.7, 4.65, 4.1, 3.9, 3.8]),  # logg
                np.zeros(10),  # feh_surf
                np.zeros(10),  # afe_surf
            ]
        ).T

        iso.predictions = [
            "mini",
            "mass",
            "logl",
            "logt",
            "logr",
            "logg",
            "feh_surf",
            "afe_surf",
        ]
        iso.eep_u = primary_eeps

        # Mock get_predictions to return appropriate data based on EEP input
        def mock_get_predictions(
            feh=0.0, afe=0.0, loga=8.5, eep=None, apply_corr=True, corr_params=None
        ):
            if eep is not None:
                # This is a call for secondary stars - create scaled-down parameters
                # Secondary masses should be binary_fraction * primary_masses
                secondary_params = primary_params.copy()
                # Scale masses and luminosities appropriately for secondaries
                secondary_params[
                    :, 0
                ] *= 0.7  # Secondary masses (will be set by binary_fraction)
                secondary_params[:, 1] *= 0.7  # Current masses
                secondary_params[:, 2] -= 0.5  # Lower luminosity for smaller stars
                secondary_params[
                    :, 5
                ] += 0.2  # Higher surface gravity for smaller stars
                return secondary_params
            else:
                # Normal call - return primary parameters
                return primary_params

        iso.get_predictions = mock_get_predictions

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_predictor = MagicMock()
            mock_predictor.sed.return_value = np.array([15.0, 14.5, 14.0, 13.8, 13.5])
            mock_predictor.NFILT = 5
            mock_predictor.sed_batch.side_effect = lambda **kw: np.tile(
                np.array([15.0, 14.5, 14.0, 13.8, 13.5]),
                (len(kw["logt"]), 1),
            )
            mock_nn.return_value = mock_predictor

            pop = StellarPop(isochrone=iso, verbose=False)
            pop.predictor = mock_predictor

            # Test single stars (no binaries)
            seds_single, params_single, params2_single = pop.get_seds(
                binary_fraction=0.0
            )

            # Test equal-mass binaries (simple case)
            seds_binary, params_binary, params2_binary = pop.get_seds(
                binary_fraction=1.0
            )

            # Verify basic structure
            assert isinstance(seds_single, np.ndarray)
            assert isinstance(seds_binary, np.ndarray)
            assert seds_single.shape == seds_binary.shape  # Same population size

            # For binaries, secondary parameters should be populated
            assert isinstance(params2_binary, dict)
            assert "mini" in params2_binary

            # Test moderate binary fraction
            seds_moderate, params_moderate, params2_moderate = pop.get_seds(
                binary_fraction=0.7
            )
            assert isinstance(seds_moderate, np.ndarray)

            # The test passes if we can call binary functionality without crashes
            # Detailed flux combination testing would require more complex mocking

    def test_empty_predictions(self):
        """Test behavior with empty prediction arrays."""
        with patch("h5py.File") as mock_file:
            mock_context = MagicMock()
            mock_file.return_value.__enter__.return_value = mock_context
            feh_grid = np.array([0.0])
            afe_grid = np.array([0.0])
            loga_grid = np.array([9.0])
            eep_grid = np.arange(202, 808)
            predictions = np.random.random((1, 1, 1, len(eep_grid), 8))

            mock_predictions = MagicMock()
            mock_predictions.__getitem__.return_value = predictions
            mock_predictions.attrs = {
                "labels": [
                    "mini",
                    "mass",
                    "logl",
                    "logt",
                    "logr",
                    "logg",
                    "feh_surf",
                    "afe_surf",
                ]
            }

            mock_context.__getitem__.side_effect = lambda key: {
                "feh": feh_grid,
                "afe": afe_grid,
                "loga": loga_grid,
                "eep": eep_grid,
                "predictions": mock_predictions,
            }[key]

            iso = Isochrone(predictions=[], verbose=False)  # Empty predictions

            # Should handle empty predictions gracefully
            assert iso.predictions == []


class TestPopulationsPerformance:
    """Test performance characteristics of populations module."""

    @pytest.fixture
    def performance_setup(self):
        """Setup for performance testing."""
        with (
            patch("h5py.File") as mock_file,
            patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn,
        ):

            # Large mock data for performance testing
            mock_context = MagicMock()
            mock_file.return_value.__enter__.return_value = mock_context
            feh_grid = np.linspace(-2, 1, 50)
            afe_grid = np.array([0.0, 0.2, 0.4])
            loga_grid = np.linspace(8, 10.5, 25)
            eep_grid = np.arange(202, 808)
            predictions = np.random.random((50, 3, 25, len(eep_grid), 8))

            mock_predictions = MagicMock()
            mock_predictions.__getitem__.return_value = predictions
            mock_predictions.attrs = {
                "labels": [
                    "mini",
                    "mass",
                    "logl",
                    "logt",
                    "logr",
                    "logg",
                    "feh_surf",
                    "afe_surf",
                ]
            }

            mock_context.__getitem__.side_effect = lambda key: {
                "feh": feh_grid,
                "afe": afe_grid,
                "loga": loga_grid,
                "eep": eep_grid,
                "predictions": mock_predictions,
            }[key]

            mock_predictor = MagicMock()
            mock_predictor.sed.return_value = np.array([15.0, 14.5, 14.0, 13.8, 13.5])
            mock_predictor.NFILT = 5
            mock_predictor.sed_batch.side_effect = lambda **kw: np.tile(
                np.array([15.0, 14.5, 14.0, 13.8, 13.5]),
                (len(kw["logt"]), 1),
            )
            mock_nn.return_value = mock_predictor

            iso = Isochrone(verbose=False)
            pop = StellarPop(isochrone=iso, verbose=False)
            pop.predictor = mock_predictor  # Explicitly set predictor

            return iso, pop

    def test_large_population_synthesis(self, performance_setup):
        """Test synthesis of large stellar populations."""
        iso, pop = performance_setup

        # Should handle large populations without crashing
        seds, params, params2 = pop.get_seds()

        assert isinstance(seds, np.ndarray)
        assert seds.shape[0] > 100  # Should have many stars (606 EEP grid points)
        assert seds.shape[1] == 5  # Should have 5 filters
        # Some stars may be filtered - just check we get reasonable output
        assert np.any(np.isfinite(seds))  # At least some stars should be finite

    def test_multiple_parameter_sets(self, performance_setup):
        """Test multiple parameter combinations efficiently."""
        iso, pop = performance_setup

        # Test multiple metallicities
        metallicities = [-1.0, -0.5, 0.0, 0.5]

        for feh in metallicities:
            params = iso.get_predictions(feh=feh, loga=9.0)
            assert isinstance(params, np.ndarray)
            assert np.all(np.isfinite(params))
