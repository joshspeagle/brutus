#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus photometric offset analysis.

This test suite includes:
1. Unit tests for PhotometricOffsetsConfig class
2. Comprehensive tests for photometric_offsets function
3. Parameter recovery tests using real get_seds function
4. Performance and edge case tests
5. Integration tests with other brutus components
"""

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def mock_data():
    """Create realistic mock data for testing."""
    np.random.seed(42)

    n_obj, n_filt, n_samp, n_models = 50, 5, 20, 100

    # Mock photometry
    phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
    err = 0.05 * phot + 0.01  # Realistic errors
    mask = np.random.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])

    # Mock models with realistic coefficients
    models = np.random.random((n_models, n_filt, 3))
    models[:, :, 0] = np.random.uniform(14, 20, (n_models, n_filt))  # Base mags
    models[:, :, 1] = np.random.uniform(0.5, 2.0, (n_models, n_filt))  # A(V) coeffs
    models[:, :, 2] = np.random.uniform(0.0, 0.3, (n_models, n_filt))  # R(V) coeffs

    # Mock fitted parameters
    idxs = np.random.randint(0, n_models, (n_obj, n_samp))
    reds = np.random.uniform(0.0, 3.0, (n_obj, n_samp))
    dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
    dists = np.random.uniform(0.5, 10.0, (n_obj, n_samp))

    return {
        "phot": phot,
        "err": err,
        "mask": mask,
        "models": models,
        "idxs": idxs,
        "reds": reds,
        "dreds": dreds,
        "dists": dists,
    }


@pytest.fixture
def simple_mocks():
    """Set up a simple, working get_seds mock for basic functionality tests."""

    def mock_get_seds(models_input, av=None, rv=None, return_flux=True):
        n_requested = len(models_input) if hasattr(models_input, "__len__") else 1
        # Infer n_filt from models_input shape
        if hasattr(models_input, "shape") and len(models_input.shape) >= 2:
            n_filt = models_input.shape[1]
        else:
            n_filt = 5  # fallback
        if return_flux:
            return np.random.uniform(0.1, 10.0, (n_requested, n_filt))
        else:
            return np.random.uniform(14, 20, (n_requested, n_filt))

    return mock_get_seds


class TestPhotometricOffsetsConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        config = PhotometricOffsetsConfig()

        assert config.min_bands_used == 4
        assert config.min_bands_unused == 3
        assert config.n_bootstrap == 300
        assert config.uncertainty_method == "bootstrap_iqr"
        assert config.progress_interval == 10
        assert config.use_vectorized_bootstrap == True
        assert config.random_seed is None
        assert config.validate_inputs == True

    def test_custom_config(self):
        """Test custom configuration values."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        config = PhotometricOffsetsConfig(
            min_bands_used=5,
            min_bands_unused=4,
            n_bootstrap=100,
            uncertainty_method="bootstrap_std",
            progress_interval=5,
            use_vectorized_bootstrap=False,
            random_seed=42,
            validate_inputs=False,
        )

        assert config.min_bands_used == 5
        assert config.min_bands_unused == 4
        assert config.n_bootstrap == 100
        assert config.uncertainty_method == "bootstrap_std"
        assert config.progress_interval == 5
        assert config.use_vectorized_bootstrap == False
        assert config.random_seed == 42
        assert config.validate_inputs == False

    def test_config_validation(self):
        """Test configuration validation."""
        from brutus.analysis.offsets import PhotometricOffsetsConfig

        # Test invalid min_bands_used
        with pytest.raises(ValueError, match="min_bands_used must be >= 1"):
            PhotometricOffsetsConfig(min_bands_used=0)

        # Test invalid uncertainty_method
        with pytest.raises(ValueError, match="Unknown uncertainty_method"):
            PhotometricOffsetsConfig(uncertainty_method="invalid")

        # Test invalid n_bootstrap
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            PhotometricOffsetsConfig(n_bootstrap=0)

        # Test invalid progress_interval
        with pytest.raises(ValueError, match="progress_interval must be >= 0"):
            PhotometricOffsetsConfig(progress_interval=-1)


class TestPhotometricOffsetsCore:
    """Test core photometric_offsets functionality."""

    def test_basic_functionality(self, mock_data, simple_mocks):
        """Test basic function execution."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            config = PhotometricOffsetsConfig(
                n_bootstrap=10,
                progress_interval=0,
                validate_inputs=False,
                min_bands_used=2,
                min_bands_unused=1,
            )

            offsets, errors, n_used = photometric_offsets(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
                config=config,
                verbose=False,
            )

            # Check output shapes
            assert offsets.shape == (5,)
            assert errors.shape == (5,)
            assert n_used.shape == (5,)

            # Check output types and ranges
            assert np.all(np.isfinite(offsets))
            assert np.all(errors >= 0)
            assert np.all(n_used >= 0)

    def test_uncertainty_methods(self, mock_data, simple_mocks):
        """Test different uncertainty estimation methods."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            methods = ["bootstrap_std", "bootstrap_iqr"]
            results = {}

            for method in methods:
                config = PhotometricOffsetsConfig(
                    n_bootstrap=15,
                    uncertainty_method=method,
                    progress_interval=0,
                    validate_inputs=False,
                )

                offsets, errors, n_used = photometric_offsets(
                    mock_data["phot"],
                    mock_data["err"],
                    mock_data["mask"],
                    mock_data["models"],
                    mock_data["idxs"],
                    mock_data["reds"],
                    mock_data["dreds"],
                    mock_data["dists"],
                    config=config,
                    verbose=False,
                )

                results[method] = {"offsets": offsets, "errors": errors}

            # All methods should produce finite results
            for method, result in results.items():
                assert np.all(
                    np.isfinite(result["offsets"])
                ), f"Method {method} produced non-finite offsets"
                assert np.all(
                    result["errors"] >= 0
                ), f"Method {method} produced negative errors"

    def test_vectorized_vs_loop_bootstrap(self, mock_data, simple_mocks):
        """Test vectorized vs loop bootstrap implementations."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            for use_vectorized in [True, False]:
                config = PhotometricOffsetsConfig(
                    n_bootstrap=10,
                    use_vectorized_bootstrap=use_vectorized,
                    progress_interval=0,
                    validate_inputs=False,
                )

                offsets, errors, n_used = photometric_offsets(
                    mock_data["phot"],
                    mock_data["err"],
                    mock_data["mask"],
                    mock_data["models"],
                    mock_data["idxs"],
                    mock_data["reds"],
                    mock_data["dreds"],
                    mock_data["dists"],
                    config=config,
                    verbose=False,
                )

                # Both should produce finite results
                assert np.all(np.isfinite(offsets))
                assert np.all(errors >= 0)

    def test_optional_parameters(self, mock_data, simple_mocks):
        """Test optional parameter handling."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            config = PhotometricOffsetsConfig(
                n_bootstrap=10, progress_interval=0, validate_inputs=False
            )

            # Test with all optional parameters
            n_obj, n_filt = mock_data["phot"].shape
            n_samp = mock_data["idxs"].shape[1]

            sel = np.random.choice([True, False], n_obj, p=[0.8, 0.2])
            weights = np.random.uniform(0.5, 1.5, (n_obj, n_samp))
            mask_fit = np.random.choice([True, False], n_filt, p=[0.7, 0.3])
            old_offsets = np.random.uniform(0.9, 1.1, n_filt)

            offsets, errors, n_used = photometric_offsets(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
                sel=sel,
                weights=weights,
                mask_fit=mask_fit,
                old_offsets=old_offsets,
                dim_prior=True,
                config=config,
                verbose=False,
            )

            assert np.all(np.isfinite(offsets))
            assert np.all(errors >= 0)

    def test_prior_application(self, mock_data, simple_mocks):
        """Test Gaussian prior application."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            config = PhotometricOffsetsConfig(
                n_bootstrap=10, progress_interval=0, validate_inputs=False
            )

            # Test without priors
            offsets_no_prior, errors_no_prior, _ = photometric_offsets(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
                config=config,
                verbose=False,
            )

            # Test with priors
            prior_mean = np.ones(5) * 1.05
            prior_std = np.ones(5) * 0.02

            offsets_with_prior, errors_with_prior, _ = photometric_offsets(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
                prior_mean=prior_mean,
                prior_std=prior_std,
                config=config,
                verbose=False,
            )

            # Both should produce finite results
            assert np.all(np.isfinite(offsets_no_prior))
            assert np.all(np.isfinite(offsets_with_prior))
            assert np.all(errors_no_prior >= 0)
            assert np.all(errors_with_prior >= 0)

            # Results should be different when priors are applied
            assert not np.allclose(offsets_no_prior, offsets_with_prior, rtol=0.1)


class TestInputValidation:
    """Test input validation functionality."""

    def test_array_shape_validation(self, mock_data):
        """Test validation of array shapes."""
        from brutus.analysis.offsets import _validate_inputs

        # Test correct shapes (should not raise)
        _validate_inputs(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
        )

        # Test incorrect err shape
        with pytest.raises(ValueError, match="err shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"][:25],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test incorrect mask shape
        with pytest.raises(ValueError, match="mask shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"][:25],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

    def test_array_value_validation(self, mock_data):
        """Test validation of array values."""
        from brutus.analysis.offsets import _validate_inputs

        # Test negative errors
        bad_err = -np.abs(mock_data["err"])
        with pytest.raises(ValueError, match="err must be positive"):
            _validate_inputs(
                mock_data["phot"],
                bad_err,
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test invalid mask values
        bad_mask = mock_data["mask"].copy().astype(float)
        bad_mask[0, 0] = 0.5  # Should be 0 or 1
        with pytest.raises(ValueError, match="mask must contain only 0s and 1s"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                bad_mask,
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test negative distances
        bad_dists = -np.abs(mock_data["dists"])
        with pytest.raises(ValueError, match="dists must be positive"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                bad_dists,
            )

        # Test non-finite photometry
        bad_phot = mock_data["phot"].copy()
        bad_phot[0, 0] = np.nan
        with pytest.raises(ValueError, match="phot contains non-finite values"):
            _validate_inputs(
                bad_phot,
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )


class TestParameterRecovery:
    """Test parameter recovery using real get_seds function."""

    def test_unity_recovery_real_get_seds(self):
        """Test unity recovery using real get_seds function."""
        # Import should always work
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            get_seds,
            photometric_offsets,
        )

        n_obj, n_filt = 60, 5
        n_models, n_samp = 25, 12

        # Create realistic model grid
        np.random.seed(42)
        models = np.random.random((n_models, n_filt, 3))
        models[:, :, 0] = np.random.uniform(15, 19, (n_models, n_filt))
        models[:, :, 1] = np.random.uniform(0.6, 1.8, (n_models, n_filt))
        models[:, :, 2] = np.random.uniform(0.0, 0.25, (n_models, n_filt))

        # True parameters for data generation
        true_model_idxs = np.random.randint(0, n_models, n_obj)
        true_avs = np.random.uniform(0.0, 1.2, n_obj)
        true_rvs = np.random.uniform(3.1, 3.5, n_obj)
        true_dists = np.random.uniform(0.9, 1.1, n_obj)

        # Generate observed photometry using real get_seds
        observed_phot = np.zeros((n_obj, n_filt))
        for i in range(n_obj):
            true_sed = get_seds(
                models[true_model_idxs[i : i + 1]],
                av=np.array([true_avs[i]]),
                rv=np.array([true_rvs[i]]),
                return_flux=True,
            )
            observed_phot[i] = true_sed[0] / (true_dists[i] ** 2)

        # Add realistic noise
        noise = np.random.normal(0, 0.025 * observed_phot)
        observed_phot += noise
        observed_phot = np.maximum(observed_phot, 0.1 * np.abs(observed_phot))

        err = 0.03 * observed_phot
        mask = np.ones((n_obj, n_filt), dtype=int)

        # Create samples centered on true values
        idxs = np.zeros((n_obj, n_samp), dtype=int)
        reds = np.zeros((n_obj, n_samp))
        dreds = np.zeros((n_obj, n_samp))
        dists = np.zeros((n_obj, n_samp))

        for i in range(n_obj):
            idxs[i] = true_model_idxs[i]
            reds[i] = np.random.normal(true_avs[i], 0.015, n_samp)
            dreds[i] = np.random.normal(true_rvs[i], 0.008, n_samp)
            dists[i] = np.random.normal(true_dists[i], 0.008, n_samp)

            reds[i] = np.clip(reds[i], 0.0, 2.5)
            dreds[i] = np.clip(dreds[i], 2.9, 3.9)
            dists[i] = np.clip(dists[i], 0.6, 1.5)

        # Run the algorithm (real likelihood reweighting)
        config = PhotometricOffsetsConfig(
            n_bootstrap=20,
            min_bands_used=3,
            min_bands_unused=2,
            validate_inputs=False,
            progress_interval=0,
        )

        offsets, errors, n_used = photometric_offsets(
            observed_phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )

        # Check unity recovery
        max_deviation = np.max(np.abs(offsets - 1.0))
        max_rel_error = np.max(np.abs((offsets - 1.0) / 1.0))

        # Unity recovery should be very accurate
        assert (
            max_rel_error < 0.05
        ), f"Unity recovery failed: max relative error {max_rel_error:.3f} > 0.05"
        assert np.all(np.isfinite(offsets)), "Offsets contain non-finite values"
        assert np.all(errors >= 0), "Errors must be non-negative"

    def test_systematic_offset_recovery(self):
        """Test recovery of known systematic offsets."""
        # Import should always work
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            get_seds,
            photometric_offsets,
        )

        # Apply known systematic offsets
        true_offsets = np.array([1.05, 0.95, 1.08, 0.92, 1.02])
        expected_corrections = 1.0 / true_offsets

        n_obj, n_filt = 80, 5
        n_models, n_samp = 20, 10

        # Create model grid
        np.random.seed(123)
        models = np.random.random((n_models, n_filt, 3))
        models[:, :, 0] = np.random.uniform(15, 18, (n_models, n_filt))
        models[:, :, 1] = np.random.uniform(0.7, 1.6, (n_models, n_filt))
        models[:, :, 2] = np.random.uniform(0.0, 0.2, (n_models, n_filt))

        # True parameters
        true_model_idxs = np.random.randint(0, n_models, n_obj)
        true_avs = np.random.uniform(0.0, 1.0, n_obj)
        true_rvs = np.random.uniform(3.15, 3.45, n_obj)
        true_dists = np.random.uniform(0.95, 1.05, n_obj)

        # Generate photometry and apply systematic offsets
        observed_phot = np.zeros((n_obj, n_filt))
        for i in range(n_obj):
            true_sed = get_seds(
                models[true_model_idxs[i : i + 1]],
                av=np.array([true_avs[i]]),
                rv=np.array([true_rvs[i]]),
                return_flux=True,
            )
            base_flux = true_sed[0] / (true_dists[i] ** 2)
            observed_phot[i] = base_flux * true_offsets  # Apply systematic errors

        # Add noise
        noise = np.random.normal(0, 0.02 * observed_phot)
        observed_phot += noise
        observed_phot = np.maximum(observed_phot, 0.05 * np.abs(observed_phot))

        err = 0.025 * observed_phot
        mask = np.ones((n_obj, n_filt), dtype=int)

        # Create samples
        idxs = np.zeros((n_obj, n_samp), dtype=int)
        reds = np.zeros((n_obj, n_samp))
        dreds = np.zeros((n_obj, n_samp))
        dists = np.zeros((n_obj, n_samp))

        for i in range(n_obj):
            idxs[i] = true_model_idxs[i]
            reds[i] = np.random.normal(true_avs[i], 0.01, n_samp)
            dreds[i] = np.random.normal(true_rvs[i], 0.005, n_samp)
            dists[i] = np.random.normal(true_dists[i], 0.005, n_samp)

            reds[i] = np.clip(reds[i], 0.0, 2.0)
            dreds[i] = np.clip(dreds[i], 3.0, 3.8)
            dists[i] = np.clip(dists[i], 0.7, 1.3)

        # Run algorithm (real likelihood reweighting)
        config = PhotometricOffsetsConfig(
            n_bootstrap=25,
            min_bands_used=3,
            min_bands_unused=2,
            validate_inputs=False,
            progress_interval=0,
        )

        recovered_corrections, errors, n_used = photometric_offsets(
            observed_phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )

        # Check recovery
        abs_errors = np.abs(recovered_corrections - expected_corrections)
        rel_errors = abs_errors / expected_corrections
        max_rel_error = np.max(rel_errors)

        # Systematic offset recovery should be accurate
        assert (
            max_rel_error < 0.15
        ), f"Systematic offset recovery failed: max relative error {max_rel_error:.3f} > 0.15"
        assert np.all(
            np.isfinite(recovered_corrections)
        ), "Recovered corrections contain non-finite values"
        assert np.all(errors >= 0), "Errors must be non-negative"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_valid_objects(self, simple_mocks):
        """Test behavior when no objects meet selection criteria."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            # Create data where no objects meet minimum band requirements
            n_obj, n_filt = 10, 5
            phot = np.random.uniform(0.1, 1.0, (n_obj, n_filt))
            err = 0.1 * phot
            mask = np.zeros((n_obj, n_filt), dtype=bool)  # No detections

            models = np.random.random((10, n_filt, 3))
            idxs = np.random.randint(0, 10, (n_obj, 5))
            reds = np.random.uniform(0, 1, (n_obj, 5))
            dreds = np.random.uniform(3, 4, (n_obj, 5))
            dists = np.random.uniform(1, 5, (n_obj, 5))

            config = PhotometricOffsetsConfig(
                n_bootstrap=5,
                min_bands_used=3,
                progress_interval=0,
                validate_inputs=False,
            )

            offsets, errors, n_used = photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                config=config,
                verbose=False,
            )

            # Should handle gracefully
            assert offsets.shape == (n_filt,)
            assert np.all(n_used == 0)  # No objects used
            assert np.all(offsets == 1.0)  # Default values
            # No data and no prior: infinite (not zero) uncertainty, so
            # the placeholder offsets cannot pose as measurements
            assert np.all(np.isinf(errors))


class TestPerformance:
    """Performance tests."""

    def test_large_dataset_performance(self, simple_mocks):
        """Test performance with larger datasets."""
        mock_get_seds = simple_mocks

        with patch("brutus.analysis.offsets.get_seds", side_effect=mock_get_seds):

            from brutus.analysis.offsets import (
                PhotometricOffsetsConfig,
                photometric_offsets,
            )

            # Create larger dataset
            n_obj, n_filt, n_samp = 500, 8, 50

            phot = np.random.uniform(0.1, 10.0, (n_obj, n_filt))
            err = 0.1 * phot
            mask = np.random.choice([False, True], (n_obj, n_filt), p=[0.1, 0.9])

            models = np.random.random((200, n_filt, 3))
            idxs = np.random.randint(0, 200, (n_obj, n_samp))
            reds = np.random.uniform(0, 3, (n_obj, n_samp))
            dreds = np.random.uniform(2.5, 4.5, (n_obj, n_samp))
            dists = np.random.uniform(0.5, 10, (n_obj, n_samp))

            config = PhotometricOffsetsConfig(
                n_bootstrap=30,
                progress_interval=0,
                validate_inputs=False,
                use_vectorized_bootstrap=True,
            )

            # This should complete in reasonable time
            offsets, errors, n_used = photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                config=config,
                verbose=False,
            )

            assert offsets.shape == (n_filt,)
            assert np.all(np.isfinite(offsets))


if __name__ == "__main__":
    pytest.main([__file__])
