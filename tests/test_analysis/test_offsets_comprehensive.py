#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests of `photometric_offsets` and `PhotometricOffsetsConfig`."""

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


TRUE_OFFSETS = np.array([1.10, 0.90, 1.08, 0.92, 1.05])


@pytest.fixture
def injected_offsets():
    """Synthetic fluxes from real `get_seds` SEDs with known offsets injected.

    Each object's samples sit at its true model, A(V), R(V) and distance, so
    `photometric_offsets` should return ``1 / TRUE_OFFSETS``.
    """
    from brutus.core.sed_utils import get_seds

    rng = np.random.RandomState(0)
    n_obj, n_filt, n_models, n_samp = 80, len(TRUE_OFFSETS), 20, 10
    models = np.zeros((n_models, n_filt, 3))
    models[:, :, 0] = rng.uniform(15, 18, (n_models, n_filt))
    models[:, :, 1] = rng.uniform(0.7, 1.6, (n_models, n_filt))
    models[:, :, 2] = rng.uniform(0.0, 0.2, (n_models, n_filt))
    idx = rng.randint(0, n_models, n_obj)
    av = rng.uniform(0.0, 1.0, n_obj)
    rv = rng.uniform(3.15, 3.45, n_obj)
    dist = rng.uniform(0.95, 1.05, n_obj)

    flux = get_seds(models[idx], av=av, rv=rv, return_flux=True) / dist[:, None] ** 2
    err = 0.02 * flux
    phot = flux * TRUE_OFFSETS + rng.normal(0.0, err)

    def per_sample(x):
        return np.repeat(x[:, None], n_samp, axis=1)

    return {
        "phot": phot,
        "err": err,
        "mask": np.ones((n_obj, n_filt), dtype=int),
        "models": models,
        "idxs": per_sample(idx),
        "reds": per_sample(av),
        "dreds": per_sample(rv),
        "dists": per_sample(dist),
    }


# The bootstrap errors are about 0.003, so 0.015 is about 5 sigma; the
# smallest injected offset moves the answer by 0.048.
RECOVERY_ATOL = 0.015


def _assert_recovered(offsets):
    np.testing.assert_allclose(offsets, 1.0 / TRUE_OFFSETS, atol=RECOVERY_ATOL)


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
    """photometric_offsets recovers offsets injected into synthetic data."""

    @staticmethod
    def _run(data, **kwargs):
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            photometric_offsets,
        )

        config = PhotometricOffsetsConfig(
            n_bootstrap=50,
            progress_interval=0,
            uncertainty_method=kwargs.pop("uncertainty_method", "bootstrap_iqr"),
            use_vectorized_bootstrap=kwargs.pop("use_vectorized_bootstrap", True),
        )
        return photometric_offsets(
            **data, config=config, verbose=False, rng=np.random.default_rng(1), **kwargs
        )

    @pytest.mark.parametrize("method", ["bootstrap_std", "bootstrap_iqr"])
    @pytest.mark.parametrize("vectorized", [True, False])
    def test_injected_offsets_are_recovered(self, injected_offsets, method, vectorized):
        offsets, errors, n_used = self._run(
            injected_offsets,
            uncertainty_method=method,
            use_vectorized_bootstrap=vectorized,
        )
        _assert_recovered(offsets)
        assert np.all((errors > 0) & (errors < RECOVERY_ATOL))
        np.testing.assert_array_equal(n_used, len(injected_offsets["phot"]))

    def test_selection_and_weights_restrict_the_objects_used(self, injected_offsets):
        n_obj, n_samp = injected_offsets["idxs"].shape
        sel = np.arange(n_obj) % 4 != 0
        weights = np.random.RandomState(2).uniform(0.5, 1.5, (n_obj, n_samp))
        offsets, _, n_used = self._run(injected_offsets, sel=sel, weights=weights)
        _assert_recovered(offsets)
        np.testing.assert_array_equal(n_used, sel.sum())

    def test_tight_prior_pulls_offsets_to_its_mean(self, injected_offsets):
        n_filt = len(TRUE_OFFSETS)
        offsets, errors, _ = self._run(
            injected_offsets,
            prior_mean=np.ones(n_filt),
            prior_std=np.full(n_filt, 1e-4),
        )
        np.testing.assert_allclose(offsets, 1.0, atol=1e-3)
        assert np.all(errors < 1e-4)


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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_valid_objects(self):
        """Test behavior when no objects meet selection criteria."""
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            photometric_offsets,
        )

        rng = np.random.RandomState(3)
        n_obj, n_filt = 10, 5
        phot = rng.uniform(0.1, 1.0, (n_obj, n_filt))
        err = 0.1 * phot
        mask = np.zeros((n_obj, n_filt), dtype=bool)  # No detections

        models = rng.random_sample((10, n_filt, 3))
        idxs = rng.randint(0, 10, (n_obj, 5))
        reds = rng.uniform(0, 1, (n_obj, 5))
        dreds = rng.uniform(3, 4, (n_obj, 5))
        dists = rng.uniform(1, 5, (n_obj, 5))

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


if __name__ == "__main__":
    pytest.main([__file__])
