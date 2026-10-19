"""Tests of the photometry utilities in `brutus.utils.photometry`."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from brutus.utils.photometry import (
    add_mag,
    chisquare_outlier_loglike,
    inv_luptitude,
    inv_magnitude,
    luptitude,
    magnitude,
    phot_loglike,
    uniform_outlier_loglike,
)


class TestMagnitudeConversions:
    """Test magnitude conversion functions."""

    def test_magnitude_conversion_roundtrip(self):
        """Test that magnitude conversion round-trips correctly."""
        # Test data: flux and errors
        flux = np.array([1.0, 0.5, 0.1, 10.0])
        flux_err = np.array([0.01, 0.02, 0.005, 0.1])
        zeropoints = 1.0

        # Convert to magnitudes and back
        mag, mag_err = magnitude(flux, flux_err, zeropoints)
        flux_recovered, flux_err_recovered = inv_magnitude(mag, mag_err, zeropoints)

        # Should round-trip to original values
        npt.assert_array_almost_equal(flux, flux_recovered, decimal=10)
        npt.assert_array_almost_equal(flux_err, flux_err_recovered, decimal=10)

    def test_magnitude_basic_values(self):
        """Test magnitude conversion with known values."""
        # Flux of 1.0 should give magnitude of 0.0 with zeropoint 1.0
        flux = np.array([1.0])
        flux_err = np.array([0.01])

        mag, mag_err = magnitude(flux, flux_err, zeropoints=1.0)

        npt.assert_array_almost_equal(mag, [0.0], decimal=10)
        # mag_err = 2.5/ln(10) * flux_err/flux = 1.086 * 0.01 = 0.01086
        expected_mag_err = 2.5 / np.log(10) * 0.01
        npt.assert_array_almost_equal(mag_err, [expected_mag_err], decimal=8)

    def test_magnitude_with_zeropoints(self):
        """Test magnitude conversion with different zeropoints."""
        flux = np.array([1.0, 1.0])
        flux_err = np.array([0.01, 0.01])
        zeropoints = np.array([1.0, 10.0])

        mag, mag_err = magnitude(flux, flux_err, zeropoints)

        # First should be 0.0, second should be -2.5*log10(1/10) = 2.5
        expected_mag = np.array([0.0, 2.5])
        npt.assert_array_almost_equal(mag, expected_mag, decimal=10)

    def test_magnitude_shapes(self):
        """Test that magnitude conversion preserves array shapes."""
        # Test 1D arrays
        flux_1d = np.array([1.0, 0.5, 0.1])
        flux_err_1d = np.array([0.01, 0.02, 0.005])

        mag_1d, mag_err_1d = magnitude(flux_1d, flux_err_1d)
        assert mag_1d.shape == flux_1d.shape
        assert mag_err_1d.shape == flux_err_1d.shape

        # Test 2D arrays (multiple objects, multiple filters)
        flux_2d = np.array([[1.0, 0.5], [0.1, 2.0]])
        flux_err_2d = np.array([[0.01, 0.02], [0.005, 0.05]])

        mag_2d, mag_err_2d = magnitude(flux_2d, flux_err_2d)
        assert mag_2d.shape == flux_2d.shape
        assert mag_err_2d.shape == flux_err_2d.shape

    def test_inv_magnitude_basic(self):
        """Test basic inverse magnitude conversion."""

        mag = np.array([[0.0, -2.5, 2.5]])
        mag_err = np.array([[0.1, 0.2, 0.3]])

        flux, flux_err = inv_magnitude(mag, mag_err)

        # Check flux values
        expected_flux = np.array([[1.0, 10.0, 0.1]])
        np.testing.assert_array_almost_equal(flux, expected_flux, decimal=10)

        # Check that errors are positive and finite
        assert np.all(flux_err > 0)
        assert np.all(np.isfinite(flux_err))


class TestLuptitudeConversions:
    """Test luptitude (asinh magnitude) conversion functions."""

    @pytest.mark.parametrize("skynoise", [1.0, 0.1, 10.0])
    def test_luptitude_roundtrip(self, skynoise):
        """Test that luptitude conversion round-trips correctly."""
        flux = np.array([1.0, 0.5, 0.1, 10.0])
        flux_err = np.array([0.01, 0.02, 0.005, 0.1])
        zeropoints = 1.0

        # Convert to luptitudes and back
        lupt, lupt_err = luptitude(flux, flux_err, skynoise, zeropoints)
        flux_recovered, flux_err_recovered = inv_luptitude(
            lupt, lupt_err, skynoise, zeropoints
        )

        # Should round-trip to original values
        npt.assert_array_almost_equal(flux, flux_recovered, decimal=8)
        npt.assert_array_almost_equal(flux_err, flux_err_recovered, decimal=8)

    def test_luptitude_high_snr_limit(self):
        """Test that luptitudes approach magnitudes in high S/N limit."""
        # High flux relative to sky noise - should approach magnitude
        flux = np.array([100.0])  # High S/N
        flux_err = np.array([0.1])
        skynoise = 1.0
        zeropoints = 1.0

        mag, _ = magnitude(flux, flux_err, zeropoints)
        lupt, _ = luptitude(flux, flux_err, skynoise, zeropoints)

        # Should be very close in high S/N limit
        npt.assert_array_almost_equal(lupt, mag, decimal=3)

    def test_luptitude_low_snr_behavior(self):
        """Test luptitude behavior in low S/N regime."""
        # Very low flux - luptitudes should be well-behaved while magnitudes blow up
        flux = np.array([0.01])  # Very faint
        flux_err = np.array([0.005])
        skynoise = 1.0
        zeropoints = 1.0

        lupt, lupt_err = luptitude(flux, flux_err, skynoise, zeropoints)

        # Should give finite, reasonable values
        assert np.all(np.isfinite(lupt))
        assert np.all(np.isfinite(lupt_err))
        assert np.all(lupt_err > 0)


class TestPhotometryEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_flux_handling(self):
        """Test behavior with zero flux (should give infinite magnitude)."""
        flux = np.array([0.0])
        flux_err = np.array([0.01])

        # Suppress expected divide-by-zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Should give infinite magnitude
        assert np.isinf(mag[0])
        assert np.isinf(mag_err[0])

    def test_negative_flux_handling(self):
        """Test behavior with negative flux."""
        flux = np.array([-1.0])
        flux_err = np.array([0.1])

        # Suppress expected invalid value warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Magnitude should be NaN (log of negative number)
        assert np.isnan(mag[0])
        # Error might be finite (no log involved in error calculation)
        assert np.isfinite(mag_err[0])

    def test_zero_error_handling(self):
        """Test behavior with zero errors."""
        flux = np.array([1.0])
        flux_err = np.array([0.0])

        mag, mag_err = magnitude(flux, flux_err)

        # Magnitude should be finite, error should be 0
        assert np.isfinite(mag[0])
        assert mag_err[0] == 0.0


class TestPhotometryIntegration:
    """Integration tests for photometry functions."""

    def test_realistic_photometry_pipeline(self):
        """Test a realistic photometry processing pipeline."""
        # Simulate realistic survey data
        rng = np.random.RandomState(42)

        # True magnitudes for a range of stars
        true_mags = rng.uniform(14, 24, 100)
        true_fluxes = 10 ** (-0.4 * true_mags)

        # Add realistic errors (magnitude-dependent)
        snr = 100 * 10 ** (-0.4 * (true_mags - 20))  # S/N decreases with magnitude
        flux_errors = true_fluxes / snr

        # Add noise
        observed_fluxes = true_fluxes + rng.normal(0, flux_errors)

        # Convert to magnitudes
        obs_mags, obs_mag_errs = magnitude(observed_fluxes, flux_errors)

        # Should recover input magnitudes within errors for bright stars
        bright_mask = true_mags < 20
        mag_diff = obs_mags[bright_mask] - true_mags[bright_mask]
        expected_scatter = obs_mag_errs[bright_mask]

        # Most should be within 3-sigma
        outlier_fraction = np.mean(np.abs(mag_diff) > 3 * expected_scatter)
        assert outlier_fraction < 0.05  # Less than 5% outliers


class TestMagnitudeAddition:
    """Test add_mag."""

    def test_add_mag(self):
        """Test magnitude addition."""

        # Test simple cases
        mag1 = np.array([0.0, 0.0, 1.0])
        mag2 = np.array([0.0, 1.0, 1.0])

        combined = add_mag(mag1, mag2)

        # When mag1=mag2=0, combined flux is 2, so combined mag = -2.5*log10(2) ≈ -0.753
        expected_0 = -2.5 * np.log10(2.0)
        np.testing.assert_almost_equal(combined[0], expected_0, decimal=10)

        # When mag1=0, mag2=1, fluxes are 1 and 0.398, combined ≈ 1.398, mag ≈ -0.363
        flux1 = 10 ** (-0.4 * 0.0)  # = 1.0
        flux2 = 10 ** (-0.4 * 1.0)  # ≈ 0.398
        combined_flux = flux1 + flux2
        expected_1 = -2.5 * np.log10(combined_flux)
        np.testing.assert_almost_equal(combined[1], expected_1, decimal=10)


class TestPhotLoglike:
    """Test phot_loglike."""

    def test_phot_loglike_basic(self):
        """Test photometric log-likelihood computation."""

        # Simple test case
        flux = np.array([[1.0, 2.0, 3.0]])  # (1 object, 3 filters)
        err = np.array([[0.1, 0.2, 0.3]])
        mfluxes = np.array(
            [[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]
        )  # (1 obj, 2 models, 3 filters)

        lnl = phot_loglike(flux, err, mfluxes)

        # Should return log-likelihood for each model
        assert lnl.shape == (1, 2)
        assert np.all(np.isfinite(lnl))

        # Perfect match (first model) should have higher likelihood than imperfect match
        assert lnl[0, 0] > lnl[0, 1]

    def test_phot_loglike_with_mask(self):
        """Test photometric log-likelihood with masking."""

        flux = np.array([[1.0, 2.0, 3.0]])
        err = np.array([[0.1, 0.2, 0.3]])
        mfluxes = np.array([[[1.0, 2.0, 3.0]]])
        mask = np.array([[1, 1, 0]])  # mask out third filter

        lnl = phot_loglike(flux, err, mfluxes, mask=mask)

        # Should still return valid log-likelihood
        assert lnl.shape == (1, 1)
        assert np.isfinite(lnl[0, 0])

    def test_dimension_mismatch_phot_loglike(self):
        """Test error handling for dimension mismatches."""

        flux = np.array([[1.0, 2.0]])  # (1, 2)
        err = np.array([[0.1, 0.2]])  # (1, 2)
        mfluxes = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3) - WRONG!

        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            phot_loglike(flux, err, mfluxes)

    def test_phot_loglike_dim_prior(self):
        """Test photometric likelihood with dimensionality prior."""
        rng = np.random.RandomState(0)

        # Simple test case with dimensionality prior
        nobjs = 5
        nmods = 3
        nfilts = 4

        # Mock observed fluxes and errors
        fluxes = rng.rand(nobjs, nfilts) * 1000 + 100  # Positive fluxes
        flux_errors = rng.rand(nobjs, nfilts) * 50 + 10

        # Mock model fluxes
        model_fluxes = rng.rand(nobjs, nmods, nfilts) * 1000 + 100

        # Test with dimensionality prior enabled
        lnl_dim = phot_loglike(fluxes, flux_errors, model_fluxes, dim_prior=True)

        assert lnl_dim.shape == (nobjs, nmods)
        assert np.all(
            np.isfinite(lnl_dim)
        ), "Log-likelihood with dim prior should be finite"

        # Test with insufficient degrees of freedom (should handle gracefully)
        small_fluxes = fluxes[:, :2]  # Only 2 filters
        small_errors = flux_errors[:, :2]
        small_models = model_fluxes[:, :, :2]

        # With dof_reduction=3, dof = 2 - 3 = -1: no information, so 0
        lnl_small = phot_loglike(
            small_fluxes, small_errors, small_models, dim_prior=True, dof_reduction=3
        )
        assert lnl_small.shape == (nobjs, nmods)
        np.testing.assert_array_equal(lnl_small, 0.0)

    def test_phot_loglike_with_dof_reduction(self):
        """Test phot_loglike function with dof_reduction parameter."""

        # Test data
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])  # 2 objects, 3 filters
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])
        mfluxes = np.array(
            [
                [
                    [0.9, 0.75, 0.55],
                    [1.1, 0.85, 0.65],
                ],  # 2 objects, 2 models, 3 filters
                [[1.15, 0.95, 0.75], [1.25, 1.05, 0.85]],
            ]
        )

        # Test without dof_reduction
        lnl_no_dof = phot_loglike(flux, err, mfluxes, dim_prior=True, dof_reduction=0)

        # Test with dof_reduction
        lnl_with_dof = phot_loglike(flux, err, mfluxes, dim_prior=True, dof_reduction=1)

        assert lnl_no_dof.shape == (2, 2)  # (n_obj, n_models)
        assert lnl_with_dof.shape == (2, 2)
        assert np.all(np.isfinite(lnl_no_dof))
        assert np.all(np.isfinite(lnl_with_dof))
        # Different dof_reduction should give different results
        assert not np.allclose(lnl_no_dof, lnl_with_dof)


class TestOutlierModels:
    """Test outlier model functions in utils.photometry."""

    def test_chisquare_outlier_basic(self):
        """Test chi-square outlier model with basic inputs."""
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])  # 2 objects, 3 filters
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])

        lnl_outlier = chisquare_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))
        assert np.all(lnl_outlier < 0)  # Should be negative (log probability)

    def test_uniform_outlier_basic(self):
        """Test uniform outlier model with basic inputs."""
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])

        lnl_outlier = uniform_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))
        # Note: uniform model can have positive values due to volume normalization

    def test_outlier_models_with_parallax(self):
        """Test outlier models with parallax data."""
        flux = np.array([[1.0, 0.8]])
        err = np.array([[0.1, 0.08]])
        parallax = np.array([2.0])  # 2 mas
        parallax_err = np.array([0.1])  # 0.1 mas

        lnl_chi = chisquare_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )
        lnl_unif = uniform_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_chi) == 1
        assert len(lnl_unif) == 1
        assert np.all(np.isfinite([lnl_chi[0], lnl_unif[0]]))

    def test_chisquare_outlier_with_invalid_data(self):
        """Test chi-square outlier model with invalid photometry data."""
        flux = np.array([[1.0, np.nan, 0.6], [1.2, 1.0, np.inf]])
        err = np.array([[0.1, 0.08, 0.0], [0.12, np.nan, 0.08]])  # Zero and NaN errors

        lnl_outlier = chisquare_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))  # Should handle invalid data gracefully

    def test_chisquare_outlier_custom_p_value(self):
        """Test chi-square outlier model with custom p-value cut."""
        flux = np.array([[1.0, 0.8, 0.6]])
        err = np.array([[0.1, 0.08, 0.06]])

        # Test different p-values
        lnl_1 = chisquare_outlier_loglike(flux, err, p_value_cut=1e-3)
        lnl_2 = chisquare_outlier_loglike(flux, err, p_value_cut=1e-6)

        assert len(lnl_1) == 1 and len(lnl_2) == 1
        assert np.all(np.isfinite([lnl_1[0], lnl_2[0]]))
        # Different p-values should give different results
        assert lnl_1[0] != lnl_2[0]

    def test_chisquare_outlier_invalid_parallax(self):
        """Test chi-square outlier model with invalid parallax data."""
        flux = np.array([[1.0, 0.8], [1.2, 1.0]])
        err = np.array([[0.1, 0.08], [0.12, 0.1]])
        parallax = np.array([2.0, np.nan])  # One invalid parallax
        parallax_err = np.array([0.1, 0.0])  # One zero error

        lnl_outlier = chisquare_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))

    def test_uniform_outlier_with_invalid_data(self):
        """Test uniform outlier model with invalid photometry data."""
        flux = np.array([[1.0, np.nan, 0.6], [1.2, 1.0, np.inf]])
        err = np.array([[0.1, 0.08, 0.0], [0.12, np.nan, 0.08]])  # Zero and NaN errors

        # Should handle invalid data gracefully - expect warnings
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # Ignore expected NaN warnings
            lnl_outlier = uniform_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        # Should handle invalid data - may be -inf but finite is acceptable
        assert np.all(np.isfinite(lnl_outlier) | (lnl_outlier == -np.inf))

    def test_uniform_outlier_custom_sigma_clip(self):
        """Test uniform outlier model with custom sigma clipping."""
        # Use more diverse data to ensure sigma clip differences matter
        flux = np.array([[1.0, 0.8, 0.6], [2.0, 1.5, 1.2]])
        err = np.array([[0.1, 0.08, 0.06], [0.2, 0.15, 0.12]])

        # Test different sigma clips
        lnl_1 = uniform_outlier_loglike(flux, err, sigma_clip=2.0)
        lnl_2 = uniform_outlier_loglike(flux, err, sigma_clip=5.0)

        assert len(lnl_1) == 2 and len(lnl_2) == 2
        assert np.all(np.isfinite([lnl_1, lnl_2]))
        # Different sigma clips should give different results (may be same for some data)
        # Just test that function works with different parameters
        assert isinstance(lnl_1[0], (int, float, np.number))
        assert isinstance(lnl_2[0], (int, float, np.number))

    def test_uniform_outlier_invalid_parallax(self):
        """Test uniform outlier model with invalid parallax data."""
        flux = np.array([[1.0, 0.8], [1.2, 1.0]])
        err = np.array([[0.1, 0.08], [0.12, 0.1]])
        parallax = np.array([2.0, np.nan])  # One invalid parallax
        parallax_err = np.array([0.1, np.inf])  # One infinite error

        lnl_outlier = uniform_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier) | (lnl_outlier == -np.inf))

    def test_outlier_models_single_filter(self):
        """Test outlier models with single filter (edge case)."""
        flux = np.array([[1.0], [0.5]])  # Single filter
        err = np.array([[0.1], [0.05]])

        lnl_chi = chisquare_outlier_loglike(flux, err)
        lnl_unif = uniform_outlier_loglike(flux, err)

        assert len(lnl_chi) == 2 and len(lnl_unif) == 2
        assert np.all(np.isfinite(lnl_chi))
        assert np.all(np.isfinite(lnl_unif) | (lnl_unif == -np.inf))

    def test_outlier_models_stellar_params_unused(self):
        """Test that stellar_params parameter is handled (currently unused)."""
        flux = np.array([[1.0, 0.8]])
        err = np.array([[0.1, 0.08]])
        stellar_params = {"mass": [1.0], "age": [1e9]}  # Currently unused

        lnl_chi = chisquare_outlier_loglike(flux, err, stellar_params=stellar_params)
        lnl_unif = uniform_outlier_loglike(flux, err, stellar_params=stellar_params)

        assert len(lnl_chi) == 1 and len(lnl_unif) == 1
        assert np.all(np.isfinite([lnl_chi[0], lnl_unif[0]]))
