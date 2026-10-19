#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus photometry utilities.

This test suite includes:
1. Unit tests for the new photometry functions
2. Comparison tests between old and new implementations
3. Integration tests for realistic workflows
"""

import numpy as np
import pytest


class TestPhotometryFunctions:
    """Unit tests for individual photometry functions."""

    def test_magnitude_basic(self):
        """Test basic magnitude conversion."""
        # Test case: flux=1 should give magnitude=0 (by definition)
        flux = np.array([[1.0, 10.0, 0.1]])
        flux_err = np.array([[0.1, 1.0, 0.01]])

        # Import from new location
        from brutus.utils.photometry import magnitude

        mag, mag_err = magnitude(flux, flux_err)

        # Check magnitude values
        expected_mag = np.array([[0.0, -2.5, 2.5]])
        np.testing.assert_array_almost_equal(mag, expected_mag, decimal=10)

        # Check that errors are positive and finite
        assert np.all(mag_err > 0)
        assert np.all(np.isfinite(mag_err))

    def test_inv_magnitude_basic(self):
        """Test basic inverse magnitude conversion."""
        from brutus.utils.photometry import inv_magnitude

        mag = np.array([[0.0, -2.5, 2.5]])
        mag_err = np.array([[0.1, 0.2, 0.3]])

        flux, flux_err = inv_magnitude(mag, mag_err)

        # Check flux values
        expected_flux = np.array([[1.0, 10.0, 0.1]])
        np.testing.assert_array_almost_equal(flux, expected_flux, decimal=10)

        # Check that errors are positive and finite
        assert np.all(flux_err > 0)
        assert np.all(np.isfinite(flux_err))

    def test_magnitude_inverse_roundtrip(self):
        """Test that magnitude <-> flux conversion is invertible."""
        from brutus.utils.photometry import inv_magnitude, magnitude

        # Start with some flux values
        original_flux = np.array([[1.0, 5.0, 0.1, 100.0]])
        original_flux_err = np.array([[0.1, 0.5, 0.01, 10.0]])

        # Convert to magnitude
        mag, mag_err = magnitude(original_flux, original_flux_err)

        # Convert back to flux
        recovered_flux, recovered_flux_err = inv_magnitude(mag, mag_err)

        # Should recover original values
        np.testing.assert_array_almost_equal(recovered_flux, original_flux, decimal=10)
        np.testing.assert_array_almost_equal(
            recovered_flux_err, original_flux_err, decimal=10
        )

    def test_luptitude_basic(self):
        """Test basic luptitude conversion."""
        from brutus.utils.photometry import luptitude

        flux = np.array([[1.0, 10.0, 0.1]])
        flux_err = np.array([[0.1, 1.0, 0.01]])
        skynoise = np.array([1.0, 1.0, 1.0])

        lupt, lupt_err = luptitude(flux, flux_err, skynoise=skynoise)

        # Luptitudes should be finite
        assert np.all(np.isfinite(lupt))
        assert np.all(np.isfinite(lupt_err))

        # Errors should be positive
        assert np.all(lupt_err > 0)

    def test_luptitude_inverse_roundtrip(self):
        """Test that luptitude <-> flux conversion is invertible."""
        from brutus.utils.photometry import inv_luptitude, luptitude

        # Start with some flux values
        original_flux = np.array([[1.0, 5.0, 0.1, 100.0]])
        original_flux_err = np.array([[0.1, 0.5, 0.01, 10.0]])
        skynoise = 1.0

        # Convert to luptitude
        lupt, lupt_err = luptitude(original_flux, original_flux_err, skynoise=skynoise)

        # Convert back to flux
        recovered_flux, recovered_flux_err = inv_luptitude(
            lupt, lupt_err, skynoise=skynoise
        )

        # Should recover original values
        np.testing.assert_array_almost_equal(recovered_flux, original_flux, decimal=8)
        np.testing.assert_array_almost_equal(
            recovered_flux_err, original_flux_err, decimal=8
        )

    def test_add_mag(self):
        """Test magnitude addition."""
        from brutus.utils.photometry import add_mag

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

    def test_phot_loglike_basic(self):
        """Test photometric log-likelihood computation."""
        from brutus.utils.photometry import phot_loglike

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
        from brutus.utils.photometry import phot_loglike

        flux = np.array([[1.0, 2.0, 3.0]])
        err = np.array([[0.1, 0.2, 0.3]])
        mfluxes = np.array([[[1.0, 2.0, 3.0]]])
        mask = np.array([[1, 1, 0]])  # mask out third filter

        lnl = phot_loglike(flux, err, mfluxes, mask=mask)

        # Should still return valid log-likelihood
        assert lnl.shape == (1, 1)
        assert np.isfinite(lnl[0, 0])

    def test_zeropoints_effect(self):
        """Test that zeropoints work correctly."""
        from brutus.utils.photometry import inv_magnitude, magnitude

        flux = np.array([[1.0, 1.0]])
        flux_err = np.array([[0.1, 0.1]])
        zeropoints = np.array([1.0, 10.0])  # Second filter has 10x higher zeropoint

        mag, mag_err = magnitude(flux, flux_err, zeropoints=zeropoints)

        # With zeropoint=10, effective flux is 1/10=0.1, so mag should be 2.5
        expected_mag = np.array([[0.0, 2.5]])
        np.testing.assert_array_almost_equal(mag, expected_mag, decimal=10)


class TestPhotometryEdgeCases:
    """Test zero, negative and zero-error fluxes, and bad shapes."""

    def test_zero_flux_magnitude(self):
        """Test handling of zero flux (should give infinite magnitude)."""
        from brutus.utils.photometry import magnitude

        flux = np.array([[0.0, 1.0]])
        flux_err = np.array([[0.1, 0.1]])

        # Suppress expected divide-by-zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Zero flux should give infinite magnitude
        assert np.isinf(mag[0, 0])
        # Non-zero flux should be finite
        assert np.isfinite(mag[0, 1])

    def test_negative_flux_magnitude(self):
        """Test handling of negative flux."""
        from brutus.utils.photometry import magnitude

        flux = np.array([[-1.0, 1.0]])
        flux_err = np.array([[0.1, 0.1]])

        # Suppress expected invalid value warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, mag_err = magnitude(flux, flux_err)

        # Negative flux should give NaN magnitude (log of negative number)
        assert np.isnan(mag[0, 0])
        # Positive flux should be finite
        assert np.isfinite(mag[0, 1])

    def test_zero_error_magnitude(self):
        """Test handling of zero error."""
        from brutus.utils.photometry import magnitude

        flux = np.array([[1.0]])
        flux_err = np.array([[0.0]])

        mag, mag_err = magnitude(flux, flux_err)

        # Magnitude should be finite, error should be 0
        assert np.isfinite(mag[0])
        assert mag_err[0] == 0.0

    def test_dimension_mismatch_phot_loglike(self):
        """Test error handling for dimension mismatches."""
        from brutus.utils.photometry import phot_loglike

        flux = np.array([[1.0, 2.0]])  # (1, 2)
        err = np.array([[0.1, 0.2]])  # (1, 2)
        mfluxes = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3) - WRONG!

        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            phot_loglike(flux, err, mfluxes)


class TestPhotometryIntegration:
    """Integration tests for photometry functions."""

    def test_realistic_photometry_pipeline(self):
        """Test a realistic photometry processing pipeline."""
        from brutus.utils.photometry import add_mag, inv_magnitude, magnitude

        # Simulate realistic survey data
        np.random.seed(42)

        # True magnitudes for a range of stars
        true_mags = np.random.uniform(14, 24, 100)
        true_fluxes = 10 ** (-0.4 * true_mags)

        # Add realistic errors (magnitude-dependent)
        snr = 100 * 10 ** (-0.4 * (true_mags - 20))  # S/N decreases with magnitude
        flux_errors = true_fluxes / snr

        # Add noise
        observed_fluxes = true_fluxes + np.random.normal(0, flux_errors)

        # Convert to magnitudes
        obs_mags, obs_mag_errs = magnitude(
            observed_fluxes.reshape(-1, 1), flux_errors.reshape(-1, 1)
        )

        # Should recover input magnitudes within errors for bright stars
        bright_mask = true_mags < 20
        mag_diff = obs_mags.flatten()[bright_mask] - true_mags[bright_mask]
        expected_scatter = obs_mag_errs.flatten()[bright_mask]

        # Most should be within 3-sigma
        outlier_fraction = np.mean(np.abs(mag_diff) > 3 * expected_scatter)
        assert outlier_fraction < 0.05  # Less than 5% outliers

    def test_multiband_photometry_consistency(self):
        """Test consistency across multiple photometric bands."""
        from brutus.utils.photometry import magnitude, phot_loglike

        # Simulate 5-band photometry for 50 stars
        np.random.seed(42)
        nstars, nbands = 50, 5

        # Create correlated fluxes (bluer bands have higher flux for hot stars)
        base_flux = np.random.uniform(0.1, 10.0, nstars)
        color_term = np.random.normal(0, 0.5, nstars)

        fluxes = np.zeros((nstars, nbands))
        for i in range(nbands):
            # Simulate color gradient across bands
            band_modifier = 1.0 + color_term * (i - 2) * 0.2
            fluxes[:, i] = base_flux * band_modifier

        # Add realistic errors
        flux_errors = 0.05 * fluxes + 0.01  # 5% + constant floor

        # Convert to magnitudes
        mags, mag_errs = magnitude(fluxes, flux_errors)

        # Check that colors are reasonable (adjacent bands within ~2 mag)
        for i in range(nbands - 1):
            colors = mags[:, i] - mags[:, i + 1]
            assert np.all(np.abs(colors) < 3.0)  # Colors within reasonable range

        # Test likelihood computation with perfect model
        model_fluxes = fluxes[:, None, :]  # Add model dimension (Nobj, Nmod=1, Nfilt)
        lnl = phot_loglike(fluxes, flux_errors, model_fluxes)

        # Likelihood should be reasonable (not too negative)
        assert np.all(lnl > -1000)  # Somewhat arbitrary but reasonable threshold

    def test_phot_loglike_dim_prior(self):
        """Test photometric likelihood with dimensionality prior."""
        import numpy as np

        from brutus.utils.photometry import phot_loglike

        # Simple test case with dimensionality prior
        nobjs = 5
        nmods = 3
        nfilts = 4

        # Mock observed fluxes and errors
        fluxes = np.random.rand(nobjs, nfilts) * 1000 + 100  # Positive fluxes
        flux_errors = np.random.rand(nobjs, nfilts) * 50 + 10

        # Mock model fluxes
        model_fluxes = np.random.rand(nobjs, nmods, nfilts) * 1000 + 100

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

        # With dof_reduction=3, dof = 2 - 3 = -1 (should give -inf)
        lnl_small = phot_loglike(
            small_fluxes, small_errors, small_models, dim_prior=True, dof_reduction=3
        )
        assert lnl_small.shape == (nobjs, nmods)
        # Should be -inf when dof <= 0
        assert np.all(
            lnl_small == -np.inf
        ), "Should be -inf when degrees of freedom <= 0"


class TestOutlierModels:
    """Test outlier likelihood functions."""

    def test_chisquare_outlier_basic(self):
        """Test chi-square outlier model."""
        from brutus.utils.photometry import chisquare_outlier_loglike

        # Create test data
        flux = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        err = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        lnl = chisquare_outlier_loglike(flux, err)

        assert lnl.shape == (2,)
        assert np.all(np.isfinite(lnl))
        assert np.all(lnl < 0)  # Log-probabilities should be negative

    def test_chisquare_outlier_with_parallax(self):
        """Test chi-square outlier model with parallax."""
        from brutus.utils.photometry import chisquare_outlier_loglike

        flux = np.array([[1.0, 2.0], [3.0, 4.0]])
        err = np.array([[0.1, 0.2], [0.3, 0.4]])
        parallax = np.array([10.0, 20.0])
        parallax_err = np.array([1.0, 2.0])

        lnl = chisquare_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert lnl.shape == (2,)
        assert np.all(np.isfinite(lnl))

    def test_uniform_outlier_basic(self):
        """Test uniform outlier model."""
        from brutus.utils.photometry import uniform_outlier_loglike

        flux = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        err = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        lnl = uniform_outlier_loglike(flux, err)

        assert lnl.shape == (2,)
        assert np.all(np.isfinite(lnl))

    def test_uniform_outlier_with_parallax(self):
        """Test uniform outlier model with parallax."""
        from brutus.utils.photometry import uniform_outlier_loglike

        flux = np.array([[1.0, 2.0], [3.0, 4.0]])
        err = np.array([[0.1, 0.2], [0.3, 0.4]])
        parallax = np.array([10.0, 20.0])
        parallax_err = np.array([1.0, 2.0])

        lnl = uniform_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert lnl.shape == (2,)
        assert np.all(np.isfinite(lnl))


if __name__ == "__main__":
    pytest.main([__file__])
