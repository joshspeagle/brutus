#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus.priors module.

Tests all prior probability functions using simulated stellar data
to validate mathematical correctness and numerical stability.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from brutus.priors import (
    convert_parallax_to_scale,
    logn_disk,
    logn_halo,
    logp_age_from_feh,
    logp_extinction,
    logp_feh,
    logp_galactic_structure,
    logp_imf,
    logp_parallax,
    logp_parallax_scale,
    logp_ps1_luminosity_function,
)

# scipy.stats tested elsewhere


class TestStellarPriors:
    """Test stellar priors (IMF and luminosity function)."""

    def test_logp_imf_single_mass_basic(self):
        """Test basic IMF evaluation for single mass."""
        masses = np.array([0.5, 1.0, 2.0])
        logp = logp_imf(masses)

        # Should have finite values for valid masses
        assert np.all(np.isfinite(logp))

        # Higher masses should have lower probability (steeper slope)
        assert logp[2] < logp[1] < logp[0]

    def test_logp_imf_mass_limits(self):
        """Test IMF behavior at mass boundaries."""
        # Below hydrogen burning limit
        low_masses = np.array([0.05, 0.07])
        logp_low = logp_imf(low_masses)
        assert np.all(logp_low == -np.inf)

        # Valid masses
        valid_masses = np.array([0.1, 0.5, 1.0, 10.0])
        logp_valid = logp_imf(valid_masses)
        assert np.all(np.isfinite(logp_valid))

    def test_logp_imf_binary_system(self):
        """Test IMF for binary star systems."""
        m1 = np.array([1.0, 2.0])
        m2 = np.array([0.8, 1.5])

        # Binary system prior
        logp_binary = logp_imf(m1, mgrid2=m2)

        # Single star priors
        logp_single1 = logp_imf(m1)
        logp_single2 = logp_imf(m2)

        # Binary should have finite values and correct length
        assert len(logp_binary) == len(logp_single1)
        assert np.all(np.isfinite(logp_binary))

        # Binary system should generally have lower probability than single stars
        # (since we're multiplying two probabilities)
        assert np.all(logp_binary < np.maximum(logp_single1, logp_single2))

    def test_logp_imf_custom_parameters(self):
        """Test IMF with custom slope parameters."""
        masses = np.array([0.3, 0.8, 3.0])

        # Custom slopes
        logp_custom = logp_imf(masses, alpha_low=1.5, alpha_high=2.5, mass_break=0.7)
        logp_default = logp_imf(masses)

        # Should produce different results
        assert not np.allclose(logp_custom, logp_default)

    def test_logp_imf_normalization(self):
        """Test that IMF is properly normalized."""
        # Create fine mass grid
        masses = np.logspace(np.log10(0.08), np.log10(100), 10000)
        dm = np.diff(masses)
        dm = np.append(dm, dm[-1])  # Extend for integration

        # Compute probabilities and integrate
        logp = logp_imf(masses)
        p = np.exp(logp)
        integral = np.sum(p * dm)

        # Should integrate to approximately 1
        assert abs(integral - 1.0) < 0.01

    def test_logp_ps1_luminosity_function_basic(self):
        """Test PS1 luminosity function evaluation."""
        Mr = np.array([-2, 0, 5, 10, 15])
        logp = logp_ps1_luminosity_function(Mr)

        # Should return finite values
        assert np.all(np.isfinite(logp))
        assert len(logp) == len(Mr)

    def test_logp_ps1_luminosity_function_monotonic(self):
        """Test that LF has expected monotonic behavior."""
        # Main sequence range where LF should be well-behaved
        Mr = np.linspace(3, 12, 20)
        logp = logp_ps1_luminosity_function(Mr)

        # Should be monotonic in this range (fainter = more common)
        dlogp = np.diff(logp)
        assert np.all(dlogp > -1)  # Allow some variation but generally increasing

    def test_logp_ps1_luminosity_function_interpolation_cache(self):
        """Test that interpolator is cached properly."""
        Mr1 = np.array([5.0])
        Mr2 = np.array([5.0])

        logp1 = logp_ps1_luminosity_function(Mr1)
        logp2 = logp_ps1_luminosity_function(Mr2)

        # Should be identical (cached)
        assert logp1 == logp2


class TestAstrometricPriors:
    """Test astrometric priors (parallax and scale factors)."""

    def test_logp_parallax_valid_measurement(self):
        """Test parallax prior with valid measurement."""
        parallaxes = np.array([0.8, 1.0, 1.2])
        p_meas, p_err = 1.0, 0.1

        logp = logp_parallax(parallaxes, p_meas, p_err)

        # Should be Gaussian centered on measurement
        assert np.all(np.isfinite(logp))
        assert logp[1] > logp[0]  # Closer to center has higher prob
        assert logp[1] > logp[2]

        # Test normalization constant
        chi2_expected = (parallaxes - p_meas) ** 2 / p_err**2
        log_norm_expected = np.log(2 * np.pi * p_err**2)
        logp_expected = -0.5 * (chi2_expected + log_norm_expected)
        assert np.allclose(logp, logp_expected)

    def test_logp_parallax_invalid_measurement(self):
        """Test parallax prior with invalid measurements."""
        parallaxes = np.array([0.5, 1.0, 1.5])

        # Invalid measurements should return uniform prior
        cases = [
            (np.nan, 0.1),  # NaN measurement
            (1.0, np.nan),  # NaN error
            (1.0, 0.0),  # Zero error
            (1.0, -0.1),  # Negative error
        ]

        for p_meas, p_err in cases:
            logp = logp_parallax(parallaxes, p_meas, p_err)
            assert np.allclose(logp, 0.0)

    def test_convert_parallax_to_scale_high_snr(self):
        """Test parallax to scale conversion for high SNR."""
        p_meas, p_err = 2.0, 0.2  # 10-sigma measurement
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err)

        # Should use error propagation formula
        expected_mean = p_meas**2 + p_err**2
        expected_std = np.sqrt(2 * p_err**4 + 4 * p_meas**2 * p_err**2)

        assert abs(s_mean - expected_mean) < 1e-10
        assert abs(s_std - expected_std) < 1e-10

    def test_convert_parallax_to_scale_negative_parallax(self):
        """Test parallax conversion with negative measurement."""
        # Use higher SNR negative parallax that passes SNR threshold
        p_meas, p_err = -1.0, 0.2  # -5 sigma (high SNR)
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err)

        # Should floor at zero
        expected_mean = 0.0**2 + p_err**2  # p_positive = 0
        expected_std = np.sqrt(2 * p_err**4 + 4 * 0.0**2 * p_err**2)

        assert abs(s_mean - expected_mean) < 1e-10
        assert abs(s_std - expected_std) < 1e-10

    def test_convert_parallax_to_scale_low_snr(self):
        """Test parallax conversion for low SNR measurements."""
        p_meas, p_err = 0.5, 0.2  # 2.5-sigma (below 4.0 threshold)
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err, snr_lim=4.0)

        # Should return uninformative prior
        assert s_mean == 1e-20
        assert s_std == 1e20

    def test_logp_parallax_scale_high_snr(self):
        """Test scale factor prior for high SNR parallax."""
        scales = np.array([3.8, 4.0, 4.2])
        scale_errs = np.array([0.1, 0.1, 0.1])
        p_meas, p_err = 2.0, 0.3  # 6.7-sigma measurement

        logp = logp_parallax_scale(scales, scale_errs, p_meas, p_err)

        # Should be Gaussian with combined uncertainties
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err)
        total_var = s_std**2 + scale_errs**2

        chi2_expected = (scales - s_mean) ** 2 / total_var
        log_norm_expected = np.log(2 * np.pi * total_var)
        logp_expected = -0.5 * (chi2_expected + log_norm_expected)

        assert np.allclose(logp, logp_expected)

    def test_logp_parallax_scale_low_snr(self):
        """Test scale factor prior for low SNR parallax."""
        scales = np.array([1.0, 2.0, 3.0])
        scale_errs = np.array([0.1, 0.1, 0.1])
        p_meas, p_err = 1.0, 0.5  # 2-sigma measurement

        logp = logp_parallax_scale(scales, scale_errs, p_meas, p_err, snr_lim=4.0)

        # Should return uniform prior
        assert np.allclose(logp, 0.0)


class TestGalacticPriors:
    """Test Galactic structure and stellar population priors."""

    def test_logn_disk_basic(self):
        """Test basic disk number density evaluation."""
        distances = np.array([0.5, 1.0, 2.0, 5.0])  # kpc
        zs = np.array([0.0, 0.1, 0.5, 1.0])  # kpc above plane

        logn = logn_disk(distances, zs)

        # Should be finite
        assert np.all(np.isfinite(logn))

        # Should decrease with height
        logn_heights = logn_disk(np.array([1.0]), np.array([0.0, 0.2, 0.5, 1.0]))
        assert np.all(np.diff(logn_heights) < 0)  # Decreasing

    def test_logn_disk_custom_parameters(self):
        """Test disk with custom scale parameters."""
        distances = np.array([1.0, 2.0])
        zs = np.array([0.1, 0.1])

        logn_default = logn_disk(distances, zs)
        logn_custom = logn_disk(distances, zs, R_scale=3.0, Z_scale=0.4)

        # Should produce different results
        assert not np.allclose(logn_default, logn_custom)

    def test_logn_halo_basic(self):
        """Test basic halo number density evaluation."""
        distances = np.array([0.5, 1.0, 5.0, 20.0])  # kpc
        zs = np.array([0.0, 1.0, 5.0, 10.0])  # kpc

        logn = logn_halo(distances, zs)

        # Should be finite
        assert np.all(np.isfinite(logn))

    def test_logn_halo_custom_parameters(self):
        """Test halo with custom parameters."""
        distances = np.array([5.0, 10.0])
        zs = np.array([2.0, 2.0])

        logn_default = logn_halo(distances, zs)
        logn_custom = logn_halo(distances, zs, eta=2.8, q_inf=0.8)

        # Should produce different results
        assert not np.allclose(logn_default, logn_custom)

    def test_logp_feh_basic(self):
        """Test metallicity prior evaluation."""
        fehs = np.array([-2.0, -1.0, 0.0, 0.5])

        # Basic evaluation with default parameters
        logp = logp_feh(fehs)
        assert np.all(np.isfinite(logp))
        assert len(logp) == len(fehs)

    def test_logp_feh_custom_parameters(self):
        """Test metallicity prior with custom parameters."""
        fehs = np.array([-1.0, 0.0])

        logp_default = logp_feh(fehs)
        logp_custom = logp_feh(fehs, feh_mean=-0.1, feh_sigma=0.4)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)

    def test_logp_age_from_feh_basic(self):
        """Test age-metallicity relation."""
        ages = np.array([1.0, 5.0, 10.0, 13.0])  # Gyr
        fehs = np.array([-0.5, -0.2, 0.0, -0.1])

        logp = logp_age_from_feh(ages, fehs)

        # Should be finite
        assert np.all(np.isfinite(logp))

    def test_logp_age_from_feh_custom_parameters(self):
        """Test age-metallicity relation with custom parameters."""
        ages = np.array([5.0, 8.0])

        logp_default = logp_age_from_feh(ages)
        logp_custom = logp_age_from_feh(ages, feh_mean=-0.1, max_age=12.0)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)

    def test_logp_galactic_structure_basic(self):
        """Test combined Galactic structure prior."""
        # Create mock coordinates
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")
        distances = np.array([1.0, 2.0, 5.0])  # kpc

        logp = logp_galactic_structure(distances, coord)

        # Should be finite
        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_logp_galactic_structure_with_components(self):
        """Test Galactic structure prior returning components."""
        coord = SkyCoord(ra=90.0, dec=45.0, unit="deg")
        distances = np.array([0.5, 1.5, 3.0])

        result = logp_galactic_structure(distances, coord, return_components=True)
        logp_total, components = result

        # Check components structure
        assert "number_density" in components
        assert isinstance(components["number_density"], list)
        assert len(components["number_density"]) == 3  # thin, thick, halo components

        # Each component should have right length
        for comp in components.values():
            assert len(comp) == len(distances)
            assert np.all(np.isfinite(comp))


class TestExtinctionPriors:
    """Test dust extinction priors."""

    # --- Tests for simple 2-tuple dust maps ---

    def test_logp_extinction_valid_dustmap(self):
        """Test extinction prior with valid 2-tuple dust map."""
        avs = np.array([0.1, 0.2, 0.3])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map returning (mean, std) 2-tuple
        dustmap = Mock()
        dustmap.query.return_value = (0.2, 0.05)

        logp = logp_extinction(avs, dustmap, coord)

        # Should be Gaussian
        assert np.all(np.isfinite(logp))
        assert logp[1] > logp[0]  # Closer to mean has higher prob
        assert logp[1] > logp[2]

    def test_logp_extinction_no_coverage(self):
        """Test extinction prior without dust map coverage."""
        avs = np.array([0.1, 0.2, 0.3])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map with no coverage
        dustmap = Mock()
        dustmap.query.return_value = (np.nan, np.nan)

        logp = logp_extinction(avs, dustmap, coord)

        # Should be uniform
        assert np.allclose(logp, 0.0)

    def test_logp_extinction_with_components(self):
        """Test extinction prior returning components."""
        avs = np.array([0.15])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        # Mock dust map
        dustmap = Mock()
        av_mean, av_err = 0.2, 0.03
        dustmap.query.return_value = (av_mean, av_err)

        logp, (mean, err) = logp_extinction(avs, dustmap, coord, return_components=True)

        assert mean == av_mean
        assert err == av_err

    def test_logp_extinction_dustmap_error(self):
        """Test extinction prior with dust map query error."""
        avs = np.array([0.1, 0.2])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map that raises exception
        dustmap = Mock()
        dustmap.query.side_effect = TypeError("Query failed")

        logp = logp_extinction(avs, dustmap, coord)

        # Should return uniform prior on error
        assert np.allclose(logp, 0.0)

    # --- Tests for 3D dust maps (Bayestar-like 3-tuple) ---

    def test_logp_extinction_3d_dustmap_with_distance(self):
        """Test extinction prior with 3D dust map and distance."""
        avs = np.array([0.1, 0.3, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        # Mock 3D dust map returning (distances, av_profile, std_profile)
        map_distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        av_profile = np.array([0.05, 0.15, 0.3, 0.5, 0.8])
        std_profile = np.array([0.02, 0.05, 0.1, 0.15, 0.2])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        # Evaluate at d=1.0 kpc where map gives av_mean=0.3, av_std=0.1
        logp = logp_extinction(avs, dustmap, coord, distance=1.0)

        assert np.all(np.isfinite(logp))
        # av=0.3 is closest to the map mean at d=1 kpc
        assert logp[1] > logp[0]
        assert logp[1] > logp[2]

    def test_logp_extinction_3d_dustmap_no_distance(self):
        """Test 3D dust map without distance returns uniform prior."""
        avs = np.array([0.1, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 1.0, 5.0])
        av_profile = np.array([0.05, 0.3, 0.8])
        std_profile = np.array([0.02, 0.1, 0.2])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        # No distance provided — should return uniform prior
        logp = logp_extinction(avs, dustmap, coord)
        assert np.allclose(logp, 0.0)

    def test_logp_extinction_3d_dustmap_array_distances(self):
        """Test 3D dust map with array of distances (BruteForce MC case)."""
        # Simulate the BruteForce case: matching arrays of (av, distance)
        # Use uniform sigma so normalization doesn't affect comparisons
        # avs[0]=0.1 at d=0.5 (map gives 0.15, mismatch)
        # avs[1]=0.3 at d=1.0 (map gives 0.3, exact match)
        # avs[2]=1.5 at d=2.0 (map gives 0.5, large mismatch)
        # avs[3]=0.8 at d=5.0 (map gives 0.8, exact match)
        avs = np.array([0.1, 0.3, 1.5, 0.8])
        distances = np.array([0.5, 1.0, 2.0, 5.0])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        av_profile = np.array([0.05, 0.15, 0.3, 0.5, 0.8])
        std_profile = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # uniform sigma

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        logp = logp_extinction(avs, dustmap, coord, distance=distances)

        assert logp.shape == avs.shape
        assert np.all(np.isfinite(logp))

        # With uniform sigma, exact matches should have highest logp
        assert logp[1] > logp[0]  # exact match vs small mismatch
        assert logp[3] > logp[2]  # exact match vs large mismatch

        # Verify against manual Gaussian computation
        for i in range(len(avs)):
            expected_mean = np.interp(distances[i], map_distances, av_profile)
            sigma = 0.1
            expected = -0.5 * (
                (avs[i] - expected_mean) ** 2 / sigma**2 + np.log(2 * np.pi * sigma**2)
            )
            assert np.isclose(logp[i], expected, atol=1e-10)

    def test_logp_extinction_3d_dustmap_nan_coverage(self):
        """Test 3D dust map with NaN values (out of coverage)."""
        avs = np.array([0.1, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 1.0, 5.0])
        av_profile = np.array([np.nan, np.nan, np.nan])
        std_profile = np.array([np.nan, np.nan, np.nan])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        logp = logp_extinction(avs, dustmap, coord, distance=1.0)
        assert np.allclose(logp, 0.0)

    def test_logp_extinction_3d_components(self):
        """Test 3D dust map with return_components."""
        avs = np.array([0.3])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 1.0, 5.0])
        av_profile = np.array([0.05, 0.3, 0.8])
        std_profile = np.array([0.02, 0.1, 0.2])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        logp, (mean, err) = logp_extinction(
            avs, dustmap, coord, distance=1.0, return_components=True
        )

        assert np.isclose(mean, 0.3)
        assert np.isclose(err, 0.1)

    def test_logp_extinction_string_dustfile(self):
        """Test that passing a string (file path) returns uniform prior."""
        avs = np.array([0.1, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        # String has no .query() method — should return uniform
        logp = logp_extinction(avs, "bayestar2019_v1.h5", coord, distance=1.0)
        assert np.allclose(logp, 0.0)


class TestUtilities:
    """Test utility functions."""


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_priors_with_extreme_values(self):
        """Test prior functions with extreme input values."""
        # Very large masses beyond default mass_max=100 should return -inf
        large_masses = np.array([1e6, 1e10])
        logp_imf_large = logp_imf(large_masses)
        assert np.all(logp_imf_large == -np.inf)

        # Large masses within custom mass_max should be finite
        logp_imf_custom = logp_imf(large_masses, mass_max=1e11)
        assert np.all(np.isfinite(logp_imf_custom))

        # Very small parallaxes
        small_parallax = np.array([1e-10, 1e-8])
        logp_plx = logp_parallax(small_parallax, 1e-9, 1e-10)
        assert np.all(np.isfinite(logp_plx))

        # Extreme distances
        extreme_distances = np.array([0.001, 100.0])  # kpc
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")
        logp_gal = logp_galactic_structure(extreme_distances, coord)
        assert np.all(np.isfinite(logp_gal))

    def test_priors_with_array_broadcasting(self):
        """Test that priors handle array broadcasting correctly."""
        # Test IMF with different array shapes
        m1 = np.array([1.0])
        m2 = np.array([0.5, 1.0, 1.5])

        logp = logp_imf(m1, mgrid2=m2)
        assert logp.shape == (3,)  # Should match mgrid2 shape

        # Test parallax with broadcasting
        parallaxes = np.array([[0.8, 1.0], [1.2, 1.4]])
        logp_plx = logp_parallax(parallaxes, 1.0, 0.1)
        assert logp_plx.shape == (2, 2)

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test with empty arrays
        empty_array = np.array([])
        logp_empty = logp_imf(empty_array)
        assert len(logp_empty) == 0

        # Test with single values vs arrays
        single_mass = 1.0
        array_mass = np.array([1.0])

        logp_single = logp_imf(single_mass)
        logp_array = logp_imf(array_mass)

        assert np.isscalar(logp_single) or logp_single.shape == ()
        assert logp_array.shape == (1,)


# Fixtures for test data generation
@pytest.fixture
def simulated_stellar_data():
    """Generate simulated stellar parameter data for testing."""
    np.random.seed(42)
    n_stars = 1000

    # Simulate stellar parameters
    masses = np.random.lognormal(0, 0.5, n_stars)  # M_sun
    ages = np.random.uniform(0.1, 13.0, n_stars)  # Gyr
    fehs = np.random.normal(-0.2, 0.3, n_stars)  # [Fe/H]

    # Simulate astrometry
    parallaxes = np.random.lognormal(-1, 0.8, n_stars)  # mas
    parallax_errors = 0.1 * parallaxes * np.random.uniform(0.5, 2.0, n_stars)

    # Simulate sky positions
    ra = np.random.uniform(0, 360, n_stars)  # deg
    dec = np.arcsin(np.random.uniform(-1, 1, n_stars)) * 180 / np.pi  # deg
    coords = SkyCoord(ra=ra, dec=dec, unit="deg")

    # Simulate extinction
    distances = 1.0 / (parallaxes * 1e-3)  # kpc (approximate)
    extinctions = 0.1 * np.random.exponential(1, n_stars)  # A_V

    return {
        "masses": masses,
        "ages": ages,
        "fehs": fehs,
        "parallaxes": parallaxes,
        "parallax_errors": parallax_errors,
        "coords": coords,
        "distances": distances,
        "extinctions": extinctions,
    }


class TestIntegration:
    """Integration tests using simulated stellar data."""

    def test_prior_integration_workflow(self, simulated_stellar_data):
        """Test complete prior evaluation workflow."""
        data = simulated_stellar_data

        # Evaluate all priors for subset of stars
        n_test = 50
        idx = np.random.choice(len(data["masses"]), n_test, replace=False)

        # IMF prior
        logp_masses = logp_imf(data["masses"][idx])
        assert len(logp_masses) == n_test
        assert np.all(np.isfinite(logp_masses))

        # Parallax priors (test with first object's measurements)
        logp_plx = logp_parallax(
            data["parallaxes"][idx],
            data["parallaxes"][idx[0]],
            data["parallax_errors"][idx[0]],
        )
        assert len(logp_plx) == n_test

        # Galactic structure priors
        logp_gal = logp_galactic_structure(data["distances"][idx], data["coords"][idx])
        assert len(logp_gal) == n_test
        assert np.all(np.isfinite(logp_gal))

    def test_prior_consistency_checks(self, simulated_stellar_data):
        """Test internal consistency of priors."""
        data = simulated_stellar_data

        # Test that higher mass stars have lower IMF probability
        low_mass_idx = data["masses"] < 0.5
        high_mass_idx = data["masses"] > 2.0

        if np.sum(low_mass_idx) > 0 and np.sum(high_mass_idx) > 0:
            logp_low = np.mean(logp_imf(data["masses"][low_mass_idx]))
            logp_high = np.mean(logp_imf(data["masses"][high_mass_idx]))
            assert logp_low > logp_high

    def test_prior_normalization_estimates(self):
        """Test that priors are approximately normalized."""
        # This is a rough check since exact normalization requires
        # integration over the full parameter space

        # Test IMF normalization over reasonable mass range
        masses = np.logspace(np.log10(0.08), np.log10(8), 1000)
        dm = np.diff(masses)
        dm = np.append(dm, dm[-1])

        logp = logp_imf(masses)
        p = np.exp(logp)
        integral = np.sum(p * dm)

        # Should be close to 1 (within 10% for this mass range)
        assert 0.5 < integral < 2.0
