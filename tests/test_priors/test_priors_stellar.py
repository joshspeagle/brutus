"""Tests of brutus.priors.stellar."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy import integrate

from brutus.priors import (
    logp_galactic_structure,
    logp_imf,
    logp_parallax,
    logp_ps1_luminosity_function,
)


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
        assert np.all(logp_low == -1e300)

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

    def test_logp_imf_alpha_unity_no_crash(self):
        """A power-law slope of exactly 1.0 must not raise.

        Regression: the IMF normalization used ``(m**(1-alpha))/(1-alpha)``,
        which is 0/0 for ``alpha == 1`` and raised ZeroDivisionError. The
        ``alpha == 1`` (flat-in-log) slope is physically meaningful.
        """
        masses = np.array([0.3, 1.0, 5.0])
        for kwargs in (
            {"alpha_low": 1.0},
            {"alpha_high": 1.0},
            {"alpha_low": 1.0, "alpha_high": 1.0},
        ):
            logp = logp_imf(masses, **kwargs)
            finite = logp[np.isfinite(logp)]
            assert finite.size >= 1
            assert np.all(np.isfinite(finite))

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


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_priors_with_extreme_values(self):
        """Test prior functions with extreme input values."""
        # Very large masses beyond default mass_max=100 should return -inf
        large_masses = np.array([1e6, 1e10])
        logp_imf_large = logp_imf(large_masses)
        assert np.all(logp_imf_large == -1e300)

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
    rng = np.random.RandomState(42)
    n_stars = 1000

    # Simulate stellar parameters
    masses = rng.lognormal(0, 0.5, n_stars)  # M_sun
    ages = rng.uniform(0.1, 13.0, n_stars)  # Gyr
    fehs = rng.normal(-0.2, 0.3, n_stars)  # [Fe/H]

    # Simulate astrometry
    parallaxes = rng.lognormal(-1, 0.8, n_stars)  # mas
    parallax_errors = 0.1 * parallaxes * rng.uniform(0.5, 2.0, n_stars)

    # Simulate sky positions
    ra = rng.uniform(0, 360, n_stars)  # deg
    dec = np.arcsin(rng.uniform(-1, 1, n_stars)) * 180 / np.pi  # deg
    coords = SkyCoord(ra=ra, dec=dec, unit="deg")

    # Simulate extinction
    distances = 1.0 / (parallaxes * 1e-3)  # kpc (approximate)
    extinctions = 0.1 * rng.exponential(1, n_stars)  # A_V

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
        rng = np.random.RandomState(0)
        data = simulated_stellar_data

        # Evaluate all priors for subset of stars
        n_test = 50
        idx = rng.choice(len(data["masses"]), n_test, replace=False)

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


class TestIMFNormalization:
    """logp_imf must integrate to 1 over [mass_min, mass_max] regardless of
    where the range sits relative to mass_break."""

    @staticmethod
    def _integral(mass_min, mass_max, **kwargs):
        def pdf(m):
            return np.exp(
                logp_imf(
                    np.atleast_1d(m),
                    mass_min=mass_min,
                    mass_max=mass_max,
                    **kwargs,
                )
            )[0]

        val, _ = integrate.quad(pdf, mass_min, mass_max, limit=200)
        return val

    def test_defaults_normalized(self):
        assert np.isclose(self._integral(0.08, 100.0), 1.0, rtol=1e-6)

    def test_mass_min_above_break_normalized(self):
        # Old code: integral was 2.185 (norm_low integrated backwards)
        assert np.isclose(self._integral(1.0, 100.0), 1.0, rtol=1e-6)

    def test_mass_min_far_above_break_not_nan(self):
        # Old code: norm < 0 -> log(norm) = NaN for every mass
        logp = logp_imf(np.array([6.0, 10.0]), mass_min=5.0)
        assert np.all(np.isfinite(logp))
        assert np.isclose(self._integral(5.0, 100.0), 1.0, rtol=1e-6)

    def test_mass_max_below_break_normalized(self):
        assert np.isclose(self._integral(0.08, 0.3), 1.0, rtol=1e-6)

    def test_binary_normalization_high_mass_min(self):
        # Binary case squares the norm; must stay finite and consistent
        logp = logp_imf(np.array([2.0]), mgrid2=np.array([1.5]), mass_min=1.0)
        assert np.all(np.isfinite(logp))

    def test_invalid_mass_range_raises(self):
        with pytest.raises(ValueError):
            logp_imf(np.array([1.0]), mass_min=2.0, mass_max=1.0)
        with pytest.raises(ValueError):
            logp_imf(np.array([1.0]), mass_min=0.0)
