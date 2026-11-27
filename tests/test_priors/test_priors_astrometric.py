"""Tests of brutus.priors.astrometric."""

import numpy as np

from brutus.priors import logp_parallax


class TestAstrometricPriors:
    """Test the parallax likelihood."""

    def test_logp_parallax_valid_measurement(self):
        """Test the parallax likelihood with a valid measurement."""
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
        """Test the parallax likelihood with invalid measurements."""
        parallaxes = np.array([0.5, 1.0, 1.5])

        # Invalid measurements give a flat likelihood (0 in log)
        cases = [
            (np.nan, 0.1),  # NaN measurement
            (1.0, np.nan),  # NaN error
            (1.0, 0.0),  # Zero error
            (1.0, -0.1),  # Negative error
        ]

        for p_meas, p_err in cases:
            logp = logp_parallax(parallaxes, p_meas, p_err)
            assert np.allclose(logp, 0.0)
