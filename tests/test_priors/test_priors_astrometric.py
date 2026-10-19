"""Tests of brutus.priors.astrometric."""

import numpy as np

from brutus.priors import convert_parallax_to_scale, logp_parallax, logp_parallax_scale
from brutus.priors.astrometric import SCALE_MAX, SCALE_MIN


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


class TestConvertParallaxToScaleZeroError:
    """p_err <= 0 must return the uninformative fallback, not crash."""

    def test_zero_error_python_float(self):
        # Old code: ZeroDivisionError
        s_mean, s_std = convert_parallax_to_scale(1.0, 0.0)
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX

    def test_zero_error_numpy_scalar(self):
        # Old code: silent (p**2, 0.0) delta-function prior
        s_mean, s_std = convert_parallax_to_scale(np.float64(1.0), np.float64(0.0))
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX

    def test_negative_error(self):
        s_mean, s_std = convert_parallax_to_scale(1.0, -0.1)
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX
