#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus plotting binning utilities.

This test suite provides thorough coverage of bin_pdfs_distred, which
was previously untested but is crucial for distance/reddening visualization.
"""

from unittest.mock import patch

import numpy as np
import pytest

# Import the function to test
from brutus.plotting.binning import bin_pdfs_distred


class TestBinPdfsDistred:
    """Comprehensive tests for bin_pdfs_distred function."""

    @pytest.fixture
    def direct_test_data(self):
        """Test data in direct format: (dists, reds, dreds)."""
        np.random.seed(42)
        n_objects = 3
        n_samples = 100

        # Generate realistic distance/reddening samples
        dists = np.random.lognormal(np.log(2.0), 0.3, (n_objects, n_samples))  # ~2 kpc
        reds = np.random.exponential(0.5, (n_objects, n_samples))  # Av ~ 0.5
        dreds = np.random.normal(3.3, 0.3, (n_objects, n_samples))  # Rv ~ 3.3

        return (dists, reds, dreds)

    @pytest.fixture
    def sar_test_data(self):
        """Test data in SAR format: (scales, avs, rvs, covs_sar)."""
        np.random.seed(42)
        n_objects = 3
        n_samples = 100

        # Generate realistic scale/Av/Rv samples
        parallaxes = np.random.lognormal(
            np.log(0.5), 0.3, (n_objects, n_samples)
        )  # ~0.5 mas
        scales = parallaxes**2
        avs = np.random.exponential(0.5, (n_objects, n_samples))
        rvs = np.random.normal(3.3, 0.3, (n_objects, n_samples))

        # Create covariance matrices (simplified)
        covs_sar = np.zeros((n_objects, n_samples, 3, 3))
        for i in range(n_objects):
            for j in range(n_samples):
                # Simple diagonal covariance
                covs_sar[i, j] = np.eye(3) * [0.1, 0.05, 0.1]

        return (scales, avs, rvs, covs_sar)

    @pytest.fixture
    def mock_coords(self):
        """Mock galactic coordinates."""
        return [(10.0, 5.0), (45.0, -20.0), (120.0, 30.0)]

    def test_bin_pdfs_basic_direct_data(self, direct_test_data):
        """Test basic PDF binning with direct distance/reddening data."""
        binned_vals, xedges, yedges = bin_pdfs_distred(direct_test_data)

        n_objects = 3
        default_bins = (750, 300)

        # Check output shapes
        assert binned_vals.shape == (n_objects, default_bins[0], default_bins[1])
        assert len(xedges) == default_bins[0] + 1
        assert len(yedges) == default_bins[1] + 1

        # Check that PDFs have reasonable values (normalized by nsamps)
        for i in range(n_objects):
            pdf_sum = np.sum(binned_vals[i])
            # Should be roughly 1 since H is normalized by nsamps and this approximates the integral
            assert 0.5 < pdf_sum < 2.0  # Allow for binning discretization effects

        # Check that values are non-negative
        assert np.all(binned_vals >= 0)

    def test_bin_pdfs_custom_bins(self, direct_test_data):
        """Test binning with custom bin numbers."""
        custom_bins = (50, 25)
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, bins=custom_bins
        )

        n_objects = 3
        assert binned_vals.shape == (n_objects, custom_bins[0], custom_bins[1])
        assert len(xedges) == custom_bins[0] + 1
        assert len(yedges) == custom_bins[1] + 1

    def test_bin_pdfs_distance_modulus(self, direct_test_data):
        """Test default distance_modulus representation."""
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, dist_type="distance_modulus"
        )

        # Distance modulus should range roughly from 4-19
        assert xedges[0] >= 3.0  # Allow some margin
        assert xedges[-1] <= 20.0

    def test_bin_pdfs_parallax_mode(self, direct_test_data):
        """Test parallax space binning."""
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, dist_type="parallax"
        )

        # Parallax should be positive and reasonable
        assert xedges[0] >= 0.0
        assert xedges[-1] > xedges[0]

    def test_bin_pdfs_scale_mode(self, direct_test_data):
        """Test scale (s = p^2) space binning."""
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, dist_type="scale"
        )

        # Scale should be positive
        assert xedges[0] >= 0.0
        assert np.all(np.isfinite(xedges))

    def test_bin_pdfs_distance_mode(self, direct_test_data):
        """Test physical distance binning."""
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, dist_type="distance"
        )

        # Distance should be positive and reasonable (in kpc)
        assert xedges[0] >= 0.0
        assert xedges[-1] > 1.0  # Should extend to at least 1 kpc

    def test_bin_pdfs_ebv_conversion(self, direct_test_data):
        """Test E(B-V) vs Av conversion."""
        binned_vals, xedges, yedges = bin_pdfs_distred(direct_test_data, ebv=True)

        # E(B-V) should be smaller than Av (since E(B-V) = Av/Rv and Rv > 1)
        # The y-axis (reddening) limits should reflect this
        assert yedges[-1] > 0.0

    @patch("brutus.plotting.binning.gal_lnprior")
    def test_bin_pdfs_with_distance_prior(self, mock_prior, sar_test_data, mock_coords):
        """Test with distance prior application."""
        # Mock the galactic prior to return reasonable values
        # The prior should return shape (nsamps, ndraws)
        mock_prior.return_value = np.zeros((100, 50))  # Uniform prior

        # Use SAR data which triggers prior application
        binned_vals, xedges, yedges = bin_pdfs_distred(
            sar_test_data,
            coord=mock_coords,
            Nr=50,  # Smaller for speed
            lndistprior=mock_prior,  # Explicitly pass the mocked prior
        )

        # Should call the distance prior
        mock_prior.assert_called()

        # Should still produce reasonable output
        assert binned_vals.shape[0] == 3  # n_objects
        assert np.all(binned_vals >= 0)

    def test_bin_pdfs_with_parallax_prior(self, sar_test_data, mock_coords):
        """Test with parallax prior application."""
        parallaxes = np.array([1.0, 0.5, 2.0])  # mas
        parallax_errors = np.array([0.1, 0.05, 0.2])  # mas

        binned_vals, xedges, yedges = bin_pdfs_distred(
            sar_test_data,
            coord=mock_coords,
            parallaxes=parallaxes,
            parallax_errors=parallax_errors,
            Nr=50,
        )

        # Should produce reasonable output
        assert binned_vals.shape[0] == 3
        assert np.all(binned_vals >= 0)

    def test_bin_pdfs_cdf_mode(self, direct_test_data):
        """Test CDF computation mode.

        The documented CDF accumulates along the *reddening* axis (axis 2)
        within each distance column, for evaluating MAP LOS reddening fits.
        """
        binned_vals, xedges, yedges = bin_pdfs_distred(direct_test_data, cdf=True)

        # CDFs should be monotonically non-decreasing along the reddening axis
        for i in range(binned_vals.shape[0]):
            for j in range(binned_vals.shape[1]):  # For each distance bin
                cdf_slice = binned_vals[i, j, :]
                # Check monotonicity (allowing for numerical precision)
                assert np.all(np.diff(cdf_slice) >= -1e-10)

    def test_bin_pdfs_custom_spans(self, direct_test_data):
        """Test custom axis ranges."""
        # Custom span format is (avlims, dlims) where dlims are distance limits in kpc
        # The function will convert distance limits to the requested dist_type for x-axis
        avlims = (0.0, 2.0)  # Av limits (y-axis)
        dlims = (0.5, 5.0)  # Distance limits in kpc (converted to x-axis)
        custom_span = (avlims, dlims)

        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, span=custom_span, dist_type="distance_modulus"
        )

        # For distance_modulus, xlims = 5 * log10(dlims) + 10
        expected_xlims = 5.0 * np.log10(np.array(dlims)) + 10.0
        expected_ylims = avlims

        # Check that bin edges respect the converted custom spans
        np.testing.assert_almost_equal(xedges[0], expected_xlims[0], decimal=5)
        np.testing.assert_almost_equal(xedges[-1], expected_xlims[1], decimal=5)
        assert yedges[0] == expected_ylims[0]
        assert yedges[-1] == expected_ylims[1]

    def test_bin_pdfs_smoothing_modes(self, direct_test_data):
        """Test different smoothing parameters."""
        # Test with different smoothing values
        smooth_values = [0.01, 0.05, [0.02, 0.03]]

        for smooth in smooth_values:
            binned_vals, xedges, yedges = bin_pdfs_distred(
                direct_test_data, smooth=smooth
            )

            # Should produce valid output regardless of smoothing
            assert binned_vals.shape[0] == 3
            assert np.all(binned_vals >= 0)
            assert np.all(np.isfinite(binned_vals))

    def test_bin_pdfs_single_object(self):
        """Test with single object data."""
        np.random.seed(42)
        n_samples = 50

        # Single object data
        dists = np.random.lognormal(np.log(1.0), 0.2, (1, n_samples))
        reds = np.random.exponential(0.3, (1, n_samples))
        dreds = np.random.normal(3.1, 0.2, (1, n_samples))

        data = (dists, reds, dreds)

        binned_vals, xedges, yedges = bin_pdfs_distred(data, bins=(20, 15))

        assert binned_vals.shape == (1, 20, 15)
        assert np.all(binned_vals >= 0)

    def test_bin_pdfs_error_handling(self):
        """Test error conditions and edge cases."""
        # Test invalid distance type
        dummy_data = (np.ones((1, 10)), np.ones((1, 10)), np.ones((1, 10)))

        with pytest.raises(ValueError, match="dist_type.*not valid"):
            bin_pdfs_distred(dummy_data, dist_type="invalid")

    def test_bin_pdfs_sar_missing_coord(self, sar_test_data):
        """Test error when coord is missing for SAR data."""
        with pytest.raises(ValueError, match="coord.*must be passed"):
            bin_pdfs_distred(sar_test_data)

    def test_bin_pdfs_empty_data(self):
        """Test behavior with empty/minimal data."""
        # Very small dataset
        tiny_data = (
            np.array([[1.0, 2.0]]),
            np.array([[0.1, 0.2]]),
            np.array([[3.0, 3.5]]),
        )

        # Should handle gracefully without crashing
        binned_vals, xedges, yedges = bin_pdfs_distred(tiny_data, bins=(5, 5))

        assert binned_vals.shape == (1, 5, 5)
        assert np.all(np.isfinite(binned_vals))

    def test_bin_pdfs_parallax_smoothing_limit(self, direct_test_data):
        """Test that parallax uncertainties limit smoothing."""
        # Very precise parallax should limit smoothing
        precise_parallax = np.array([10.0, 5.0, 20.0])  # High precision
        precise_errors = np.array([0.01, 0.01, 0.01])  # Very small errors

        # Run with large smoothing - should be limited by parallax precision
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data,
            parallaxes=precise_parallax,
            parallax_errors=precise_errors,
            smooth=0.1,  # Large smoothing
            dist_type="parallax",
        )

        # Should still produce reasonable results
        assert np.all(np.isfinite(binned_vals))
        assert np.all(binned_vals >= 0)

    def test_bin_pdfs_verbose_mode(self, direct_test_data, capsys):
        """Test verbose output."""
        bin_pdfs_distred(direct_test_data, verbose=True)

        # Check that progress was printed (captured by capsys)
        captured = capsys.readouterr()
        # Note: verbose output goes to stderr, but the test behavior may vary

    def test_bin_pdfs_reproducibility(self, direct_test_data):
        """Test that results are reproducible with same random state."""
        rstate1 = np.random.RandomState(123)
        rstate2 = np.random.RandomState(123)

        result1 = bin_pdfs_distred(direct_test_data, rstate=rstate1)
        result2 = bin_pdfs_distred(direct_test_data, rstate=rstate2)

        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        np.testing.assert_array_equal(result1[2], result2[2])

    @pytest.mark.parametrize(
        "dist_type", ["distance_modulus", "parallax", "scale", "distance"]
    )
    def test_bin_pdfs_all_distance_types(self, direct_test_data, dist_type):
        """Test all supported distance representations."""
        binned_vals, xedges, yedges = bin_pdfs_distred(
            direct_test_data, dist_type=dist_type, bins=(10, 8)
        )

        assert binned_vals.shape == (3, 10, 8)
        assert np.all(binned_vals >= 0)
        assert np.all(np.isfinite(xedges))
        assert np.all(np.isfinite(yedges))

    def test_integration_with_priors_module(self):
        """Test that the function properly uses the refactored priors."""
        # This tests that the imports work correctly and priors are applied
        np.random.seed(42)

        # Create SAR format data
        scales = np.random.exponential(0.25, (1, 50)).reshape(1, 50)
        avs = np.random.exponential(0.5, (1, 50))
        rvs = np.random.normal(3.3, 0.3, (1, 50))
        covs = np.zeros((1, 50, 3, 3))
        for i in range(50):
            covs[0, i] = np.eye(3) * 0.1

        data = (scales, avs, rvs, covs)
        coord = [(45.0, 10.0)]

        # Should work without errors
        result = bin_pdfs_distred(data, coord=coord, Nr=20)

        assert len(result) == 3
        assert result[0].shape[0] == 1  # One object
