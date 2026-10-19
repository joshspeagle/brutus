#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus plotting utilities.

This test suite provides comprehensive coverage for the hist2d function
and other plotting utility functions.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.plotting.utils import hist2d


class TestHist2d:
    """Comprehensive tests for hist2d function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample 2D data for testing."""
        rng = np.random.RandomState(42)
        n_samples = 200

        # Create correlated 2D data
        mean = [0, 0]
        cov = [[1, 0.7], [0.7, 1]]  # Some correlation
        data = rng.multivariate_normal(mean, cov, n_samples)
        x, y = data.T

        return x, y

    @pytest.fixture
    def weighted_sample_data(self):
        """Generate sample 2D data with weights."""
        rng = np.random.RandomState(42)
        n_samples = 100

        x = rng.normal(0, 1, n_samples)
        y = rng.normal(0, 1, n_samples)
        weights = rng.exponential(1, n_samples)  # Exponential weights

        return x, y, weights

    def test_hist2d_basic(self, sample_data):
        """Test basic hist2d functionality."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))
        hist2d(x, y)

        # Should complete without errors
        plt.close()

    def test_hist2d_with_weights(self, weighted_sample_data):
        """Test hist2d with sample weights."""
        x, y, weights = weighted_sample_data

        plt.figure(figsize=(6, 6))
        hist2d(x, y, weights=weights)

        plt.close()

    def test_hist2d_custom_span(self, sample_data):
        """Test custom span specification."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))

        # Test with explicit bounds
        hist2d(x, y, span=[(-2, 2), (-2, 2)])
        plt.close()

        # Test with fraction specification
        plt.figure(figsize=(6, 6))
        hist2d(x, y, span=[0.95, 0.95])
        plt.close()

    def test_hist2d_levels(self, sample_data):
        """Test custom contour levels."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))
        custom_levels = [0.68, 0.95]  # 1 and 2 sigma equivalent
        hist2d(x, y, levels=custom_levels)

        plt.close()

    def test_hist2d_color_options(self, sample_data):
        """Test different color specifications."""
        x, y = sample_data

        # Test custom color
        plt.figure(figsize=(6, 6))
        hist2d(x, y, color="red")
        plt.close()

        # Test different color
        plt.figure(figsize=(6, 6))
        hist2d(x, y, color="blue")
        plt.close()

    def test_hist2d_plot_options(self, sample_data):
        """Test different plotting mode combinations."""
        x, y = sample_data

        # Plot datapoints only
        plt.figure(figsize=(6, 6))
        hist2d(x, y, plot_datapoints=True, plot_density=False, plot_contours=False)
        plt.close()

        # Plot density only
        plt.figure(figsize=(6, 6))
        hist2d(x, y, plot_datapoints=False, plot_density=True, plot_contours=False)
        plt.close()

        # Plot contours only
        plt.figure(figsize=(6, 6))
        hist2d(x, y, plot_datapoints=False, plot_density=False, plot_contours=True)
        plt.close()

    def test_hist2d_contour_options(self, sample_data):
        """Test contour-specific options."""
        x, y = sample_data

        # Test no fill contours
        plt.figure(figsize=(6, 6))
        hist2d(x, y, no_fill_contours=True)
        plt.close()

        # Test unfilled contours
        plt.figure(figsize=(6, 6))
        hist2d(x, y, fill_contours=False)
        plt.close()

        # Test filled contours
        plt.figure(figsize=(6, 6))
        hist2d(x, y, fill_contours=True)
        plt.close()

    def test_hist2d_kwargs(self, sample_data):
        """Test additional keyword arguments."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))

        contour_kwargs = {"linewidths": 2, "alpha": 0.8}
        contourf_kwargs = {"alpha": 0.6}
        data_kwargs = {"marker": "s", "alpha": 0.3}

        hist2d(
            x,
            y,
            contour_kwargs=contour_kwargs,
            contourf_kwargs=contourf_kwargs,
            data_kwargs=data_kwargs,
            plot_datapoints=True,
        )

        plt.close()

    def test_hist2d_smoothing(self, sample_data):
        """Test different smoothing options."""
        x, y = sample_data

        # Test different smoothing values
        for smooth in [0.01, 0.05, [0.02, 0.03]]:
            plt.figure(figsize=(6, 6))
            hist2d(x, y, smooth=smooth)
            plt.close()

        # Test integer smoothing (bin-based)
        for smooth in [10, 20, [15, 25]]:
            plt.figure(figsize=(6, 6))
            hist2d(x, y, smooth=smooth)
            plt.close()

    def test_hist2d_edge_cases(self):
        """Test edge cases and error conditions."""
        # Very small dataset
        x_small = np.array([0, 1])
        y_small = np.array([0, 1])

        plt.figure(figsize=(6, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings for small dataset
            hist2d(x_small, y_small)
        plt.close()

        # Single point (should handle gracefully)
        x_single = np.array([0])
        y_single = np.array([0])

        plt.figure(figsize=(6, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings for single point
            hist2d(x_single, y_single)
        plt.close()

    def test_hist2d_no_dynamic_range_warning(self):
        """Test warning when data has no dynamic range."""
        # All points the same
        x_constant = np.ones(10)
        y_constant = np.ones(10)

        plt.figure(figsize=(6, 6))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hist2d(x_constant, y_constant)
            # Should produce warnings about identical axis limits
            assert len(w) >= 1
            # Check that at least one warning is about axis limits
            warning_messages = [str(warning.message) for warning in w]
            assert any("identical" in msg and "lims" in msg for msg in warning_messages)

        plt.close()

    def test_hist2d_insufficient_points_warning(self):
        """Test warning when too few points for contours."""
        # Very sparse data that may trigger contour warnings
        x_sparse = np.array([0, 1, 2])
        y_sparse = np.array([0, 1, 2])

        plt.figure(figsize=(6, 6))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hist2d(x_sparse, y_sparse)
            # May or may not produce warnings depending on the data

        plt.close()

    def test_hist2d_span_dimension_mismatch(self, sample_data):
        """Test error when span dimensions don't match."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            hist2d(x, y, span=[0.95])  # Only 1 element, should be 2

        plt.close()

    def test_hist2d_with_ax_parameter(self, sample_data):
        """Test providing custom axes."""
        x, y = sample_data

        fig, ax = plt.subplots(figsize=(6, 6))
        hist2d(x, y, ax=ax)

        plt.close()

    def test_hist2d_quantile_span_processing(self, sample_data):
        """Test span processing with quantiles."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))

        # Mixed span types
        hist2d(x, y, span=[(-1, 1), 0.9])  # Explicit bounds + quantile

        plt.close()

    def test_hist2d_weighted_quantiles(self, weighted_sample_data):
        """Test quantile calculation with weights."""
        x, y, weights = weighted_sample_data

        plt.figure(figsize=(6, 6))
        hist2d(x, y, weights=weights, span=[0.8, 0.8])

        plt.close()

    def test_hist2d_colormap_generation(self, sample_data):
        """Test internal colormap generation."""
        x, y = sample_data

        # This tests the internal LinearSegmentedColormap creation
        plt.figure(figsize=(6, 6))
        hist2d(x, y, color="green", plot_density=True)

        plt.close()

    def test_hist2d_contour_level_processing(self, sample_data):
        """Test contour level computation and sorting."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))

        # Test with levels that might create duplicates
        custom_levels = [0.1, 0.5, 0.9]
        hist2d(x, y, levels=custom_levels)

        plt.close()

    def test_hist2d_histogram_extension(self, sample_data):
        """Test histogram extension for contour edges."""
        x, y = sample_data

        # This tests the H2 extension logic for contour plotting at edges
        plt.figure(figsize=(6, 6))
        hist2d(x, y, smooth=5)  # Integer smoothing to test different code path

        plt.close()

    @pytest.mark.parametrize(
        "plot_datapoints,plot_density,plot_contours",
        [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_hist2d_all_combinations(
        self, sample_data, plot_datapoints, plot_density, plot_contours
    ):
        """Test all valid combinations of plotting options."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))
        hist2d(
            x,
            y,
            plot_datapoints=plot_datapoints,
            plot_density=plot_density,
            plot_contours=plot_contours,
        )

        plt.close()

    def test_hist2d_return_value(self, sample_data):
        """Test that hist2d doesn't return anything (pure plotting function)."""
        x, y = sample_data

        plt.figure(figsize=(6, 6))
        result = hist2d(x, y)

        # hist2d should not return anything
        assert result is None

        plt.close()


class TestHist2dIntegration:
    """Integration tests for hist2d with other components."""

    def test_hist2d_realistic_astronomical_data(self):
        """Test with realistic astronomical data distributions."""
        rng = np.random.RandomState(42)

        # Simulate stellar parameters with realistic distributions
        # Distance modulus vs. reddening
        n_stars = 300

        # Distance moduli clustered around a few values (like star clusters)
        dm_clusters = [12.0, 14.5, 16.0]
        dm = []
        av = []

        for dm_center in dm_clusters:
            n_cluster = n_stars // len(dm_clusters)
            dm_cluster = rng.normal(dm_center, 0.3, n_cluster)
            # Reddening correlated with distance for realistic ISM
            av_cluster = rng.exponential(0.2 * (dm_center - 10), n_cluster)
            av_cluster = np.maximum(av_cluster, 0)  # No negative reddening

            dm.extend(dm_cluster)
            av.extend(av_cluster)

        dm = np.array(dm)
        av = np.array(av)

        plt.figure(figsize=(8, 6))
        hist2d(
            dm,
            av,
            plot_datapoints=True,
            plot_contours=True,
            data_kwargs={"alpha": 0.3, "ms": 1},
        )

        plt.xlabel("Distance Modulus")
        plt.ylabel("Reddening (mag)")
        plt.close()

    def test_hist2d_performance_large_dataset(self):
        """Test performance with larger dataset."""
        rng = np.random.RandomState(42)
        n_large = 2000

        x = rng.standard_t(3, n_large)  # Heavy-tailed distribution
        y = rng.standard_t(3, n_large)

        plt.figure(figsize=(6, 6))
        hist2d(x, y, smooth=0.02)

        plt.close()
