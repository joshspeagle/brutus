#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for brutus corner plotting functions.

This test suite provides comprehensive coverage for the cornerplot function
and related corner plot visualization functionality.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.plotting.corner import cornerplot


class TestCornerplot:
    """Comprehensive tests for cornerplot function."""

    @pytest.fixture
    def basic_params(self):
        """Generate basic structured parameter array for testing."""
        np.random.seed(42)
        n_models = 100

        # Create a structured array with stellar parameters
        dtype = np.dtype(
            [
                ("mass", "f4"),
                ("age", "f4"),
                ("feh", "f4"),
                ("agewt", "f4"),  # This should be ignored
            ]
        )

        params = np.empty(n_models, dtype=dtype)
        params["mass"] = np.random.uniform(0.5, 2.0, n_models)
        params["age"] = np.random.uniform(0.1, 10.0, n_models)
        params["feh"] = np.random.uniform(-2.0, 0.5, n_models)
        params["agewt"] = np.ones(n_models)  # Should be ignored

        return params

    @pytest.fixture
    def sample_indices(self):
        """Generate sample indices for resampling."""
        np.random.seed(42)
        n_samples = 50
        n_models = 100

        # Create indices that resample from the model grid
        idxs = np.random.choice(n_models, n_samples, replace=True)
        return idxs

    @pytest.fixture
    def direct_data(self):
        """Generate direct (dists, reds, dreds) data format."""
        np.random.seed(42)
        n_samples = 50

        # Generate realistic distance/reddening samples
        dists = np.random.lognormal(np.log(2.0), 0.3, n_samples)  # ~2 kpc
        reds = np.random.exponential(0.5, n_samples)  # Av ~ 0.5
        dreds = np.random.normal(3.3, 0.3, n_samples)  # Rv ~ 3.3

        return (dists, reds, dreds)

    def test_cornerplot_basic_direct_data(
        self, sample_indices, basic_params, direct_data
    ):
        """Test basic cornerplot with direct data format."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            applied_parallax=False,  # Don't require parallax for basic test
            span=[
                0.9 for _ in range(7)
            ],  # 3 params + 4 distance/reddening (Av, Rv, Parallax, Distance)
        )

        # Should return figure and axes
        assert fig is not None
        assert axes is not None
        assert axes.shape == (7, 7)  # 3 params + Av + Rv + Parallax + Distance

        plt.close(fig)

    def test_cornerplot_does_not_mutate_labels(
        self, sample_indices, basic_params, direct_data
    ):
        """A caller-supplied labels list must not be mutated, and reusing it
        must not accumulate Av/Rv/Parallax/Distance entries.

        Regression: cornerplot appended those names onto the caller's list in
        place, so a second call with the same list raised ValueError (no field
        of name 'Av') or produced mislabeled axes.
        """
        labels = ["mass", "feh"]
        fig1, _ = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            labels=labels,
            applied_parallax=False,
            span=[0.9 for _ in range(6)],
        )
        plt.close(fig1)
        assert labels == ["mass", "feh"]  # not mutated in place

        # Reuse the same list object: must not raise or mislabel.
        fig2, _ = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            labels=labels,
            applied_parallax=False,
            span=[0.9 for _ in range(6)],
        )
        plt.close(fig2)
        assert labels == ["mass", "feh"]

    def test_cornerplot_with_parallax(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with parallax information."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            parallax=2.0,
            parallax_err=0.1,
            applied_parallax=True,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_no_parallax_applied(
        self, sample_indices, basic_params, direct_data
    ):
        """Test cornerplot with applied_parallax=False."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_with_weights(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with sample weights."""
        np.random.seed(42)
        weights = np.random.exponential(1.0, len(sample_indices))

        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            weights=weights,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_custom_span(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with custom span specification."""
        # Mix of explicit bounds and fractions
        span = [(0.5, 2.0), 0.95, (-1.5, 0.3), 0.9, 0.9, 0.9, 0.95]

        fig, axes = cornerplot(
            sample_indices, direct_data, basic_params, applied_parallax=False, span=span
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_quantiles_and_truths(
        self, sample_indices, basic_params, direct_data
    ):
        """Test cornerplot with quantiles and truth values."""
        quantiles = [0.16, 0.5, 0.84]  # 1-sigma equivalent
        truths = [1.0, 5.0, -0.5, 0.3, 3.1, 0.5, 2.0]  # Values for each parameter

        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            quantiles=quantiles,
            truths=truths,
            truth_color="red",
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_smooth_options(self, sample_indices, basic_params, direct_data):
        """Test different smoothing options."""
        # Test integer smoothing (histogram bins)
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            smooth=10,  # 10 bins
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )
        assert fig is not None
        plt.close(fig)

        # Test float smoothing (kernel)
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            smooth=0.05,  # 5% kernel
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )
        assert fig is not None
        plt.close(fig)

    def test_cornerplot_show_titles(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with titles."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            show_titles=True,
            title_fmt=".3f",
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_verbose_mode(self, sample_indices, basic_params, direct_data):
        """Test verbose output."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            fig, axes = cornerplot(
                sample_indices,
                direct_data,
                basic_params,
                verbose=True,
                quantiles=[0.16, 0.5, 0.84],
                applied_parallax=False,
                span=[0.9 for _ in range(7)],
            )

        output = f.getvalue()
        assert "Quantiles:" in output  # Should print quantile information

        plt.close(fig)

    def test_cornerplot_error_conditions(self, sample_indices, basic_params):
        """Test error handling."""
        # Create properly sized dummy data that matches sample_indices
        n_samples = len(sample_indices)
        dummy_data = (np.ones(n_samples), np.ones(n_samples), np.ones(n_samples))

        # Test parallax error without both parallax and parallax_err
        with pytest.raises(ValueError, match="must be provided together"):
            cornerplot(
                sample_indices,
                dummy_data,
                basic_params,
                parallax=1.0,  # Missing parallax_err
                applied_parallax=True,
            )

        # Test dimension mismatch in span
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cornerplot(
                sample_indices,
                dummy_data,
                basic_params,
                applied_parallax=False,
                span=[0.9, 0.9],
            )  # Too few elements

        # Test weights dimension mismatch
        wrong_weights = np.ones(5)  # Different size than sample_indices
        with pytest.raises(ValueError, match="disagree"):
            cornerplot(
                sample_indices,
                dummy_data,
                basic_params,
                applied_parallax=False,
                weights=wrong_weights,
            )

    def test_cornerplot_sar_data(self, sample_indices, basic_params):
        """Test cornerplot with SAR data format."""
        np.random.seed(42)
        n_samples = 50

        # Create SAR format data
        scales = np.random.exponential(1.0, n_samples)
        avs = np.random.exponential(0.5, n_samples)
        rvs = np.random.normal(3.3, 0.3, n_samples)

        # Create proper positive semidefinite covariance matrices
        covs_sar = np.zeros((n_samples, 3, 3))
        for i in range(n_samples):
            A = np.random.randn(3, 3)
            covs_sar[i] = np.dot(A, A.T) * 0.01

        sar_data = (scales, avs, rvs, covs_sar)
        coord = (45.0, 30.0)  # galactic coordinates

        fig, axes = cornerplot(
            sample_indices,
            sar_data,
            basic_params,
            coord=coord,
            Nr=10,  # Small number for speed
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_sar_no_coord_error(self, sample_indices, basic_params):
        """Test SAR data without coordinates raises error."""
        np.random.seed(42)
        n_samples = len(sample_indices)  # Match the sample_indices size

        scales = np.random.exponential(1.0, n_samples)
        avs = np.random.exponential(0.5, n_samples)
        rvs = np.random.normal(3.3, 0.3, n_samples)
        covs_sar = np.zeros((n_samples, 3, 3))
        for i in range(n_samples):
            A = np.random.randn(3, 3)
            covs_sar[i] = np.dot(A, A.T) * 0.01

        sar_data = (scales, avs, rvs, covs_sar)

        with pytest.raises(ValueError, match="coord.*must be passed"):
            cornerplot(
                sample_indices,
                sar_data,
                basic_params,
                applied_parallax=False,  # Disable parallax check first
                coord=None,
            )  # Should raise error for SAR data

    def test_cornerplot_return_values(self, sample_indices, basic_params, direct_data):
        """Test return value format."""
        result = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        # Should return tuple of (fig, axes)
        assert len(result) == 2
        fig, axes = result

        # Check types
        assert hasattr(fig, "subplots_adjust")  # Figure-like object
        assert hasattr(axes, "shape")  # Array-like object

        plt.close(fig)

    def test_cornerplot_no_quantiles(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with no quantiles."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            quantiles=None,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_custom_colors(self, sample_indices, basic_params, direct_data):
        """Test cornerplot with custom colors."""
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            color="blue",
            truth_color="green",
            pcolor="orange",
            truths=[1.0, 5.0, -0.5, 0.3, 3.1, 0.5, 2.0],
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )

        assert fig is not None
        plt.close(fig)

    def test_cornerplot_tick_options(self, sample_indices, basic_params, direct_data):
        """Test tick-related options."""
        # Test with no ticks
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            max_n_ticks=0,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )
        assert fig is not None
        plt.close(fig)

        # Test with math text
        fig, axes = cornerplot(
            sample_indices,
            direct_data,
            basic_params,
            use_math_text=True,
            applied_parallax=False,
            span=[0.9 for _ in range(7)],
        )
        assert fig is not None
        plt.close(fig)

    def test_cornerplot_single_parameter(self):
        """Test cornerplot with single parameter."""
        np.random.seed(42)
        n_models = 50
        n_samples = 20

        # Single parameter
        dtype = np.dtype([("mass", "f4")])
        params = np.empty(n_models, dtype=dtype)
        params["mass"] = np.random.uniform(0.5, 2.0, n_models)

        idxs = np.random.choice(n_models, n_samples, replace=True)

        # Simple direct data
        dists = np.random.lognormal(np.log(2.0), 0.3, n_samples)
        reds = np.random.exponential(0.5, n_samples)
        dreds = np.random.normal(3.3, 0.3, n_samples)
        data = (dists, reds, dreds)

        fig, axes = cornerplot(
            idxs, data, params, applied_parallax=False, span=[0.9, 0.9, 0.9, 0.9, 0.9]
        )

        assert fig is not None
        plt.close(fig)


class TestCornerplotIntegration:
    """Integration tests for cornerplot with different data scenarios."""

    def test_cornerplot_realistic_stellar_data(self):
        """Test with realistic stellar parameter distributions."""
        np.random.seed(42)
        n_models = 100
        n_samples = 40

        # Create realistic stellar parameter grid
        dtype = np.dtype(
            [
                ("mass", "f4"),
                ("age", "f4"),
                ("feh", "f4"),
            ]
        )

        params = np.empty(n_models, dtype=dtype)

        # Main sequence stars concentrated around solar values
        params["mass"] = np.random.lognormal(np.log(1.0), 0.3, n_models)
        params["age"] = np.random.uniform(0.1, 13.0, n_models)
        params["feh"] = np.random.normal(0.0, 0.2, n_models)

        # Sample with preference for solar-like stars
        mass_weights = np.exp(-(((params["mass"] - 1.0) / 0.2) ** 2))
        idxs = np.random.choice(
            n_models, n_samples, p=mass_weights / mass_weights.sum()
        )

        # Distance and reddening data
        dists = np.random.lognormal(
            np.log(1.0), 0.5, n_samples
        )  # Mix of nearby/distant
        reds = np.random.exponential(0.3, n_samples)
        dreds = np.random.normal(3.1, 0.2, n_samples)
        data = (dists, reds, dreds)

        fig, axes = cornerplot(
            idxs,
            data,
            params,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            applied_parallax=False,
            span=[0.99 for _ in range(7)],  # 3 params + 4 distance/reddening
        )

        # Should handle this realistic case without issues
        assert fig is not None
        assert axes.shape == (7, 7)

        plt.close(fig)

    def test_cornerplot_edge_cases(self):
        """Test various edge cases."""
        np.random.seed(42)

        # Very small sample
        n_models = 10
        n_samples = 5

        dtype = np.dtype([("mass", "f4")])
        params = np.empty(n_models, dtype=dtype)
        params["mass"] = np.random.uniform(0.8, 1.2, n_models)

        idxs = np.random.choice(n_models, n_samples, replace=True)

        # Minimal data
        dists = np.random.uniform(1.0, 3.0, n_samples)
        reds = np.random.uniform(0.0, 0.5, n_samples)
        dreds = np.random.uniform(2.5, 4.0, n_samples)
        data = (dists, reds, dreds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings for small dataset
            fig, axes = cornerplot(
                idxs,
                data,
                params,
                applied_parallax=False,
                span=[0.95, 0.95, 0.95, 0.95, 0.95],
            )

        assert fig is not None
        plt.close(fig)
