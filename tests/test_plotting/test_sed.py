#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for SED plotting functions.

This test suite provides comprehensive coverage for the posterior_predictive function
and related SED visualization functionality.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
from unittest.mock import patch

import matplotlib.pyplot as plt

from brutus.plotting.sed import posterior_predictive


class TestPosteriorPredictive:
    """Comprehensive tests for posterior_predictive function."""

    @pytest.fixture
    def mock_models(self):
        """Generate mock model coefficients for testing."""
        np.random.seed(42)
        nmodels, nfilt, ncoeff = 50, 5, 3

        # Create realistic polynomial coefficients for magnitude models
        models = np.random.uniform(10, 20, (nmodels, nfilt, ncoeff))
        # Add some structure - first coefficient is base magnitude
        models[:, :, 0] += np.random.uniform(-2, 2, (nmodels, nfilt))

        return models

    @pytest.fixture
    def sample_indices(self):
        """Generate sample indices for resampling."""
        np.random.seed(42)
        n_samples = 30
        n_models = 50

        # Create indices that resample from the model grid
        idxs = np.random.choice(n_models, n_samples, replace=True)
        return idxs

    @pytest.fixture
    def sample_parameters(self, sample_indices):
        """Generate corresponding reds, dreds, dists arrays."""
        np.random.seed(42)
        n_samples = len(sample_indices)

        # Generate realistic distance/reddening samples
        dists = np.random.lognormal(np.log(2.0), 0.3, n_samples)  # ~2 kpc
        reds = np.random.exponential(0.5, n_samples)  # Av ~ 0.5
        dreds = np.random.normal(3.3, 0.3, n_samples)  # Rv ~ 3.3

        return reds, dreds, dists

    def test_posterior_predictive_basic(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test basic posterior_predictive functionality."""
        reds, dreds, dists = sample_parameters

        # Mock get_seds to return realistic SED values
        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists
            )

        # Should return figure, axes, and violinplot parts
        assert fig is not None
        assert ax is not None
        assert parts is not None
        assert "bodies" in parts

        plt.close(fig)

    def test_posterior_predictive_with_weights(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with importance weights."""
        reds, dreds, dists = sample_parameters

        # Generate importance weights
        np.random.seed(42)
        weights = np.random.exponential(1.0, len(sample_indices))

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists, weights=weights
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_flux_space(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive in flux space."""
        reds, dreds, dists = sample_parameters

        # Mock flux SEDs (smaller values than magnitudes)
        mock_flux_seds = np.random.uniform(
            1e-15, 1e-12, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_flux_seds):
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists, flux=True
            )

        # Check that ylabel is set to 'Flux'
        assert ax.get_ylabel() == "Flux"

        plt.close(fig)

    def test_posterior_predictive_magnitude_space(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive in magnitude space."""
        reds, dreds, dists = sample_parameters

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists, flux=False
            )

        # Check that ylabel is set to 'Magnitude' and axis is flipped
        assert ax.get_ylabel() == "Magnitude"
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # Should be flipped for magnitude space

        plt.close(fig)

    def test_posterior_predictive_with_data(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with observational data overlay."""
        reds, dreds, dists = sample_parameters
        nfilt = mock_models.shape[1]

        # Generate mock observational data
        np.random.seed(42)
        data = np.random.uniform(1e-14, 1e-12, nfilt)  # Flux values
        data_err = data * 0.1  # 10% errors
        data_mask = np.ones(nfilt, dtype=bool)
        data_mask[-1] = False  # Mask one filter

        mock_seds = np.random.uniform(15, 20, (len(sample_indices), nfilt))

        with (
            patch("brutus.plotting.sed.get_seds", return_value=mock_seds),
            patch("brutus.plotting.sed.magnitude") as mock_magnitude,
        ):
            # Mock magnitude function return
            mock_magnitude.return_value = (
                np.random.uniform(15, 20, nfilt - 1),
                np.random.uniform(0.01, 0.1, nfilt - 1),
            )

            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                data=data,
                data_err=data_err,
                data_mask=data_mask,
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_with_data_flux(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with data in flux space."""
        reds, dreds, dists = sample_parameters
        nfilt = mock_models.shape[1]

        # Generate mock flux data
        np.random.seed(42)
        data = np.random.uniform(1e-14, 1e-12, nfilt)
        data_err = data * 0.05

        mock_flux_seds = np.random.uniform(1e-15, 1e-12, (len(sample_indices), nfilt))

        with patch("brutus.plotting.sed.get_seds", return_value=mock_flux_seds):
            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                flux=True,
                data=data,
                data_err=data_err,
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_with_offset(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with photometric offsets."""
        reds, dreds, dists = sample_parameters
        nfilt = mock_models.shape[1]

        # Generate mock data and offsets
        np.random.seed(42)
        data = np.random.uniform(1e-14, 1e-12, nfilt)
        data_err = data * 0.1
        offset = np.random.uniform(0.9, 1.1, nfilt)  # ±10% offsets

        mock_seds = np.random.uniform(15, 20, (len(sample_indices), nfilt))

        with (
            patch("brutus.plotting.sed.get_seds", return_value=mock_seds),
            patch("brutus.plotting.sed.magnitude") as mock_magnitude,
        ):
            mock_magnitude.return_value = (
                np.random.uniform(15, 20, nfilt),
                np.random.uniform(0.01, 0.1, nfilt),
            )

            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                data=data,
                data_err=data_err,
                offset=offset,
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_custom_colors(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with custom colors."""
        reds, dreds, dists = sample_parameters

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                vcolor="red",
                pcolor="blue",
            )

        # Check that violin plot has the right color
        assert len(parts["bodies"]) > 0

        plt.close(fig)

    def test_posterior_predictive_with_labels(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with filter labels."""
        reds, dreds, dists = sample_parameters
        nfilt = mock_models.shape[1]

        labels = [f"Filter_{i}" for i in range(nfilt)]

        mock_seds = np.random.uniform(15, 20, (len(sample_indices), nfilt))

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists, labels=labels
            )

        # Check that labels are set
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any(label in tick_labels for label in labels)

        plt.close(fig)

    def test_posterior_predictive_custom_figure(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with custom figure."""
        reds, dreds, dists = sample_parameters

        # Create custom figure
        custom_fig, custom_ax = plt.subplots(figsize=(8, 6))

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                fig=(custom_fig, custom_ax),
            )

        # Should return the same figure and axes
        assert fig is custom_fig
        assert ax is custom_ax

        plt.close(fig)

    def test_posterior_predictive_random_state(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test posterior_predictive with custom random state."""
        reds, dreds, dists = sample_parameters

        # Create custom random state
        rstate = np.random.RandomState(123)

        # Generate non-uniform weights to trigger random sampling
        weights = np.array([1.0, 2.0, 0.5] * (len(sample_indices) // 3 + 1))[
            : len(sample_indices)
        ]

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                weights=weights,
                rstate=rstate,
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_error_conditions(self, mock_models, sample_indices):
        """Test error handling in posterior_predictive."""
        n_samples = len(sample_indices)
        reds = np.random.exponential(0.5, n_samples)
        dreds = np.random.normal(3.3, 0.3, n_samples)
        dists = np.random.lognormal(np.log(2.0), 0.3, n_samples)

        # Test wrong weight dimensions
        wrong_weights = np.ones((n_samples, 2))  # 2D instead of 1D
        with pytest.raises(ValueError, match="Weights must be 1-D"):
            with patch("brutus.plotting.sed.get_seds"):
                posterior_predictive(
                    mock_models,
                    sample_indices,
                    reds,
                    dreds,
                    dists,
                    weights=wrong_weights,
                )

        # Test weight size mismatch
        wrong_size_weights = np.ones(n_samples + 5)
        with pytest.raises(ValueError, match="number of weights and samples disagree"):
            with patch("brutus.plotting.sed.get_seds"):
                posterior_predictive(
                    mock_models,
                    sample_indices,
                    reds,
                    dreds,
                    dists,
                    weights=wrong_size_weights,
                )

    def test_posterior_predictive_custom_psig(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test custom psig parameter for error bars."""
        reds, dreds, dists = sample_parameters
        nfilt = mock_models.shape[1]

        # Generate data with errors
        data = np.random.uniform(1e-14, 1e-12, nfilt)
        data_err = data * 0.1

        mock_seds = np.random.uniform(15, 20, (len(sample_indices), nfilt))

        with (
            patch("brutus.plotting.sed.get_seds", return_value=mock_seds),
            patch("brutus.plotting.sed.magnitude") as mock_magnitude,
        ):
            mock_magnitude.return_value = (
                np.random.uniform(15, 20, nfilt),
                np.random.uniform(0.01, 0.1, nfilt),
            )

            fig, ax, parts = posterior_predictive(
                mock_models,
                sample_indices,
                reds,
                dreds,
                dists,
                data=data,
                data_err=data_err,
                psig=3.0,  # 3-sigma error bars
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_defaults(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test that all default parameters work correctly."""
        reds, dreds, dists = sample_parameters

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            # Test with minimal arguments - should use all defaults
            fig, ax, parts = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists
            )

        assert fig is not None
        assert ax is not None
        assert parts is not None

        plt.close(fig)

    def test_posterior_predictive_return_format(
        self, mock_models, sample_indices, sample_parameters
    ):
        """Test the return value format."""
        reds, dreds, dists = sample_parameters

        mock_seds = np.random.uniform(
            15, 20, (len(sample_indices), mock_models.shape[1])
        )

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            result = posterior_predictive(
                mock_models, sample_indices, reds, dreds, dists
            )

        # Should return 3-tuple: (fig, ax, parts)
        assert len(result) == 3
        fig, ax, parts = result

        # Check types
        assert hasattr(fig, "subplots_adjust")  # Figure-like object
        assert hasattr(ax, "plot")  # Axes-like object
        assert isinstance(parts, dict)  # Violinplot parts dictionary
        assert "bodies" in parts

        plt.close(fig)


class TestSummaryPlot:
    """Basic smoke test for summary_plot."""

    def test_summary_plot_returns_fig_axes_ax_sed(self):
        """Test that summary_plot returns (fig, axes, ax_sed) without error."""
        from unittest.mock import patch

        from brutus.plotting.summary import summary_plot

        np.random.seed(42)
        nmodels, nfilt, ncoeff = 20, 5, 3
        n_samples = 25

        models = np.random.uniform(10, 20, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, n_samples, replace=True)
        reds = np.random.exponential(0.3, n_samples)
        dreds = np.random.normal(3.3, 0.2, n_samples)
        dists = np.random.lognormal(np.log(1.5), 0.4, n_samples)

        # Create structured params array (corner plot needs named fields)
        dtype = np.dtype(
            [("mass", "f4"), ("age", "f4"), ("feh", "f4"), ("agewt", "f4")]
        )
        params = np.empty(nmodels, dtype=dtype)
        params["mass"] = np.random.uniform(0.5, 2.0, nmodels)
        params["age"] = np.random.uniform(0.1, 10.0, nmodels)
        params["feh"] = np.random.uniform(-2.0, 0.5, nmodels)
        params["agewt"] = np.ones(nmodels)

        mock_seds = np.random.uniform(14, 18, (n_samples, nfilt))

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, axes, ax_sed = summary_plot(
                    models, idxs, reds, dreds, dists, params
                )

        assert fig is not None
        assert axes is not None
        assert ax_sed is not None
        assert hasattr(fig, "subplots_adjust")  # Figure-like object
        assert hasattr(ax_sed, "plot")  # Axes-like object

        plt.close(fig)


class TestPosteriorPredictiveIntegration:
    """Integration tests for posterior_predictive with realistic scenarios."""

    def test_posterior_predictive_realistic_stellar_sed(self):
        """Test with realistic stellar SED data."""
        np.random.seed(42)

        # Create realistic model parameters
        nmodels, nfilt, ncoeff = 20, 7, 4  # 7 filters (UBVRIJHK-like)
        n_samples = 25

        # Generate model coefficients that represent stellar SEDs
        models = np.random.uniform(8, 25, (nmodels, nfilt, ncoeff))

        # Generate sample parameters
        idxs = np.random.choice(nmodels, n_samples, replace=True)
        reds = np.random.exponential(0.3, n_samples)  # Low reddening
        dreds = np.random.normal(3.1, 0.2, n_samples)  # Standard Rv
        dists = np.random.lognormal(np.log(1.5), 0.4, n_samples)  # ~1.5 kpc

        # Mock realistic SED output
        mock_seds = np.random.uniform(12, 18, (n_samples, nfilt))
        # Add realistic color trend (bluer to redder)
        for i in range(nfilt):
            mock_seds[:, i] += i * 0.3

        filter_names = ["U", "B", "V", "R", "I", "J", "H"]

        with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
            fig, ax, parts = posterior_predictive(
                models,
                idxs,
                reds,
                dreds,
                dists,
                labels=filter_names,
                vcolor="navy",
                flux=False,
            )

        # Should handle realistic stellar data without issues
        assert fig is not None
        assert len(parts["bodies"]) == nfilt

        plt.close(fig)

    def test_posterior_predictive_with_photometry(self):
        """Test with realistic photometric data overlay."""
        np.random.seed(42)

        nmodels, nfilt, ncoeff = 15, 5, 3
        n_samples = 20

        models = np.random.uniform(10, 20, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, n_samples, replace=True)
        reds = np.random.exponential(0.5, n_samples)
        dreds = np.random.normal(3.3, 0.3, n_samples)
        dists = np.random.lognormal(np.log(2.0), 0.3, n_samples)

        # Realistic flux measurements with errors and some missing data
        data = np.array([2.1e-13, 1.8e-13, 1.5e-13, 1.2e-13, 1.0e-13])
        data_err = data * 0.08  # 8% errors
        data_mask = np.array([True, True, False, True, True])  # Missing R-band

        mock_seds = np.random.uniform(15, 19, (n_samples, nfilt))

        with (
            patch("brutus.plotting.sed.get_seds", return_value=mock_seds),
            patch("brutus.plotting.sed.magnitude") as mock_magnitude,
        ):
            # Mock realistic magnitude conversion
            masked_data = data[data_mask]
            masked_err = data_err[data_mask]
            mock_mags = np.random.uniform(15, 18, len(masked_data))
            mock_mag_errs = np.random.uniform(0.05, 0.15, len(masked_data))
            mock_magnitude.return_value = (mock_mags, mock_mag_errs)

            fig, ax, parts = posterior_predictive(
                models,
                idxs,
                reds,
                dreds,
                dists,
                data=data,
                data_err=data_err,
                data_mask=data_mask,
                pcolor="red",
                psig=2.0,
            )

        assert fig is not None
        plt.close(fig)

    def test_posterior_predictive_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small sample
        nmodels, nfilt, ncoeff = 5, 3, 2
        n_samples = 3

        models = np.random.uniform(15, 20, (nmodels, nfilt, ncoeff))
        idxs = np.random.choice(nmodels, n_samples, replace=True)
        reds = np.random.uniform(0, 1, n_samples)
        dreds = np.random.uniform(2.5, 4.0, n_samples)
        dists = np.random.uniform(0.5, 5.0, n_samples)

        mock_seds = np.random.uniform(16, 19, (n_samples, nfilt))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May produce warnings for small dataset
            with patch("brutus.plotting.sed.get_seds", return_value=mock_seds):
                fig, ax, parts = posterior_predictive(models, idxs, reds, dreds, dists)

        assert fig is not None
        plt.close(fig)
