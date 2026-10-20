#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for distance vs reddening plotting functions.

This test suite verifies that the refactored dist_vs_red function
works correctly and maintains backward compatibility.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.plotting.distance import dist_vs_red


class TestDistVsRed:
    """Tests for dist_vs_red function."""

    @pytest.fixture
    def direct_test_data(self):
        """Test data in direct format: (dists, reds, dreds)."""
        rng = np.random.RandomState(42)
        n_samples = 50

        # Generate realistic distance/reddening samples for single object
        dists = rng.lognormal(np.log(2.0), 0.3, n_samples)  # ~2 kpc
        reds = rng.exponential(0.5, n_samples)  # Av ~ 0.5
        dreds = rng.normal(3.3, 0.3, n_samples)  # Rv ~ 3.3

        return (dists, reds, dreds)

    def test_dist_vs_red_basic(self, direct_test_data):
        """Test basic distance vs reddening plot generation."""
        # Create a figure
        plt.figure(figsize=(6, 6))

        # Generate the plot
        H, xedges, yedges, img = dist_vs_red(
            direct_test_data, bins=(20, 15)  # Small bins for speed
        )

        # Check return values
        assert H.shape == (20, 15)
        assert len(xedges) == 21  # bins + 1
        assert len(yedges) == 16  # bins + 1
        assert img is not None  # matplotlib image object

        # Check that histogram has reasonable values
        assert np.all(H >= 0)
        assert np.sum(H) > 0

        plt.close()

    def test_dist_vs_red_uses_binning(self, direct_test_data):
        """Test that dist_vs_red uses bin_pdfs_distred internally."""
        # This is an integration test - we verify behavior is consistent
        # with bin_pdfs_distred by checking that both produce the same results

        from brutus.plotting.binning import bin_pdfs_distred

        # Convert single object data to multi-object format for bin_pdfs_distred
        multi_obj_data = tuple(arr[None, :] for arr in direct_test_data)

        # Get binned data directly
        binned_vals, xedges_bin, yedges_bin = bin_pdfs_distred(
            multi_obj_data, bins=(15, 10)
        )

        # Get the same data via dist_vs_red
        plt.figure(figsize=(6, 6))
        H, xedges_plot, yedges_plot, img = dist_vs_red(direct_test_data, bins=(15, 10))
        plt.close()

        # Should produce the same binning (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(H, binned_vals[0], decimal=4)
        np.testing.assert_array_almost_equal(xedges_plot, xedges_bin, decimal=10)
        np.testing.assert_array_almost_equal(yedges_plot, yedges_bin, decimal=10)

    def test_dist_vs_red_distance_modulus(self, direct_test_data):
        """Test distance_modulus representation."""
        plt.figure(figsize=(6, 6))

        H, xedges, yedges, img = dist_vs_red(
            direct_test_data, dist_type="distance_modulus", bins=(10, 8)
        )

        # Distance modulus should be reasonable range
        assert xedges[0] > 0  # Should be positive
        assert xedges[-1] > xedges[0]  # Should be increasing

        plt.close()

    def test_dist_vs_red_parallax_mode(self, direct_test_data):
        """Test parallax space plotting."""
        plt.figure(figsize=(6, 6))

        H, xedges, yedges, img = dist_vs_red(
            direct_test_data, dist_type="parallax", bins=(10, 8)
        )

        # Parallax should be positive
        assert xedges[0] >= 0
        assert np.all(np.isfinite(xedges))

        plt.close()

    def test_dist_vs_red_ebv_conversion(self, direct_test_data):
        """Test E(B-V) vs Av conversion."""
        plt.figure(figsize=(6, 6))

        H, xedges, yedges, img = dist_vs_red(direct_test_data, ebv=True, bins=(10, 8))

        # Should work without errors
        assert H.shape == (10, 8)
        assert np.all(H >= 0)

        plt.close()

    def test_dist_vs_red_with_truth_values(self, direct_test_data):
        """Test plotting with truth value overlays."""
        plt.figure(figsize=(6, 6))

        # Plot with truth values
        H, xedges, yedges, img = dist_vs_red(
            direct_test_data,
            truths=[12.0, 0.5],  # [distance_modulus, Av]
            truth_color="red",
            bins=(10, 8),
        )

        # Should complete without errors
        assert H.shape == (10, 8)

        plt.close()

    def test_dist_vs_red_error_handling(self):
        """Test error conditions."""
        # Test with invalid distance type
        dummy_data = (np.ones(10), np.ones(10), np.ones(10))

        plt.figure(figsize=(6, 6))

        with pytest.raises(ValueError):
            dist_vs_red(dummy_data, dist_type="invalid")

        plt.close()

    def test_dist_vs_red_custom_colormap(self, direct_test_data):
        """Test custom colormap and plot options."""
        plt.figure(figsize=(6, 6))

        H, xedges, yedges, img = dist_vs_red(
            direct_test_data, cmap="viridis", plot_kwargs={"alpha": 0.8}, bins=(8, 6)
        )

        # Should complete successfully
        assert H.shape == (8, 6)

        plt.close()

    @pytest.mark.parametrize(
        "dist_type", ["distance_modulus", "parallax", "scale", "distance"]
    )
    def test_dist_vs_red_all_distance_types(self, direct_test_data, dist_type):
        """Test all supported distance representations."""
        plt.figure(figsize=(6, 6))

        H, xedges, yedges, img = dist_vs_red(
            direct_test_data, dist_type=dist_type, bins=(8, 6)
        )

        assert H.shape == (8, 6)
        assert np.all(H >= 0)
        assert np.all(np.isfinite(xedges))
        assert np.all(np.isfinite(yedges))

        plt.close()


class TestDistVsRedBackwardCompatibility:
    """Test backward compatibility with original dist_vs_red."""

    def test_signature_compatibility(self):
        """Test that function signature matches original."""
        import inspect

        sig = inspect.signature(dist_vs_red)
        param_names = list(sig.parameters.keys())

        # Should have all the expected parameters
        expected_params = [
            "data",
            "ebv",
            "dist_type",
            "lndistprior",
            "coord",
            "avlim",
            "rvlim",
            "weights",
            "parallax",
            "parallax_err",
            "Nr",
            "cmap",
            "bins",
            "span",
            "smooth",
            "plot_kwargs",
            "truths",
            "truth_color",
            "truth_kwargs",
            "rstate",
        ]

        for param in expected_params:
            assert param in param_names

    def test_return_format_compatibility(self):
        """Test that return format matches original."""
        plt.figure(figsize=(6, 6))

        # Create simple test data
        rng = np.random.RandomState(42)
        n_samples = 20
        test_data = (
            rng.lognormal(np.log(2.0), 0.3, n_samples),  # distances
            rng.exponential(0.5, n_samples),  # Av
            rng.normal(3.3, 0.3, n_samples),  # Rv
        )

        result = dist_vs_red(test_data, bins=(5, 4))

        # Should return 4-tuple: (H, xedges, yedges, img)
        assert len(result) == 4
        H, xedges, yedges, img = result

        assert isinstance(H, np.ndarray)
        assert isinstance(xedges, np.ndarray)
        assert isinstance(yedges, np.ndarray)
        # img should be a matplotlib image object

        plt.close()

    def test_dist_vs_red_with_sar_data_format(self):
        """Test with SAR (4-tuple) data format."""
        rng = np.random.RandomState(42)
        n_samples = 30

        # Create SAR format data: (scales, avs, rvs, covs_sar)
        scales = rng.exponential(1.0, n_samples)
        avs = rng.exponential(0.5, n_samples)
        rvs = rng.normal(3.3, 0.3, n_samples)

        # Create proper positive semidefinite covariance matrices
        covs_sar = np.zeros((n_samples, 3, 3))
        for i in range(n_samples):
            # Create a random positive semidefinite matrix
            A = rng.randn(3, 3)
            covs_sar[i] = np.dot(A, A.T) * 0.01  # Small covariance

        sar_data = (scales, avs, rvs, covs_sar)

        plt.figure(figsize=(6, 6))

        # This should trigger the 4-tuple branch (lines 157-163)
        H, xedges, yedges, img = dist_vs_red(
            sar_data,
            coord=(45.0, 30.0),  # This should trigger coord conversion (line 168)
            bins=(6, 5),
        )

        assert H.shape == (6, 5)

        plt.close()

    def test_dist_vs_red_with_parallax_info(self):
        """Test with parallax and parallax_err provided."""
        rng = np.random.RandomState(42)
        n_samples = 30

        # Generate test data
        dists = rng.lognormal(np.log(2.0), 0.3, n_samples)
        reds = rng.exponential(0.5, n_samples)
        dreds = rng.normal(3.3, 0.3, n_samples)
        test_data = (dists, reds, dreds)

        plt.figure(figsize=(6, 6))

        # This should trigger parallax conversion lines (172, 174)
        H, xedges, yedges, img = dist_vs_red(
            test_data,
            parallax=2.5,  # Should trigger line 172
            parallax_err=0.1,  # Should trigger line 174
            bins=(8, 6),
        )

        assert H.shape == (8, 6)

        plt.close()

    def test_dist_vs_red_multi_object_case(self):
        """Test with multi-object data format."""
        rng = np.random.RandomState(42)
        n_objects = 2
        n_samples = 20

        # Multi-object data: shape (n_objects, n_samples)
        dists = rng.lognormal(np.log(2.0), 0.3, (n_objects, n_samples))
        reds = rng.exponential(0.5, (n_objects, n_samples))
        dreds = rng.normal(3.3, 0.3, (n_objects, n_samples))

        multi_data = (dists, reds, dreds)

        plt.figure(figsize=(6, 6))

        # This should trigger multi-object case (line 177, 205)
        H, xedges, yedges, img = dist_vs_red(multi_data, bins=(6, 5))

        # Should use first object's data (line 205)
        assert H.shape == (6, 5)

        plt.close()


def test_scalar_parallax_with_several_objects_matches_an_array():
    """Regression (#177): a scalar parallax raised TypeError for several
    objects."""
    rng = np.random.RandomState(3)
    data = (
        rng.lognormal(np.log(2.0), 0.3, (3, 200)),
        rng.exponential(0.5, (3, 200)),
        rng.normal(3.3, 0.3, (3, 200)),
    )
    kwargs = dict(bins=(30, 20), dist_type="parallax", smooth=(0.2, 0.01))
    H_scalar = dist_vs_red(data, parallax=0.5, parallax_err=0.01, **kwargs)[0]
    H_array = dist_vs_red(
        data, parallax=np.full(3, 0.5), parallax_err=np.full(3, 0.01), **kwargs
    )[0]
    plt.close("all")
    np.testing.assert_array_equal(H_scalar, H_array)


def test_parallax_of_wrong_shape_raises():
    data = tuple(np.ones((3, 10)) for _ in range(3))
    with pytest.raises(ValueError, match="`parallax`"):
        dist_vs_red(data, parallax=[1.0, 1.0], parallax_err=0.1)
