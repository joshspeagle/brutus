#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for line-of-sight dust extinction analysis module.

This module tests the functionality of los_dust.py using synthetic
stellar posterior samples to ensure correct behavior of the multi-cloud
extinction model.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from brutus.analysis.los_dust import (
    kernel_gauss,
    kernel_lorentz,
    kernel_tophat,
    los_clouds_loglike_samples,
    los_clouds_priortransform,
)


class TestKernelFunctions:
    """Test the various kernel functions used for LOS dust modeling."""

    def test_kernel_tophat_basic(self):
        """Test basic top-hat kernel functionality."""
        reds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        kp = (0.5, 0.2)  # mean=0.5, half-width=0.2, so range [0.3, 0.7]

        logw = kernel_tophat(reds, kp)

        # Check shape
        assert logw.shape == reds.shape

        # Samples within range should have higher weights than those outside
        # For top-hat, within range should have weight 1/(2*width) = 1/0.4 = 2.5
        # Outside range should have weight 0 (log = -inf)
        expected_weight = 1.0 / (2 * 0.2)  # 2.5
        expected_log_weight = np.log(expected_weight)

        # Samples at 0.3, 0.5 should be included (note: >= klow, < khigh)
        assert np.allclose(logw[1:3], expected_log_weight)  # 0.3, 0.5

        # Samples at 0.1, 0.7, 0.9 should be outside (0.7 is boundary case)
        assert logw[0] == -np.inf  # 0.1 < 0.3
        assert logw[3] == -np.inf  # 0.7 >= 0.7 (not < 0.7)
        assert logw[4] == -np.inf  # 0.9 > 0.7

    def test_kernel_tophat_edge_cases(self):
        """Test top-hat kernel edge cases."""
        reds = np.array([1.0, 2.0])

        # Test zero width - should raise error
        with pytest.raises(ValueError, match="Kernel width must be positive"):
            kernel_tophat(reds, (1.5, 0))

        # Test negative width - should raise error
        with pytest.raises(ValueError, match="Kernel width must be positive"):
            kernel_tophat(reds, (1.5, -0.1))

    def test_kernel_gauss_basic(self):
        """Test basic Gaussian kernel functionality."""
        reds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        kp = (0.5, 0.1)  # mean=0.5, std=0.1

        logw = kernel_gauss(reds, kp)

        # Check shape
        assert logw.shape == reds.shape

        # All weights should be finite
        assert np.all(np.isfinite(logw))

        # Sample at mean should have highest weight
        max_idx = np.argmax(logw)
        assert max_idx == 2  # Index of 0.5

        # Weights should be symmetric around mean
        assert np.allclose(logw[1], logw[3])  # 0.4 and 0.6, equidistant from 0.5
        assert np.allclose(logw[0], logw[4])  # 0.3 and 0.7, equidistant from 0.5

        # Further samples should have lower weights
        assert logw[2] > logw[1] > logw[0]  # 0.5 > 0.4 > 0.3

    def test_kernel_gauss_mathematical_accuracy(self):
        """Test Gaussian kernel mathematical accuracy."""
        reds = np.array([0.5])  # Single point at mean
        kp = (0.5, 0.2)  # mean=0.5, std=0.2

        logw = kernel_gauss(reds, kp)

        # At mean, log-weight should be -log(sqrt(2*pi)*std)
        expected_logw = -np.log(np.sqrt(2 * np.pi) * 0.2)
        assert np.allclose(logw[0], expected_logw)

    def test_kernel_gauss_edge_cases(self):
        """Test Gaussian kernel edge cases."""
        reds = np.array([1.0, 2.0])

        # Test zero std - should raise error
        with pytest.raises(
            ValueError, match="Kernel standard deviation must be positive"
        ):
            kernel_gauss(reds, (1.5, 0))

        # Test negative std - should raise error
        with pytest.raises(
            ValueError, match="Kernel standard deviation must be positive"
        ):
            kernel_gauss(reds, (1.5, -0.1))

    def test_kernel_lorentz_basic(self):
        """Test basic Lorentzian kernel functionality."""
        reds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        kp = (0.5, 0.1)  # mean=0.5, HWHM=0.1

        logw = kernel_lorentz(reds, kp)

        # Check shape
        assert logw.shape == reds.shape

        # All weights should be finite
        assert np.all(np.isfinite(logw))

        # Sample at mean should have highest weight
        max_idx = np.argmax(logw)
        assert max_idx == 2  # Index of 0.5

        # Weights should be symmetric around mean
        assert np.allclose(logw[1], logw[3])  # 0.4 and 0.6
        assert np.allclose(logw[0], logw[4])  # 0.3 and 0.7

    def test_kernel_lorentz_mathematical_accuracy(self):
        """Test Lorentzian kernel mathematical accuracy."""
        reds = np.array([0.5])  # Single point at mean
        kp = (0.5, 0.2)  # mean=0.5, HWHM=0.2

        logw = kernel_lorentz(reds, kp)

        # At mean, log-weight should be -log(pi*HWHM)
        expected_logw = -np.log(np.pi * 0.2)
        assert np.allclose(logw[0], expected_logw)

    def test_kernel_lorentz_edge_cases(self):
        """Test Lorentzian kernel edge cases."""
        reds = np.array([1.0, 2.0])

        # Test zero HWHM - should raise error
        with pytest.raises(ValueError, match="Kernel HWHM must be positive"):
            kernel_lorentz(reds, (1.5, 0))

        # Test negative HWHM - should raise error
        with pytest.raises(ValueError, match="Kernel HWHM must be positive"):
            kernel_lorentz(reds, (1.5, -0.1))

    def test_kernel_per_object_mean(self):
        """Per-object cloud means must not collapse to object 0's mean.

        Regression: the kernels previously did ``kmean = kmean.flat[0]``, so a
        per-object mean array (as produced by a reddening template) wrongly
        applied object 0's mean to every object -- giving wrong (often -inf
        for the tophat) weights for objects 1..N-1.
        """
        nobj, nsamps = 3, 4
        kmean = np.array([[0.5] * nsamps, [1.5] * nsamps, [2.5] * nsamps])
        reds = kmean.copy()  # each object sits exactly on its own mean
        kstd = np.full((nobj, nsamps), 0.1)

        lg = kernel_gauss(reds, (kmean, kstd))
        assert np.all(np.isfinite(lg))
        # Every object is at its own peak -> all rows identical.
        assert np.allclose(lg[0], lg[1]) and np.allclose(lg[0], lg[2])

        lt = kernel_tophat(reds, (kmean, np.full((nobj, nsamps), 0.2)))
        assert np.all(np.isfinite(lt))  # rows 1, 2 were -inf before the fix

        ll = kernel_lorentz(reds, (kmean, kstd))
        assert np.allclose(ll[0], ll[1]) and np.allclose(ll[0], ll[2])


class TestPriorTransform:
    """Test the prior transform function for nested sampling."""

    def test_prior_transform_basic_1cloud(self):
        """Test basic prior transform for 1-cloud model."""
        # 1-cloud model: pb, s0, s, fred, dist1, red1 (6 parameters)
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])

        x = los_clouds_priortransform(u)

        # Check shape preserved
        assert x.shape == u.shape
        assert len(x) == 6

        # Check that some parameters are transformed (not equal to input)
        assert not np.allclose(x, u)  # Should be transformed

        # Check distance ordering (monotonic)
        # For 1 cloud, only one distance at index 4
        assert x[4] >= 4.0  # Within dlims
        assert x[4] <= 19.0

        # Check reddening bounds
        assert x[3] >= 0.0  # fred within rlims
        assert x[3] <= 6.0
        assert x[5] >= 0.0  # red1 within rlims
        assert x[5] <= 6.0

    def test_prior_transform_2cloud(self):
        """Test prior transform for 2-cloud model."""
        # 2-cloud model: pb, s0, s, fred, dist1, red1, dist2, red2 (8 parameters)
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        x = los_clouds_priortransform(u)

        # Check shape
        assert x.shape == u.shape
        assert len(x) == 8

        # Check distance ordering (should be monotonically increasing)
        dist1, dist2 = x[4], x[6]
        assert dist1 <= dist2  # Distances should be sorted

        # Check all within bounds
        assert 4.0 <= dist1 <= 19.0
        assert 4.0 <= dist2 <= 19.0
        assert 0.0 <= x[3] <= 6.0  # fred
        assert 0.0 <= x[5] <= 6.0  # red1
        assert 0.0 <= x[7] <= 6.0  # red2

    def test_prior_transform_dust_template(self):
        """Test prior transform with dust template option."""
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])

        x = los_clouds_priortransform(u, dust_template=True, nlims=(0.5, 1.5))

        # When dust_template=True, cloud reddenings become rescaling factors
        # Should be within nlims instead of rlims
        assert 0.5 <= x[5] <= 1.5  # red1 should be rescaling factor

    def test_prior_transform_custom_limits(self):
        """Test prior transform with custom limits."""
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])

        # Custom limits
        rlims_custom = (0.1, 2.0)
        dlims_custom = (5.0, 15.0)

        x = los_clouds_priortransform(u, rlims=rlims_custom, dlims=dlims_custom)

        # Check custom bounds are respected
        assert dlims_custom[0] <= x[4] <= dlims_custom[1]  # distance
        assert rlims_custom[0] <= x[3] <= rlims_custom[1]  # fred
        assert rlims_custom[0] <= x[5] <= rlims_custom[1]  # red1

    def test_prior_transform_input_validation(self):
        """Test input validation for prior transform."""
        # Test invalid u dimensions
        u_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="Input u must be a 1D array"):
            los_clouds_priortransform(u_2d)

        # Test u values outside [0,1]
        u_invalid = np.array([0.1, 0.2, 1.5, 0.4, 0.6, 0.8])
        with pytest.raises(ValueError, match="All values in u must be between 0 and 1"):
            los_clouds_priortransform(u_invalid)

        # Test invalid rlims
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
        with pytest.raises(ValueError, match="rlims must be a 2-tuple"):
            los_clouds_priortransform(u, rlims=(6.0, 0.0))  # reversed

        # Test invalid dlims
        with pytest.raises(ValueError, match="dlims must be a 2-tuple"):
            los_clouds_priortransform(u, dlims=(19.0, 4.0))  # reversed

        # Test invalid nlims for dust template
        with pytest.raises(ValueError, match="nlims must be a 2-tuple"):
            los_clouds_priortransform(u, dust_template=True, nlims=(2.0, 0.2))


class TestLogLikelihood:
    """Test the log-likelihood function for LOS dust modeling."""

    def setup_method(self):
        """Set up synthetic data for testing."""
        # Create synthetic stellar samples
        np.random.seed(42)  # For reproducible tests
        self.nstars = 20
        self.nsamps = 100

        # Distance samples (distance modulus, uniform in range)
        self.dsamps = np.random.uniform(6, 12, (self.nstars, self.nsamps))

        # Reddening samples (A_V extinction, with some structure)
        # Create a simple extinction profile: low at near distances, higher further
        base_extinction = 0.1 + 0.3 * (self.dsamps - 6) / 6  # Linear increase
        noise = np.random.normal(0, 0.05, self.dsamps.shape)
        self.rsamps = np.maximum(0, base_extinction + noise)

        # 1-cloud model parameters: [pb, s0, s, fred, dist1, red1]
        self.theta_1cloud = [0.1, 0.05, 0.05, 0.2, 8.0, 0.5]

        # 2-cloud model parameters: [pb, s0, s, fred, dist1, red1, dist2, red2]
        self.theta_2cloud = [0.1, 0.05, 0.05, 0.2, 7.0, 0.3, 10.0, 0.8]

    def test_loglike_basic_functionality(self):
        """Test basic log-likelihood computation."""
        loglike = los_clouds_loglike_samples(
            self.theta_1cloud, self.dsamps, self.rsamps
        )

        # Should return a finite scalar
        assert isinstance(loglike, (float, np.floating))
        assert np.isfinite(loglike)

        # For reasonable parameters and data, should not be extremely negative
        assert loglike > -1e6

    def test_loglike_different_kernels(self):
        """Test log-likelihood with different kernel functions."""
        kernels = ["gauss", "tophat", "lorentz"]
        loglikes = []

        for kernel in kernels:
            loglike = los_clouds_loglike_samples(
                self.theta_1cloud, self.dsamps, self.rsamps, kernel=kernel
            )
            loglikes.append(loglike)
            assert np.isfinite(loglike)

        # Different kernels should generally give different results
        assert not np.allclose(loglikes[0], loglikes[1])
        assert not np.allclose(loglikes[1], loglikes[2])

    def test_loglike_2cloud_model(self):
        """Test log-likelihood with 2-cloud model."""
        loglike = los_clouds_loglike_samples(
            self.theta_2cloud, self.dsamps, self.rsamps
        )

        assert isinstance(loglike, (float, np.floating))
        assert np.isfinite(loglike)

    def test_loglike_monotonic_constraint(self):
        """Test monotonic constraint enforcement."""
        # Create non-monotonic parameters (red2 < red1)
        theta_nonmono = [0.1, 0.05, 0.05, 0.2, 7.0, 0.8, 10.0, 0.3]

        # With monotonic=True (default), should return -inf
        loglike_mono = los_clouds_loglike_samples(
            theta_nonmono, self.dsamps, self.rsamps, monotonic=True
        )
        assert loglike_mono == -np.inf

        # With monotonic=False, should return finite value
        loglike_no_mono = los_clouds_loglike_samples(
            theta_nonmono, self.dsamps, self.rsamps, monotonic=False
        )
        assert np.isfinite(loglike_no_mono)

    def test_loglike_template_reds(self):
        """Test log-likelihood with template reddenings."""
        # Create template reddenings for each star
        template_reds = np.random.uniform(0.8, 1.2, self.nstars)

        loglike = los_clouds_loglike_samples(
            self.theta_1cloud, self.dsamps, self.rsamps, template_reds=template_reds
        )

        assert np.isfinite(loglike)

    def test_loglike_template_reds_per_object_means(self):
        """With per-object template reddenings, the joint log-likelihood must
        equal the sum of the per-object log-likelihoods.

        Regression for the kernel mean-collapse: every object except object 0
        was previously evaluated against object 0's cloud mean, so the joint
        likelihood did not decompose object-by-object.
        """
        nobj, nsamps = 3, 100
        dsamps = np.full((nobj, nsamps), 12.0)
        template_reds = np.array([0.5, 1.5, 2.5])
        red_fg, red_bg = 0.05, 1.0
        rsamps = np.empty((nobj, nsamps))
        for i in range(nobj):
            rsamps[i] = red_bg * template_reds[i]  # each obj at its own bg mean
        theta = [0.0, 0.05, 0.05, red_fg, 8.0, red_bg]

        full = los_clouds_loglike_samples(
            theta,
            dsamps,
            rsamps,
            kernel="gauss",
            template_reds=template_reds,
            Ndraws=nsamps,
            monotonic=False,
        )
        per_obj = sum(
            los_clouds_loglike_samples(
                theta,
                dsamps[i : i + 1],
                rsamps[i : i + 1],
                kernel="gauss",
                template_reds=template_reds[i : i + 1],
                Ndraws=nsamps,
                monotonic=False,
            )
            for i in range(nobj)
        )
        assert np.isclose(full, per_obj, atol=1e-6)

    def test_loglike_additive_foreground(self):
        """Test log-likelihood with additive foreground."""
        loglike_additive = los_clouds_loglike_samples(
            self.theta_2cloud, self.dsamps, self.rsamps, additive_foreground=True
        )

        loglike_normal = los_clouds_loglike_samples(
            self.theta_2cloud, self.dsamps, self.rsamps, additive_foreground=False
        )

        # Should give different results
        assert loglike_additive != loglike_normal
        assert np.isfinite(loglike_additive)
        assert np.isfinite(loglike_normal)

    def test_loglike_ndraws_parameter(self):
        """Test log-likelihood with different Ndraws values."""
        # Test with different numbers of samples
        for ndraws in [10, 25, 50]:
            loglike = los_clouds_loglike_samples(
                self.theta_1cloud, self.dsamps, self.rsamps, Ndraws=ndraws
            )
            assert np.isfinite(loglike)

        # Test with Ndraws larger than available samples
        loglike_large = los_clouds_loglike_samples(
            self.theta_1cloud, self.dsamps, self.rsamps, Ndraws=200
        )
        assert np.isfinite(loglike_large)

    def test_loglike_custom_kernel(self):
        """Test log-likelihood with custom kernel function."""

        def custom_kernel(reds, kp):
            # Simple uniform kernel for testing
            kmean, kwidth = kp[0], kp[1]
            return np.zeros_like(reds) - np.log(2 * kwidth)

        loglike = los_clouds_loglike_samples(
            self.theta_1cloud, self.dsamps, self.rsamps, kernel=custom_kernel
        )

        assert np.isfinite(loglike)

    def test_loglike_input_validation(self):
        """Test input validation for log-likelihood function."""
        # Test invalid theta dimensions
        with pytest.raises(ValueError, match="theta must be a 1D array"):
            los_clouds_loglike_samples(
                np.array([[1, 2], [3, 4]]), self.dsamps, self.rsamps
            )

        # Test insufficient theta parameters
        with pytest.raises(ValueError, match="theta must have at least 6 parameters"):
            los_clouds_loglike_samples([0.1, 0.05], self.dsamps, self.rsamps)

        # Test odd number of cloud parameters
        theta_odd = [0.1, 0.05, 0.05, 0.2, 8.0]  # Missing one parameter
        with pytest.raises(ValueError, match="theta must have at least 6 parameters"):
            los_clouds_loglike_samples(theta_odd, self.dsamps, self.rsamps)

        # Test mismatched sample dimensions
        dsamps_wrong = self.dsamps[:, :50]  # Different number of samples
        with pytest.raises(
            ValueError, match="dsamps and rsamps must have the same shape"
        ):
            los_clouds_loglike_samples(self.theta_1cloud, dsamps_wrong, self.rsamps)

        # Test invalid kernel
        with pytest.raises(ValueError, match="The kernel 'invalid' is not valid"):
            los_clouds_loglike_samples(
                self.theta_1cloud, self.dsamps, self.rsamps, kernel="invalid"
            )

        # Test invalid parameter values
        theta_invalid_pb = [1.5, 0.05, 0.05, 0.2, 8.0, 0.5]  # pb > 1
        with pytest.raises(ValueError, match="Outlier fraction pb must be in"):
            los_clouds_loglike_samples(theta_invalid_pb, self.dsamps, self.rsamps)

        theta_invalid_s = [0.1, -0.1, 0.05, 0.2, 8.0, 0.5]  # negative smoothing
        with pytest.raises(
            ValueError, match="Smoothing parameters must be non-negative"
        ):
            los_clouds_loglike_samples(theta_invalid_s, self.dsamps, self.rsamps)

        # Test non-monotonic distances
        theta_bad_dist = [0.1, 0.05, 0.05, 0.2, 10.0, 0.3, 8.0, 0.5]  # dist2 < dist1
        with pytest.raises(
            ValueError, match="Distances must be monotonically increasing"
        ):
            los_clouds_loglike_samples(theta_bad_dist, self.dsamps, self.rsamps)

        # Test invalid template_reds shape
        template_wrong_shape = np.random.uniform(0.8, 1.2, self.nstars + 5)
        with pytest.raises(ValueError, match="template_reds must have shape"):
            los_clouds_loglike_samples(
                self.theta_1cloud,
                self.dsamps,
                self.rsamps,
                template_reds=template_wrong_shape,
            )

    def test_loglike_edge_cases(self):
        """Test log-likelihood edge cases."""
        # Test with very small samples
        small_dsamps = self.dsamps[:2, :5]  # 2 stars, 5 samples each
        small_rsamps = self.rsamps[:2, :5]

        loglike = los_clouds_loglike_samples(
            self.theta_1cloud, small_dsamps, small_rsamps
        )
        assert np.isfinite(loglike)

        # Test with zero extinction everywhere
        zero_rsamps = np.zeros_like(small_rsamps)
        loglike_zero = los_clouds_loglike_samples(
            self.theta_1cloud, small_dsamps, zero_rsamps
        )
        assert np.isfinite(loglike_zero)


class TestIntegration:
    """Integration tests for the complete LOS dust fitting workflow."""

    def test_prior_transform_to_loglike_pipeline(self):
        """Test complete pipeline from prior transform to likelihood."""
        # Generate unit cube samples that will produce monotonic extinctions
        # pb, s0, s, fred, dist1, red1 - ensure fred < red1
        u = np.array([0.1, 0.2, 0.3, 0.2, 0.6, 0.8])  # fred will be smaller than red1

        # Transform to physical parameters
        theta = los_clouds_priortransform(u)

        # Generate synthetic data
        nstars, nsamps = 15, 50
        np.random.seed(42)
        dsamps = np.random.uniform(6, 12, (nstars, nsamps))
        rsamps = np.random.uniform(0, 1, (nstars, nsamps))

        # Compute likelihood with monotonic constraint
        loglike = los_clouds_loglike_samples(theta, dsamps, rsamps, monotonic=True)

        assert np.isfinite(loglike)
        assert isinstance(loglike, (float, np.floating))

        # Also test that non-monotonic configurations are properly rejected
        u_nonmono = np.array(
            [0.1, 0.2, 0.3, 0.8, 0.6, 0.2]
        )  # fred > red1 (non-monotonic)
        theta_nonmono = los_clouds_priortransform(u_nonmono)
        loglike_nonmono = los_clouds_loglike_samples(
            theta_nonmono, dsamps, rsamps, monotonic=True
        )

        # Non-monotonic should return -inf when monotonic=True
        assert loglike_nonmono == -np.inf

        # But should work when monotonic=False
        loglike_nonmono_allowed = los_clouds_loglike_samples(
            theta_nonmono, dsamps, rsamps, monotonic=False
        )
        assert np.isfinite(loglike_nonmono_allowed)

    def test_multiple_cloud_configurations(self):
        """Test different numbers of clouds in the model."""
        np.random.seed(456)

        # Generate synthetic data
        nstars, nsamps = 10, 30
        dsamps = np.random.uniform(7, 11, (nstars, nsamps))
        rsamps = np.random.uniform(0, 0.8, (nstars, nsamps))

        # Test 1, 2, and 3 cloud models
        loglikes = []
        for nclouds in [1, 2, 3]:
            ndim = 2 * nclouds + 4
            # Create unit cube values that will produce monotonic extinctions
            u = np.random.uniform(0, 1, ndim)
            # Ensure monotonic extinction by using sorted values for extinction parameters
            extinction_indices = list(
                range(3, ndim, 2)
            )  # indices of extinction parameters
            u_sorted_extinctions = np.sort(u[extinction_indices])
            for i, idx in enumerate(extinction_indices):
                u[idx] = u_sorted_extinctions[i]

            theta = los_clouds_priortransform(u)

            # Use monotonic=False to be more forgiving
            loglike = los_clouds_loglike_samples(theta, dsamps, rsamps, monotonic=False)
            loglikes.append(loglike)
            assert np.isfinite(loglike)

        # All should produce finite likelihoods
        assert all(np.isfinite(ll) for ll in loglikes)

    def test_realistic_extinction_profile(self):
        """Test with a realistic extinction profile."""
        np.random.seed(789)

        # Create realistic extinction profile
        nstars = 25
        nsamps = 75

        # Distance samples with realistic clustering
        dsamps = np.random.normal(8.5, 1.0, (nstars, nsamps))  # Centered around 8.5 DM
        dsamps = np.clip(dsamps, 6.0, 12.0)  # Clip to reasonable range

        # Extinction that increases with distance, with scatter
        mean_extinction = 0.05 + 0.15 * (dsamps - 6.0) / 6.0  # Linear increase
        rsamps = np.random.lognormal(np.log(mean_extinction + 0.01), 0.3)
        rsamps = np.clip(rsamps, 0.001, 3.0)  # Reasonable extinction range

        # 2-cloud model with monotonically increasing extinctions
        u = np.array(
            [0.2, 0.3, 0.25, 0.4, 0.35, 0.2, 0.8, 0.7]
        )  # 2-cloud model with fred < red1 < red2
        theta = los_clouds_priortransform(u)

        # Test with different kernels using monotonic=False for robustness
        for kernel in ["gauss", "tophat", "lorentz"]:
            loglike = los_clouds_loglike_samples(
                theta, dsamps, rsamps, kernel=kernel, monotonic=False
            )
            assert np.isfinite(loglike)
            # Should not be extremely negative for reasonable data
            assert loglike > -1e5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
