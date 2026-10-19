"""Tests of brutus.analysis.los_dust."""

import warnings

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm, truncnorm

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
        rng = np.random.RandomState(42)  # For reproducible tests
        self.nstars = 20
        self.nsamps = 100

        # Distance samples (distance modulus, uniform in range)
        self.dsamps = rng.uniform(6, 12, (self.nstars, self.nsamps))

        # Reddening samples (A_V extinction, with some structure)
        # Create a simple extinction profile: low at near distances, higher further
        base_extinction = 0.1 + 0.3 * (self.dsamps - 6) / 6  # Linear increase
        noise = rng.normal(0, 0.05, self.dsamps.shape)
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
        rng = np.random.RandomState(0)
        # Create template reddenings for each star
        template_reds = rng.uniform(0.8, 1.2, self.nstars)

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
        rng = np.random.RandomState(0)
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
        template_wrong_shape = rng.uniform(0.8, 1.2, self.nstars + 5)
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
        rng = np.random.RandomState(42)
        dsamps = rng.uniform(6, 12, (nstars, nsamps))
        rsamps = rng.uniform(0, 1, (nstars, nsamps))

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
        rng = np.random.RandomState(456)

        # Generate synthetic data
        nstars, nsamps = 10, 30
        dsamps = rng.uniform(7, 11, (nstars, nsamps))
        rsamps = rng.uniform(0, 0.8, (nstars, nsamps))

        # Test 1, 2, and 3 cloud models
        loglikes = []
        for nclouds in [1, 2, 3]:
            ndim = 2 * nclouds + 4
            # Create unit cube values that will produce monotonic extinctions
            u = rng.uniform(0, 1, ndim)
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
        rng = np.random.RandomState(789)

        # Create realistic extinction profile
        nstars = 25
        nsamps = 75

        # Distance samples with realistic clustering
        dsamps = rng.normal(8.5, 1.0, (nstars, nsamps))  # Centered around 8.5 DM
        dsamps = np.clip(dsamps, 6.0, 12.0)  # Clip to reasonable range

        # Extinction that increases with distance, with scatter
        mean_extinction = 0.05 + 0.15 * (dsamps - 6.0) / 6.0  # Linear increase
        rsamps = rng.lognormal(np.log(mean_extinction + 0.01), 0.3)
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


KERNELS = {
    "tophat": kernel_tophat,
    "gauss": kernel_gauss,
    "lorentz": kernel_lorentz,
}


def _reference_loglike_core(
    theta,
    dsamps,
    rsamps,
    kernel="gauss",
    rlims=(0.0, 6.0),
    template_reds=None,
    Ndraws=25,
    additive_foreground=False,
):
    """
    Pre-rewrite per-segment implementation of ``los_clouds_loglike_samples``.

    This is a verbatim copy of the old core (per-segment ``reds_expanded``
    list comprehension, full kernel evaluation over every segment with
    ``log(bool)`` masking, ``logsumexp`` over the (segment, draw) axes),
    with only the two separate, deliberate behavior fixes applied
    identically to both implementations so that this test isolates the
    searchsorted performance rewrite:

    - kernels are called with ``rlims`` (truncation renormalization), and
    - non-finite reddening samples are masked to -inf (NaN fix).

    The monotonicity gate is intentionally omitted (it is exercised by its
    own dedicated tests); callers compare with ``monotonic=False`` or with
    profiles both gates accept.
    """
    theta = np.asarray(theta)
    kern = KERNELS[kernel]

    pb, s0, s = theta[0], theta[1], theta[2]
    reds, dists = np.atleast_1d(theta[3::2]), np.atleast_1d(theta[4::2])
    area = rlims[1] - rlims[0]
    rsmooth = s * area
    rsmooth0 = s0 * area

    # Define cloud edges ("distance bounds")
    xedges = np.concatenate(([0], dists, [1e10]))

    # Sub-sample distance and reddening samples
    if Ndraws > dsamps.shape[1]:
        Ndraws = dsamps.shape[1]
    ds, rs = dsamps[:, :Ndraws], rsamps[:, :Ndraws]
    Nobj, Nsamps = ds.shape

    # Get reddenings to each star in each distance slice (kernel mean)
    reds_expanded = np.array([np.full_like(rs, r) for r in reds])

    if template_reds is not None:
        template_reds = np.asarray(template_reds)
        reds_expanded[1:] *= template_reds[None, :, None]

    if additive_foreground:
        reds_expanded[1:] += reds_expanded[0]

    rsmooth_full = np.full_like(rs, rsmooth)
    rsmooth0_full = np.full_like(rs, rsmooth0)

    kparams = []
    for i, r_exp in enumerate(reds_expanded):
        if i == 0:
            kparams.append((r_exp, rsmooth0_full))
        else:
            kparams.append((r_exp, rsmooth_full))

    finite_rs = np.isfinite(rs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logw = np.array(
            [
                np.where(
                    finite_rs,
                    kern(rs, kp, rlims=rlims) + np.log((ds >= xl) & (ds < xh)),
                    -np.inf,
                )
                for xl, xh, kp in zip(xedges[:-1], xedges[1:], kparams)
            ]
        )

    logls = logsumexp(logw, axis=(0, 2)) - np.log(Nsamps)

    logls = logsumexp(
        a=np.c_[logls, np.full_like(logls, -np.log(area))],
        b=[(1.0 - pb), pb],
        axis=1,
    )

    return np.sum(logls)


def _random_case(rng, nclouds, nobj=37, nsamps=40):
    """Random theta/dsamps/rsamps including out-of-bin and NaN samples."""
    dists = np.sort(rng.uniform(4.0, 19.0, nclouds))
    reds = rng.uniform(0.0, 2.0, nclouds + 1)
    theta = np.empty(4 + 2 * nclouds)
    theta[0] = rng.uniform(0.0, 0.3)  # pb
    theta[1:3] = rng.uniform(0.02, 0.2, 2)  # s0, s
    theta[3] = reds[0]
    theta[4::2] = dists
    theta[5::2] = reds[1:]

    dsamps = rng.uniform(3.0, 20.0, (nobj, nsamps))
    rsamps = np.abs(rng.normal(0.5, 0.5, (nobj, nsamps)))
    # samples outside every distance segment
    dsamps[0, :3] = [-2.0, -0.5, 5e10]
    # non-finite reddening samples
    rsamps[1, 0] = np.nan
    rsamps[2, 1] = np.inf
    return theta, dsamps, rsamps


class TestLoglikeEquivalence:
    """Searchsorted rewrite must reproduce the per-segment reference."""

    @pytest.mark.parametrize("nclouds", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("kernel", ["gauss", "tophat", "lorentz"])
    def test_equivalence_default_mode(self, nclouds, kernel):
        rng = np.random.default_rng(100 + nclouds)
        for trial in range(3):
            theta, dsamps, rsamps = _random_case(rng, nclouds)
            ref = _reference_loglike_core(theta, dsamps, rsamps, kernel=kernel)
            new = los_clouds_loglike_samples(
                theta, dsamps, rsamps, kernel=kernel, monotonic=False
            )
            assert np.allclose(new, ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("nclouds", [1, 3, 5])
    @pytest.mark.parametrize("additive_foreground", [False, True])
    def test_equivalence_template_and_additive(self, nclouds, additive_foreground):
        rng = np.random.default_rng(200 + nclouds)
        for trial in range(3):
            theta, dsamps, rsamps = _random_case(rng, nclouds)
            template = rng.uniform(0.3, 3.0, dsamps.shape[0])
            for template_reds in (None, template):
                ref = _reference_loglike_core(
                    theta,
                    dsamps,
                    rsamps,
                    template_reds=template_reds,
                    additive_foreground=additive_foreground,
                )
                new = los_clouds_loglike_samples(
                    theta,
                    dsamps,
                    rsamps,
                    template_reds=template_reds,
                    additive_foreground=additive_foreground,
                    monotonic=False,
                )
                assert np.allclose(new, ref, rtol=1e-12, atol=1e-12)

    def test_equivalence_monotonic_default_mode(self):
        """Default mode with raw-sorted reds: both gates accept, values match."""
        rng = np.random.default_rng(7)
        theta, dsamps, rsamps = _random_case(rng, 3)
        theta[3::2] = np.sort(theta[3::2])  # raw-monotone profile
        ref = _reference_loglike_core(theta, dsamps, rsamps)
        new = los_clouds_loglike_samples(theta, dsamps, rsamps, monotonic=True)
        assert np.allclose(new, ref, rtol=1e-12, atol=1e-12)

    def test_all_samples_outside_bins(self):
        """Samples in no distance segment carry zero weight (outliers only)."""
        theta = [0.1, 0.05, 0.05, 0.2, 8.0, 0.5]
        dsamps = np.full((4, 10), -1.0)  # every sample below the lower edge
        rsamps = np.full((4, 10), 0.5)
        ref = _reference_loglike_core(theta, dsamps, rsamps)
        new = los_clouds_loglike_samples(theta, dsamps, rsamps, monotonic=False)
        assert np.isfinite(new)  # pb > 0 keeps the outlier mixture finite
        assert np.allclose(new, ref, rtol=1e-12, atol=1e-12)

    def test_ndraws_subsampling_matches(self):
        rng = np.random.default_rng(11)
        theta, dsamps, rsamps = _random_case(rng, 2)
        for ndraws in (5, 25, 10**6):
            ref = _reference_loglike_core(theta, dsamps, rsamps, Ndraws=ndraws)
            new = los_clouds_loglike_samples(
                theta, dsamps, rsamps, Ndraws=ndraws, monotonic=False
            )
            assert np.allclose(new, ref, rtol=1e-12, atol=1e-12)


class TestMonotonicEffectiveProfile:
    """Monotonicity must be judged on the effective extinction profile."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.dsamps = rng.uniform(6, 12, (20, 50))
        self.rsamps = rng.uniform(0, 2, (20, 50))

    def test_additive_foreground_monotone_effective_accepted(self):
        """fred=1.0 with increment 0.3 -> effective 1.0 -> 1.3 is monotone.

        Regression: the raw-theta check saw [1.0, 0.3] unsorted and returned
        -inf even though the effective profile increases with distance.
        """
        theta = [0.05, 0.05, 0.05, 1.0, 9.0, 0.3]
        ll_mono = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, additive_foreground=True, monotonic=True
        )
        ll_free = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, additive_foreground=True, monotonic=False
        )
        assert np.isfinite(ll_mono)
        assert ll_mono == ll_free

    def test_additive_foreground_2cloud_monotone_effective_accepted(self):
        # effective levels [0.5, 0.6, 0.8]: monotone despite raw [0.5, 0.1, 0.3]
        theta = [0.05, 0.05, 0.05, 0.5, 8.0, 0.1, 10.0, 0.3]
        ll = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, additive_foreground=True, monotonic=True
        )
        assert np.isfinite(ll)

    def test_additive_foreground_negative_increment_rejected(self):
        # first increment < 0 puts the first cloud BELOW the foreground
        theta = [0.05, 0.05, 0.05, -0.6, 8.0, -0.4, 10.0, -0.2]
        # raw theta is sorted, but effective [-0.6, -1.0, -0.8] is not monotone
        ll = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, additive_foreground=True, monotonic=True
        )
        assert ll == -np.inf

    def test_template_monotone_rescalings_accepted(self):
        """fred (A_V) must not be compared against dimensionless rescalings.

        Regression: fred=1.5 with rescaling 1.0 and template 3.0 gives the
        monotone effective profile 1.5 -> 3.0 but was rejected as raw
        [1.5, 1.0] is unsorted.
        """
        theta = [0.05, 0.05, 0.05, 1.5, 9.0, 1.0]
        template = np.full(self.dsamps.shape[0], 3.0)
        ll = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, template_reds=template, monotonic=True
        )
        assert np.isfinite(ll)

    def test_template_nonmonotone_rescalings_rejected(self):
        theta = [0.05, 0.05, 0.05, 0.1, 8.0, 1.5, 10.0, 0.8]  # rescalings 1.5 > 0.8
        template = np.full(self.dsamps.shape[0], 2.0)
        ll = los_clouds_loglike_samples(
            theta, self.dsamps, self.rsamps, template_reds=template, monotonic=True
        )
        assert ll == -np.inf

    def test_default_mode_unchanged(self):
        # non-monotone raw values still rejected in the default mode
        theta_bad = [0.1, 0.05, 0.05, 0.2, 7.0, 0.8, 10.0, 0.3]
        assert (
            los_clouds_loglike_samples(
                theta_bad, self.dsamps, self.rsamps, monotonic=True
            )
            == -np.inf
        )
        # fred above the first cloud level still rejected
        theta_bad_fg = [0.1, 0.05, 0.05, 0.9, 7.0, 0.5, 10.0, 0.8]
        assert (
            los_clouds_loglike_samples(
                theta_bad_fg, self.dsamps, self.rsamps, monotonic=True
            )
            == -np.inf
        )
        # monotone raw values accepted
        theta_ok = [0.1, 0.05, 0.05, 0.2, 7.0, 0.3, 10.0, 0.8]
        assert np.isfinite(
            los_clouds_loglike_samples(
                theta_ok, self.dsamps, self.rsamps, monotonic=True
            )
        )


class TestNonFiniteSamples:
    """Non-finite reddening samples must not poison the sightline."""

    def setup_method(self):
        rng = np.random.default_rng(3)
        self.theta = [0.1, 0.05, 0.05, 0.2, 8.0, 0.5]
        self.dsamps = rng.uniform(6, 12, (20, 50))
        self.rsamps = np.abs(rng.normal(0.4, 0.2, (20, 50)))

    def test_nan_reddening_sample_is_masked(self):
        """Regression: one NaN in rsamps made the whole loglike NaN."""
        rsamps = self.rsamps.copy()
        rsamps[0, 0] = np.nan
        ll = los_clouds_loglike_samples(self.theta, self.dsamps, rsamps)
        assert np.isfinite(ll)

    def test_nan_reddening_matches_nan_distance_handling(self):
        """A NaN rsamp is down-weighted exactly like a NaN dsamp."""
        rsamps = self.rsamps.copy()
        rsamps[0, 0] = np.nan
        ll_r = los_clouds_loglike_samples(self.theta, self.dsamps, rsamps)

        dsamps = self.dsamps.copy()
        dsamps[0, 0] = np.nan
        ll_d = los_clouds_loglike_samples(self.theta, dsamps, self.rsamps)

        # both are finite, and dropping the SAME sample by either route
        # must give the identical likelihood (both zero its weight)
        assert np.isfinite(ll_r) and np.isfinite(ll_d)
        assert ll_r == ll_d

    def test_inf_reddening_sample_is_masked(self):
        rsamps = self.rsamps.copy()
        rsamps[3, 5] = np.inf
        ll = los_clouds_loglike_samples(self.theta, self.dsamps, rsamps)
        assert np.isfinite(ll)


class TestKernelTruncation:
    """Kernels renormalized for truncation at the reddening bounds."""

    def test_gauss_truncated_normalization_exact(self):
        reds = np.array([0.0, 0.1, 0.5])
        mu, sig = 0.05, 0.15
        rlims = (0.0, 6.0)
        base = kernel_gauss(reds, (mu, sig))
        trunc = kernel_gauss(reds, (mu, sig), rlims=rlims)
        mass = norm.cdf((rlims[1] - mu) / sig) - norm.cdf((rlims[0] - mu) / sig)
        assert np.allclose(trunc, base - np.log(mass), rtol=1e-12)

    def test_lorentz_truncated_normalization_exact(self):
        reds = np.array([0.0, 0.1, 0.5])
        mu, g = 0.05, 0.15
        rlims = (0.0, 6.0)
        base = kernel_lorentz(reds, (mu, g))
        trunc = kernel_lorentz(reds, (mu, g), rlims=rlims)
        mass = (np.arctan((rlims[1] - mu) / g) - np.arctan((rlims[0] - mu) / g)) / np.pi
        assert np.allclose(trunc, base - np.log(mass), rtol=1e-12)

    def test_tophat_truncated_support_clipped(self):
        # support [-0.1, 0.3] clipped to [0.0, 0.3]: norm 0.3, not 0.4
        reds = np.array([-0.05, 0.05, 0.25, 0.35])
        logw = kernel_tophat(reds, (0.1, 0.2), rlims=(0.0, 6.0))
        assert logw[0] == -np.inf  # below the clipped support
        assert np.allclose(logw[1:3], -np.log(0.3))
        assert logw[3] == -np.inf

    def test_kernels_default_rlims_none_unchanged(self):
        reds = np.array([0.1, 0.3, 0.5])
        mu, sig = 0.3, 0.2
        assert np.allclose(
            kernel_gauss(reds, (mu, sig)),
            -0.5 * ((reds - mu) / sig) ** 2 - np.log(np.sqrt(2 * np.pi) * sig),
        )
        assert np.allclose(
            kernel_lorentz(reds, (mu, sig)),
            -np.log(1 + ((reds - mu) / sig) ** 2) - np.log(np.pi * sig),
        )
        reds_in = np.array([0.15, 0.3, 0.45])  # strictly inside [0.1, 0.5)
        assert np.allclose(kernel_tophat(reds_in, (mu, sig)), -np.log(2 * sig))

    def test_kernel_zero_mass_gives_neg_inf(self):
        # kernel centered far outside the admissible range: no mass
        logw = kernel_tophat(np.array([0.5]), (10.0, 0.2), rlims=(0.0, 6.0))
        assert logw[0] == -np.inf

    def test_boundary_foreground_not_biased_high(self):
        """Profile likelihood must peak at the true fred = 0.

        Regression: without truncation renormalization, half of a boundary
        kernel's mass lies at A_V < 0 where no sample can fall, so the
        likelihood preferred fred ~ +1 kernel width over the true fred = 0.
        """
        rng = np.random.default_rng(9)
        nobj, nsamps = 200, 25
        sig = 0.15
        rsamps = np.abs(rng.normal(0.0, sig, (nobj, nsamps)))  # truth: fred = 0
        dsamps = np.full((nobj, nsamps), 5.0)  # all in the foreground
        s0 = sig / 6.0  # kernel width s0 * area = sig

        fred_grid = np.linspace(0.0, 0.4, 41)
        lnl = [
            los_clouds_loglike_samples(
                [1e-10, s0, s0, fred, 15.0, max(fred, 0.5)],
                dsamps,
                rsamps,
                kernel="gauss",
                Ndraws=nsamps,
            )
            for fred in fred_grid
        ]
        peak = fred_grid[int(np.argmax(lnl))]
        # untruncated kernels peaked near ~0.1 (about +0.7 kernel widths)
        assert peak <= 0.02


class TestPriorTransformValidation:
    """Degenerate hyperparameters must raise instead of quietly making NaN."""

    def test_nonpositive_std_raises(self):
        u = np.full(6, 0.5)
        with pytest.raises(ValueError, match="pb_params.*standard deviation"):
            los_clouds_priortransform(u, pb_params=(-3.0, 0.0, -np.inf, 0.0))
        with pytest.raises(ValueError, match="s_params.*standard deviation"):
            los_clouds_priortransform(u, s_params=(-3.0, -0.3, -np.inf, 0.0))

    def test_inverted_bounds_raise(self):
        u = np.full(6, 0.5)
        with pytest.raises(ValueError, match="pb_params.*lower bound"):
            los_clouds_priortransform(u, pb_params=(-3.0, 0.7, 0.0, -1.0))


class TestPriorTransformClosedForm:
    """ndtri-based transform must match scipy.stats.truncnorm."""

    @staticmethod
    def _reference_transform(u, pb_params, s_params):
        """Old scipy.stats-based head of los_clouds_priortransform."""
        pb_mean, pb_std, pb_low, pb_high = pb_params
        a = (pb_low - pb_mean) / pb_std
        b = (pb_high - pb_mean) / pb_std
        x0 = np.exp(truncnorm.ppf(u[0], a, b, loc=pb_mean, scale=pb_std))
        s_mean, s_std, s_low, s_high = s_params
        a = (s_low - s_mean) / s_std
        b = (s_high - s_mean) / s_std
        x1 = np.exp(truncnorm.ppf(u[1], a, b, loc=s_mean, scale=s_std))
        x2 = np.exp(truncnorm.ppf(u[2], a, b, loc=s_mean, scale=s_std))
        return x0, x1, x2

    @pytest.mark.parametrize(
        "pb_params,s_params",
        [
            ((-3.0, 0.7, -np.inf, 0.0), (-3.0, 0.3, -np.inf, 0.0)),  # defaults
            ((-2.0, 0.5, -5.0, 0.0), (-2.5, 0.4, -4.0, -0.5)),  # finite bounds
        ],
    )
    def test_matches_truncnorm(self, pb_params, s_params):
        rng = np.random.default_rng(5)
        for _ in range(20):
            u = rng.uniform(0.001, 0.999, 6)
            x = los_clouds_priortransform(u, pb_params=pb_params, s_params=s_params)
            x0, x1, x2 = self._reference_transform(u, pb_params, s_params)
            assert np.allclose(x[:3], [x0, x1, x2], rtol=1e-9)

    def test_extreme_quantiles(self):
        # q = 0 maps to the lower truncation bound (0 for exp(-inf))
        u = np.array([0.0, 0.0, 0.0, 0.4, 0.6, 0.8])
        x = los_clouds_priortransform(u)
        assert np.allclose(x[:3], 0.0)
        # q = 1 maps to the upper truncation bound exp(0) = 1
        u = np.array([1.0, 1.0, 1.0, 0.4, 0.6, 0.8])
        x = los_clouds_priortransform(u)
        assert np.allclose(x[:3], 1.0)

    def test_distances_and_reddenings_untouched(self):
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        x = los_clouds_priortransform(u)
        assert np.allclose(x[4], 0.5 * 15 + 4)
        assert np.allclose(x[6], 0.7 * 15 + 4)
        assert np.allclose(x[3], 0.4 * 6)
        assert np.allclose(x[5], 0.6 * 6)
        assert np.allclose(x[7], 0.8 * 6)
