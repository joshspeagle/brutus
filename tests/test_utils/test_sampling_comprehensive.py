#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus sampling utilities.

This test suite includes:
1. Unit tests for the new sampling functions
2. Comparison tests between old and new implementations
3. Integration tests for statistical properties and workflows
"""

import numpy as np
import pytest


class TestQuantileFunction:
    """Unit tests for quantile computation."""

    def test_quantile_unweighted_basic(self):
        """Test basic unweighted quantile computation."""
        from brutus.utils.sampling import quantile

        # Simple test case
        x = np.array([1, 2, 3, 4, 5])
        q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        result = quantile(x, q)

        # Midpoint-CDF convention: identical to explicit uniform weights.
        expected = np.array([1.0, 1.75, 3.0, 4.25, 5.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        np.testing.assert_array_almost_equal(
            result, quantile(x, q, weights=np.ones_like(x)), decimal=12
        )

    def test_quantile_weighted_basic(self):
        """Test basic weighted quantile computation."""
        from brutus.utils.sampling import quantile

        # Test case where last sample is heavily weighted
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([1, 1, 1, 1, 10])  # Last sample heavily weighted
        q = np.array([0.5, 0.9])

        result = quantile(x, q, weights=weights)

        # With heavy weighting on last sample, quantiles should be close to 5
        assert result[0] > 4.0  # Median should be > 4
        assert result[1] > 4.5  # 90th percentile should be close to 5

    def test_quantile_single_value(self):
        """Test quantile with single quantile value."""
        from brutus.utils.sampling import quantile

        x = np.array([1, 2, 3, 4, 5])
        q = 0.5  # Single value, not array

        result = quantile(x, q)
        expected = np.percentile(x, 50)

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_quantile_edge_cases(self):
        """Test quantile edge cases."""
        from brutus.utils.sampling import quantile

        x = np.array([1, 2, 3, 4, 5])

        # Test extreme quantiles
        q_min = quantile(x, 0.0)
        q_max = quantile(x, 1.0)

        assert q_min == 1.0
        assert q_max == 5.0

    def test_quantile_error_conditions(self):
        """Test quantile error conditions."""
        from brutus.utils.sampling import quantile

        x = np.array([1, 2, 3, 4, 5])

        # Quantiles outside [0, 1] should raise error
        with pytest.raises(ValueError, match="Quantiles must be between"):
            quantile(x, [-0.1, 0.5])

        with pytest.raises(ValueError, match="Quantiles must be between"):
            quantile(x, [0.5, 1.1])

        # Mismatched dimensions should raise error
        weights = np.array([1, 2, 3])  # Wrong length
        with pytest.raises(ValueError, match="Dimension mismatch"):
            quantile(x, 0.5, weights=weights)


class TestDrawSARFunction:
    """Unit tests for draw_sar function."""

    def test_draw_sar_basic(self):
        """Test basic draw_sar functionality."""
        from brutus.utils.sampling import draw_sar

        # Simple test case
        np.random.seed(42)
        nsamps = 2
        ndraws = 100

        scales = np.array([1.0, 1.1])
        avs = np.array([0.1, 0.2])
        rvs = np.array([3.1, 3.3])
        covs_sar = np.array(
            [
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
            ]
        )

        sdraws, adraws, rdraws = draw_sar(
            scales, avs, rvs, covs_sar, ndraws=ndraws, rstate=np.random.RandomState(42)
        )

        # Check output shapes
        assert sdraws.shape == (nsamps, ndraws)
        assert adraws.shape == (nsamps, ndraws)
        assert rdraws.shape == (nsamps, ndraws)

        # Check that all draws are within bounds
        assert np.all(sdraws >= 0)
        assert np.all(adraws >= 0) and np.all(adraws <= 6)
        assert np.all(rdraws >= 1) and np.all(rdraws <= 8)

    def test_draw_sar_means(self):
        """Test that draw_sar produces samples with correct means."""
        from brutus.utils.sampling import draw_sar

        np.random.seed(42)
        nsamps = 1
        ndraws = 5000  # Large number for statistical accuracy

        # Use tight covariances so samples stay close to means
        scales = np.array([1.0])
        avs = np.array([2.0])
        rvs = np.array([3.0])
        covs_sar = np.array([[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]])

        sdraws, adraws, rdraws = draw_sar(
            scales, avs, rvs, covs_sar, ndraws=ndraws, rstate=np.random.RandomState(42)
        )

        # Sample means should be close to input means
        np.testing.assert_almost_equal(np.mean(sdraws), 1.0, decimal=1)
        np.testing.assert_almost_equal(np.mean(adraws), 2.0, decimal=1)
        np.testing.assert_almost_equal(np.mean(rdraws), 3.0, decimal=1)

    def test_draw_sar_limits(self):
        """Test draw_sar respects limits."""
        from brutus.utils.sampling import draw_sar

        np.random.seed(42)

        # Set up case where samples would naturally exceed limits
        scales = np.array([1.0])
        avs = np.array([0.1])  # Close to lower limit
        rvs = np.array([7.5])  # Close to upper limit
        # Large covariances to push samples outside limits
        covs_sar = np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])

        # Custom tight limits
        avlim = (0.05, 0.2)
        rvlim = (7.0, 8.0)

        sdraws, adraws, rdraws = draw_sar(
            scales,
            avs,
            rvs,
            covs_sar,
            ndraws=100,
            avlim=avlim,
            rvlim=rvlim,
            rstate=np.random.RandomState(42),
        )

        # All samples should respect limits
        assert np.all(adraws >= avlim[0]) and np.all(adraws <= avlim[1])
        assert np.all(rdraws >= rvlim[0]) and np.all(rdraws <= rvlim[1])


class TestSampleMultivariateNormal:
    """Unit tests for sample_multivariate_normal function."""

    def test_sample_multivariate_normal_single(self):
        """Test single distribution sampling."""
        from brutus.utils.sampling import sample_multivariate_normal

        np.random.seed(42)

        # 2D distribution
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        size = 1000

        samples = sample_multivariate_normal(
            mean, cov, size=size, rstate=np.random.RandomState(42)
        )

        # Check shape
        assert samples.shape == (2, size)

        # Check sample statistics
        sample_mean = np.mean(samples, axis=1)
        sample_cov = np.cov(samples)

        np.testing.assert_array_almost_equal(sample_mean, mean, decimal=1)
        np.testing.assert_array_almost_equal(sample_cov, cov, decimal=1)

    def test_sample_multivariate_normal_multiple(self):
        """Test multiple distribution sampling."""
        from brutus.utils.sampling import sample_multivariate_normal

        np.random.seed(42)

        # 2 distributions, 2D each
        means = np.array([[1.0, 2.0], [3.0, 4.0]])
        covs = np.array([[[1.0, 0.2], [0.2, 1.0]], [[2.0, 0.5], [0.5, 2.0]]])
        size = 500

        samples = sample_multivariate_normal(
            means, covs, size=size, rstate=np.random.RandomState(42)
        )

        # Check shape
        assert samples.shape == (2, size, 2)  # (dim, size, Ndist)

        # Check sample statistics for each distribution
        for i in range(2):
            sample_mean = np.mean(samples[:, :, i], axis=1)
            sample_cov = np.cov(samples[:, :, i])

            # Use looser tolerance for statistical tests with limited sample size
            np.testing.assert_array_almost_equal(sample_mean, means[i], decimal=0)
            np.testing.assert_array_almost_equal(sample_cov, covs[i], decimal=0)

    def test_sample_multivariate_normal_stability(self):
        """Test numerical stability with eps parameter."""
        from brutus.utils.sampling import sample_multivariate_normal

        # Near-singular covariance matrix
        mean = np.array([0.0, 0.0])
        cov = np.array([[1e-10, 0], [0, 1e-10]])

        # Should not raise error due to eps regularization
        samples = sample_multivariate_normal(
            mean, cov, size=10, eps=1e-8, rstate=np.random.RandomState(42)
        )

        assert samples.shape == (2, 10)
        assert np.all(np.isfinite(samples))


class TestSamplingComparison:
    """Comparison tests between old and new implementations.

    NOTE: These tests will be removed after refactoring is complete.
    They exist to ensure consistency during the transition period.
    """

    def test_quantile_single_sample(self):
        """Test quantile with single sample."""
        from brutus.utils.sampling import quantile

        x = np.array([5.0])
        q = np.array([0.0, 0.5, 1.0])

        result = quantile(x, q)
        expected = np.array([5.0, 5.0, 5.0])

        np.testing.assert_array_equal(result, expected)

    def test_draw_sar_tight_limits(self):
        """Test draw_sar with very tight limits."""
        from brutus.utils.sampling import draw_sar

        # Set up case where it's hard to find samples within limits
        scales = np.array([1.0])
        avs = np.array([3.0])  # Center of range
        rvs = np.array([4.5])  # Center of range
        covs_sar = np.array([[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]])

        # Very tight limits
        avlim = (2.99, 3.01)
        rvlim = (4.49, 4.51)

        sdraws, adraws, rdraws = draw_sar(
            scales,
            avs,
            rvs,
            covs_sar,
            ndraws=10,
            avlim=avlim,
            rvlim=rvlim,
            rstate=np.random.RandomState(42),
        )

        # Should still produce valid samples
        assert sdraws.shape == (1, 10)
        assert np.all(adraws >= avlim[0]) and np.all(adraws <= avlim[1])
        assert np.all(rdraws >= rvlim[0]) and np.all(rdraws <= rvlim[1])

    def test_sample_multivariate_normal_1d(self):
        """Test sample_multivariate_normal with 1D distributions."""
        from brutus.utils.sampling import sample_multivariate_normal

        # Single 1D distribution
        mean = np.array([2.0])
        cov = np.array([[1.0]])

        samples = sample_multivariate_normal(
            mean, cov, size=100, rstate=np.random.RandomState(42)
        )

        assert samples.shape == (1, 100)
        np.testing.assert_almost_equal(np.mean(samples), 2.0, decimal=1)


class TestSamplingIntegration:
    """Integration tests for sampling functions."""

    def test_quantile_weighted_properties(self):
        """Test statistical properties of weighted quantiles."""
        from brutus.utils.sampling import quantile

        # Create samples with known quantiles
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        weights = np.ones_like(x)  # Equal weights initially

        # With equal weights, median should be 5.5
        median = quantile(x, 0.5, weights=weights)
        np.testing.assert_almost_equal(median, 5.5, decimal=1)

        # Now weight the higher values more heavily
        weights[-3:] = 10  # Weight last 3 samples heavily

        # Median should shift upward
        weighted_median = quantile(x, 0.5, weights=weights)
        assert weighted_median > median

    def test_sampling_workflow_integration(self):
        """Test a realistic sampling workflow."""
        from brutus.utils.sampling import draw_sar, quantile, sample_multivariate_normal

        np.random.seed(42)

        # Step 1: Generate initial samples from multivariate normal
        means = np.array([[1.0, 0.1, 3.1], [1.1, 0.2, 3.3]])  # scale, av, rv
        covs = np.array(
            [
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
                [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.1]],
            ]
        )

        initial_samples = sample_multivariate_normal(
            means, covs, size=100, rstate=np.random.RandomState(42)
        )

        # Step 2: Extract parameters for draw_sar
        scales = means[:, 0]
        avs = means[:, 1]
        rvs = means[:, 2]

        # Step 3: Generate more samples with draw_sar
        sdraws, adraws, rdraws = draw_sar(
            scales, avs, rvs, covs, ndraws=200, rstate=np.random.RandomState(42)
        )

        # Step 4: Compute quantiles of the results
        for i in range(len(scales)):
            av_quantiles = quantile(adraws[i], [0.16, 0.5, 0.84])
            rv_quantiles = quantile(rdraws[i], [0.16, 0.5, 0.84])

            # Quantiles should be reasonable
            assert len(av_quantiles) == 3
            assert len(rv_quantiles) == 3
            assert np.all(np.isfinite(av_quantiles))
            assert np.all(np.isfinite(rv_quantiles))

            # Should be in ascending order
            assert av_quantiles[0] <= av_quantiles[1] <= av_quantiles[2]
            assert rv_quantiles[0] <= rv_quantiles[1] <= rv_quantiles[2]

    def test_mcmc_like_workflow(self):
        """Test an MCMC-like workflow using sampling utilities."""
        from brutus.utils.sampling import quantile, sample_multivariate_normal

        np.random.seed(42)

        # Simulate an MCMC chain
        nsteps = 1000
        ndim = 3

        # Starting point
        current = np.zeros(ndim)
        chain = np.zeros((nsteps, ndim))

        # Proposal covariance
        proposal_cov = 0.1 * np.eye(ndim)

        for i in range(nsteps):
            # Propose new state
            proposal = sample_multivariate_normal(
                current, proposal_cov, size=1, rstate=np.random.RandomState(i)
            )
            proposal = proposal[:, 0]  # Extract single sample

            # Simple acceptance (always accept for this test)
            current = proposal
            chain[i] = current

        # Analyze results using quantile
        for dim in range(ndim):
            dim_samples = chain[:, dim]
            quantiles = quantile(dim_samples, [0.16, 0.5, 0.84])

            # Should have reasonable spread
            iqr = (
                quantiles[2] - quantiles[0]
            )  # Interquartile range (16th to 84th percentile)
            assert iqr > 0.1  # Should have some spread
            assert (
                iqr < 15.0
            )  # But not too much (relaxed tolerance for statistical variation)


if __name__ == "__main__":
    pytest.main([__file__])
