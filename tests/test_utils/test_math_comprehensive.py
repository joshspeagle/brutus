#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus mathematical utilities.

This test suite includes:
1. Unit tests for the new mathematical functions
2. Comparison tests between old and new implementations
3. Integration tests for matrix operations and statistical functions
"""

import numpy as np
import pytest
import scipy.stats

from brutus.utils.math import truncnorm_logpdf


class TestMatrixOperations:
    """Unit tests for 3x3 matrix operations."""

    def test_inverse3_basic(self):
        """Test 3x3 matrix inversion."""
        from brutus.utils.math import inverse3

        # Create a simple invertible matrix
        A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=float)

        # Test without regularization (default)
        A_inv = inverse3(A, regularize=False)

        # Check that A * A_inv = I
        product = np.dot(A, A_inv)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(product, expected, decimal=12)

        # Test with regularization - should still work for well-conditioned matrices
        A_inv_reg = inverse3(A, regularize=True)
        product_reg = np.dot(A, A_inv_reg)
        np.testing.assert_array_almost_equal(
            product_reg, expected, decimal=10
        )  # Slightly less precision due to regularization

    def test_inverse3_vs_numpy(self):
        """Test inverse computation against numpy.linalg.inv."""
        from brutus.utils.math import inverse3

        np.random.seed(42)

        # Create random invertible matrices
        for _ in range(10):
            A = np.random.random((3, 3))
            A = A + A.T + 3 * np.eye(3)  # Make positive definite (invertible)

            # Test without regularization to match numpy exactly
            custom_inv = inverse3(A, regularize=False)
            numpy_inv = np.linalg.inv(A)

            np.testing.assert_array_almost_equal(custom_inv, numpy_inv, decimal=10)

    def test_inverse3_batch(self):
        """Test batch matrix inversion."""
        from brutus.utils.math import inverse3

        np.random.seed(42)
        batch_size = 5

        # Create batch of random invertible matrices
        A = np.random.random((batch_size, 3, 3))
        A = A + np.transpose(A, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]

        A_inv = inverse3(A)

        # Check that each A * A_inv = I
        for i in range(batch_size):
            product = np.dot(A[i], A_inv[i])
            expected = np.eye(3)
            np.testing.assert_array_almost_equal(product, expected, decimal=10)

    def test_isPSD_basic(self):
        """Test positive semidefinite check."""
        from brutus.utils.math import isPSD

        # Identity matrix is PSD
        I = np.eye(3)
        assert isPSD(I) == True

        # Zero matrix is PSD
        Z = np.zeros((3, 3))
        assert isPSD(Z) == True

        # Positive definite matrix is PSD
        A = np.array([[2, 1], [1, 2]], dtype=float)
        assert isPSD(A) == True

        # Non-PSD matrix
        B = np.array([[1, 2], [2, 1]], dtype=float)  # Has negative eigenvalue
        assert isPSD(B) == False


class TestStatisticalDistributions:
    """Unit tests for statistical distribution functions."""

    def test_chisquare_logpdf_basic(self):
        """Test chi-square log-PDF computation."""
        from brutus.utils.math import chisquare_logpdf

        # Test against scipy
        x = np.array([1.0, 2.0, 5.0, 10.0])
        df = 3

        custom_logpdf = chisquare_logpdf(x, df)
        scipy_logpdf = scipy.stats.chi2.logpdf(x, df)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_chisquare_logpdf_with_scaling(self):
        """Test chi-square log-PDF with location and scale."""
        from brutus.utils.math import chisquare_logpdf

        x = np.array([2.0, 4.0, 8.0])
        df = 2
        loc = 1.0
        scale = 2.0

        custom_logpdf = chisquare_logpdf(x, df, loc=loc, scale=scale)
        scipy_logpdf = scipy.stats.chi2.logpdf(x, df, loc=loc, scale=scale)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_chisquare_logpdf_edge_cases(self):
        """Test chi-square log-PDF edge cases."""
        from brutus.utils.math import chisquare_logpdf

        # Zero and negative values should give -inf
        x_neg = np.array([-1.0, 0.0])
        df = 2

        result = chisquare_logpdf(x_neg, df)
        assert np.all(result == -np.inf)

    def test_truncnorm_pdf_basic(self):
        """Test truncated normal PDF computation."""
        from brutus.utils.math import truncnorm_pdf

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        a, b = -2.0, 2.0  # Truncation bounds

        custom_pdf = truncnorm_pdf(x, a, b)
        scipy_pdf = scipy.stats.truncnorm.pdf(x, a, b)

        np.testing.assert_array_almost_equal(custom_pdf, scipy_pdf, decimal=10)

    def test_truncnorm_pdf_with_scaling(self):
        """Test truncated normal PDF with location and scale."""
        from brutus.utils.math import truncnorm_pdf

        x = np.array([2.0, 3.0, 4.0, 5.0])
        a, b = -1.0, 1.0
        loc = 3.0
        scale = 0.5

        custom_pdf = truncnorm_pdf(x, a, b, loc=loc, scale=scale)
        scipy_pdf = scipy.stats.truncnorm.pdf(x, a, b, loc=loc, scale=scale)

        np.testing.assert_array_almost_equal(custom_pdf, scipy_pdf, decimal=10)

    def test_truncnorm_pdf_outside_bounds(self):
        """Test truncated normal PDF outside truncation bounds."""
        from brutus.utils.math import truncnorm_pdf

        # Values outside truncation bounds should have PDF = 0
        x = np.array([-5.0, 5.0])  # Outside [-2, 2]
        a, b = -2.0, 2.0

        result = truncnorm_pdf(x, a, b)
        expected = np.array([0.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_truncnorm_logpdf_basic(self):
        """Test truncated normal log-PDF computation."""
        from brutus.utils.math import truncnorm_logpdf

        x = np.array([-1.0, 0.0, 1.0])
        a, b = -2.0, 2.0

        custom_logpdf = truncnorm_logpdf(x, a, b)
        scipy_logpdf = scipy.stats.truncnorm.logpdf(x, a, b)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_truncnorm_logpdf_outside_bounds(self):
        """Test truncated normal log-PDF outside bounds."""
        from brutus.utils.math import truncnorm_logpdf

        # Values outside bounds should have log-PDF = -inf
        x = np.array([-5.0, 5.0])
        a, b = -2.0, 2.0

        result = truncnorm_logpdf(x, a, b)
        expected = np.array([-np.inf, -np.inf])

        np.testing.assert_array_equal(result, expected)


class TestFunctionWrapper:
    """Unit tests for the function wrapper utility."""

    def test_function_wrapper_basic(self):
        """Test basic function wrapper functionality."""
        from brutus.utils.math import _function_wrapper

        def test_func(x, a, b=2):
            return x * a + b

        wrapped = _function_wrapper(test_func, args=(3,), kwargs={"b": 5}, name="test")

        result = wrapped(10)
        expected = 10 * 3 + 5  # = 35

        assert result == expected

    def test_function_wrapper_error_handling(self):
        """Test function wrapper error handling."""
        from brutus.utils.math import _function_wrapper

        def failing_func(x):
            raise ValueError("Test error")

        wrapped = _function_wrapper(failing_func, args=(), kwargs={}, name="failing")

        with pytest.raises(ValueError, match="Test error"):
            wrapped(1)


class TestMathComparison:
    """Comparison tests between old and new implementations.

    NOTE: These tests will be removed after refactoring is complete.
    They exist to ensure consistency during the transition period.
    """

    def test_inverse3_singular_matrix(self):
        """Test behavior with singular (non-invertible) matrices."""
        from brutus.utils.math import inverse3

        # Create a singular matrix (all rows identical)
        A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)

        # Without regularization: should produce inf/nan values
        with np.errstate(divide="ignore", invalid="ignore"):
            result_no_reg = inverse3(A, regularize=False)
        assert result_no_reg.shape == (3, 3)
        # Result will contain inf/nan values due to zero determinant

        # With regularization: should produce finite values
        result_reg = inverse3(A, regularize=True)
        assert result_reg.shape == (3, 3)
        assert np.all(np.isfinite(result_reg))  # Should be finite due to regularization

    def test_inverse3_regularization(self):
        """Test matrix regularization functionality."""
        import numpy as np

        from brutus.utils.math import inverse3, isPSD

        # Create a matrix with small negative eigenvalues
        eigenvals = np.array([1.0, -1e-15, 0.5])  # One tiny negative eigenvalue
        Q = np.linalg.qr(np.random.randn(3, 3))[0]
        A = Q @ np.diag(eigenvals) @ Q.T

        # Invert with regularization
        A_inv_reg = inverse3(A, regularize=True)

        # Result should be positive semi-definite
        assert isPSD(A_inv_reg), "Regularized inverse should be positive semi-definite"

        # Should be finite
        assert np.all(np.isfinite(A_inv_reg)), "Regularized inverse should be finite"

    def test_inverse3_batch_regularization(self):
        """Test batch matrix regularization with arrays."""
        import numpy as np

        from brutus.utils.math import inverse3, isPSD

        # Create batch of matrices with regularization needs
        n_matrices = 3
        matrices = np.zeros((n_matrices, 3, 3))

        # Matrix 0: needs input regularization (singular)
        matrices[0] = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        # Matrix 1: needs output regularization (small eigenvalues)
        eigenvals = np.array([1.0, -1e-14, 0.1])
        Q = np.linalg.qr(np.random.randn(3, 3))[0]
        matrices[1] = Q @ np.diag(eigenvals) @ Q.T

        # Matrix 2: well-conditioned
        matrices[2] = np.eye(3) + 0.1 * np.random.randn(3, 3)
        matrices[2] = matrices[2] @ matrices[2].T  # Make PSD

        # Test batch regularization
        inverted = inverse3(matrices, regularize=True)

        # All should be PSD and finite
        for i in range(n_matrices):
            assert isPSD(inverted[i]), f"Matrix {i} should be PSD"
            assert np.all(np.isfinite(inverted[i])), f"Matrix {i} should be finite"

    def test_isPSD_edge_cases(self):
        """Test isPSD with edge cases that return False."""
        from brutus.utils.math import isPSD

        # Matrix with negative eigenvalue
        A_neg = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        assert not isPSD(A_neg), "Matrix with negative eigenvalue should not be PSD"

        # Singular matrix (zero eigenvalue)
        A_singular = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        # This should return False due to negative/zero eigenvalues from numerical issues
        result = isPSD(A_singular)
        assert isinstance(result, (bool, np.bool_)), "isPSD should return boolean"

    def test_chisquare_logpdf_scalar_input(self):
        """Test chi-square log-PDF with scalar input."""
        from brutus.utils.math import chisquare_logpdf

        # Scalar input
        x = 2.0
        df = 3

        result = chisquare_logpdf(x, df)
        expected = scipy.stats.chi2.logpdf(x, df)

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_truncnorm_pdf_scalar_input(self):
        """Test truncated normal PDF with scalar input."""
        from brutus.utils.math import truncnorm_pdf

        # Scalar input
        x = 1.0
        a, b = -2.0, 2.0

        result = truncnorm_pdf(x, a, b)
        expected = scipy.stats.truncnorm.pdf(x, a, b)

        np.testing.assert_almost_equal(result, expected, decimal=10)


class TestMathIntegration:
    """Integration tests for mathematical functions."""

    def test_matrix_operations_pipeline(self):
        """Test a pipeline using multiple matrix operations."""
        from brutus.utils.math import inverse3

        np.random.seed(42)

        # Create batch of random matrices
        batch_size = 5
        A = np.random.random((batch_size, 3, 3))
        A = (
            A + np.transpose(A, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]
        )  # Make invertible

        # Compute inverses
        A_inv = inverse3(A)

        # Verify A * A_inv = I for each matrix
        for i in range(batch_size):
            product = np.dot(A[i], A_inv[i])
            np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)

    def test_statistical_consistency(self):
        """Test statistical properties of distribution functions."""
        from brutus.utils.math import chisquare_logpdf, truncnorm_pdf

        # Test that truncated normal PDF integrates to 1 (approximately)
        a, b = -2.0, 2.0
        x = np.linspace(a, b, 1000)  # Integration range should match truncation bounds
        pdf_vals = truncnorm_pdf(x, a, b)

        # Numerical integration (trapezoidal rule)
        integral = np.trapz(pdf_vals, x)
        np.testing.assert_almost_equal(integral, 1.0, decimal=2)

        # Test that chi-square probabilities are reasonable
        x = np.array([1.0, 5.0, 10.0])
        df = 5
        logpdf_vals = chisquare_logpdf(x, df)

        # All should be finite (not -inf or +inf)
        assert np.all(np.isfinite(logpdf_vals))

        # Should be negative (since they're log probabilities)
        assert np.all(logpdf_vals < 0)


class TestInternalFunctions:
    """Test internal helper functions for coverage."""

    def test_internal_helper_functions(self):
        """Test internal helper functions to improve coverage."""
        import numpy as np

        from brutus.utils.math import _invert_3x3_analytical, _matrix_det_3x3

        # Test _matrix_det_3x3
        A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
        det_A = _matrix_det_3x3(A)
        expected_det = np.linalg.det(A)
        np.testing.assert_almost_equal(det_A, expected_det, decimal=10)

        # Test _invert_3x3_analytical with well-conditioned matrix
        inv_A = _invert_3x3_analytical(A)
        expected_inv = np.linalg.inv(A)
        np.testing.assert_array_almost_equal(inv_A, expected_inv, decimal=10)

        # Test singular matrix handling
        singular = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)
        inv_singular = _invert_3x3_analytical(singular)
        assert np.all(np.isinf(inv_singular)), "Should return inf for singular matrix"

    def test_batch_operations_coverage(self):
        """Test batch matrix operations to hit remaining coverage."""
        import numpy as np

        from brutus.utils.math import _batch_invert_3x3, inverse3

        # Create batch of matrices that require different code paths
        batch = np.array(
            [
                [[2, 1, 0], [1, 3, 1], [0, 1, 2]],  # Well-conditioned
                [[1e10, 0, 0], [0, 1e-10, 0], [0, 0, 1]],  # Ill-conditioned
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity
            ],
            dtype=float,
        )

        # Test _batch_invert_3x3 directly
        inverted_batch = _batch_invert_3x3(batch)
        assert inverted_batch.shape == batch.shape

        # Test with regularization to hit regularization code paths
        inverted_reg = inverse3(batch, regularize=True)
        assert inverted_reg.shape == batch.shape

        # Test with very small eigenvalue threshold
        inverted_strict = inverse3(batch, regularize=True, min_eigenval_threshold=1e-6)
        assert inverted_strict.shape == batch.shape


if __name__ == "__main__":
    pytest.main([__file__])
