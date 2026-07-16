#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for audit fixes in brutus.utils.sampling and brutus.utils.math.

Covers:
1. truncnorm_pdf/truncnorm_logpdf tail-stable normalization (erf cancellation).
2. inverse3 relative (scale-invariant) singularity threshold.
3. PSD-safe _cholesky_3x3 (singular covariances no longer produce garbage).
4. quantile(): weights=None now agrees with explicit uniform weights.
5. draw_sar batched rewrite: distributional equivalence with the old
   per-sample numpy.random.multivariate_normal loop, plus preserved
   rejection/mean-padding semantics and rstate compatibility.
"""

import warnings

import numpy as np
import pytest
import scipy.stats

# ---------------------------------------------------------------------------
# 1. Truncated normal tail stability
# ---------------------------------------------------------------------------


class TestTruncnormTailStability:
    def test_logpdf_far_upper_tail(self):
        """Same-side bounds beyond ~8 sigma: old code returned +635, not -2.81."""
        from brutus.utils.math import truncnorm_logpdf

        result = truncnorm_logpdf(10.5, 10.0, 12.0)
        expected = scipy.stats.truncnorm.logpdf(10.5, 10.0, 12.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_pdf_far_upper_tail_finite(self):
        """Old code divided by an underflowed CDF difference and returned inf."""
        from brutus.utils.math import truncnorm_pdf

        with np.errstate(divide="raise"):
            result = truncnorm_pdf(9.5, 9.0, 12.0)
        expected = scipy.stats.truncnorm.pdf(9.5, 9.0, 12.0)
        assert np.isfinite(result)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_logpdf_far_lower_tail(self):
        """Mirror case: both bounds deep in the lower tail."""
        from brutus.utils.math import truncnorm_logpdf

        result = truncnorm_logpdf(-9.5, -12.0, -9.0)
        expected = scipy.stats.truncnorm.logpdf(-9.5, -12.0, -9.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tail_onset_range(self):
        """Agreement with scipy across the onset range where erf loses bits."""
        from brutus.utils.math import truncnorm_logpdf, truncnorm_pdf

        for a in [4.0, 6.0, 7.0, 8.0, 9.0, 15.0, 30.0]:
            b = a + 2.0
            x = a + 0.5
            np.testing.assert_allclose(
                truncnorm_logpdf(x, a, b),
                scipy.stats.truncnorm.logpdf(x, a, b),
                rtol=1e-9,
                err_msg=f"logpdf mismatch at a={a}",
            )
            np.testing.assert_allclose(
                truncnorm_pdf(x, a, b),
                scipy.stats.truncnorm.pdf(x, a, b),
                rtol=1e-9,
                err_msg=f"pdf mismatch at a={a}",
            )

    def test_ordinary_cases_unchanged(self):
        """Central bounds, arrays, loc/scale, and bound masking still match."""
        from brutus.utils.math import truncnorm_logpdf, truncnorm_pdf

        x = np.array([-3.0, -1.0, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(
            truncnorm_pdf(x, -2.0, 2.0),
            scipy.stats.truncnorm.pdf(x, -2.0, 2.0),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            truncnorm_logpdf(x, -2.0, 2.0),
            scipy.stats.truncnorm.logpdf(x, -2.0, 2.0),
            rtol=1e-12,
        )
        # loc/scale
        np.testing.assert_allclose(
            truncnorm_pdf(np.array([2.7, 3.0, 3.3]), -1.0, 1.0, loc=3.0, scale=0.5),
            scipy.stats.truncnorm.pdf(
                np.array([2.7, 3.0, 3.3]), -1.0, 1.0, loc=3.0, scale=0.5
            ),
            rtol=1e-12,
        )

    def test_array_bounds(self):
        """Per-sample array bounds (as passed by priors.galactic) still work
        and are tail-stable elementwise."""
        from brutus.utils.math import truncnorm_logpdf, truncnorm_pdf

        a = np.array([-2.0, 0.5, 10.0, -12.0])
        b = np.array([2.0, 3.0, 12.0, -9.0])
        x = np.array([0.5, 1.0, 10.5, -9.5])
        expected = np.array(
            [scipy.stats.truncnorm.logpdf(xi, ai, bi) for xi, ai, bi in zip(x, a, b)]
        )
        np.testing.assert_allclose(truncnorm_logpdf(x, a, b), expected, rtol=1e-10)
        np.testing.assert_allclose(truncnorm_pdf(x, a, b), np.exp(expected), rtol=1e-10)

    def test_infinite_bounds(self):
        """(-inf, inf) reduces to the standard normal log-density."""
        from brutus.utils.math import truncnorm_logpdf

        np.testing.assert_allclose(
            truncnorm_logpdf(0.5, -np.inf, np.inf),
            scipy.stats.norm.logpdf(0.5),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            truncnorm_logpdf(1.0, 0.0, np.inf),
            scipy.stats.truncnorm.logpdf(1.0, 0.0, np.inf),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# 2. inverse3 relative singularity threshold
# ---------------------------------------------------------------------------


class TestInverse3RelativeThreshold:
    def test_small_magnitude_matrix_invertible(self):
        """diag(1e-6): det=1e-18 < old absolute cutoff -> was all-inf."""
        from brutus.utils import inverse3

        A = np.eye(3) * 1e-6
        np.testing.assert_allclose(inverse3(A), np.eye(3) * 1e6, rtol=1e-12)

    def test_small_magnitude_batch(self):
        from brutus.utils import inverse3

        A = (np.eye(3) * 1e-6)[None]
        np.testing.assert_allclose(inverse3(A)[0], np.eye(3) * 1e6, rtol=1e-12)

    def test_realistic_covariance_scale(self):
        """Faint-star (scale, Av, Rv) covariance magnitudes stay invertible."""
        from brutus.utils import inverse3

        A = np.diag([1e-10, 1e-2, 0.03])
        np.testing.assert_allclose(
            inverse3(A), np.diag([1e10, 1e2, 1.0 / 0.03]), rtol=1e-12
        )

    def test_exactly_singular_still_flagged(self):
        from brutus.utils import inverse3

        A = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert np.all(np.isinf(inverse3(A)))
        # Zero matrix is singular too (guards the det_scale == 0 branch).
        assert np.all(np.isinf(inverse3(np.zeros((3, 3)))))

    def test_ill_scaled_diagonal_still_invertible(self):
        """diag(1e10, 1e-10, 1) is perfectly conditioned row-wise; the
        relative threshold must not misclassify it (max-entry^3 would)."""
        from brutus.utils import inverse3

        A = np.diag([1e10, 1e-10, 1.0])
        np.testing.assert_allclose(inverse3(A), np.diag([1e-10, 1e10, 1.0]), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. PSD-safe Cholesky and singular covariances in the batched sampler
# ---------------------------------------------------------------------------

SINGULAR_PSD = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestCholeskyPSD:
    def test_cholesky_singular_psd(self):
        """Old code divided by a zero pivot (ZeroDivisionError under JIT,
        NaN in the pure-Python fallback)."""
        from brutus.utils.sampling import _cholesky_3x3

        L = _cholesky_3x3(SINGULAR_PSD)
        assert np.all(np.isfinite(L))
        np.testing.assert_allclose(L @ L.T, SINGULAR_PSD, atol=1e-14)

    def test_cholesky_zero_matrix(self):
        from brutus.utils.sampling import _cholesky_3x3

        L = _cholesky_3x3(np.zeros((3, 3)))
        np.testing.assert_array_equal(L, np.zeros((3, 3)))

    def test_cholesky_pd_matches_numpy(self):
        """Strictly PD input must be identical to the textbook factorization."""
        from brutus.utils.sampling import _cholesky_3x3

        rng = np.random.RandomState(7)
        for _ in range(10):
            M = rng.standard_normal((3, 3))
            A = M @ M.T + 0.5 * np.eye(3)
            np.testing.assert_allclose(
                _cholesky_3x3(A), np.linalg.cholesky(A), rtol=1e-12, atol=1e-14
            )

    def test_sampler_singular_covariance(self):
        """A batch containing one singular PSD covariance previously returned
        uninitialized memory (JIT) or NaN (no-JIT) for that column."""
        from brutus.utils.sampling import sample_multivariate_normal

        means = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        covs = np.stack([np.eye(3), SINGULAR_PSD, 2.0 * np.eye(3)])
        out = sample_multivariate_normal(
            means, covs, size=4000, rstate=np.random.RandomState(3)
        )
        assert out.shape == (3, 4000, 3)
        assert np.all(np.isfinite(out))
        # Degenerate structure: components 0 and 1 are perfectly correlated,
        # so (x0 - mean0) == (x1 - mean1) draw-by-draw.
        np.testing.assert_allclose(out[0, :, 1] - 1.0, out[1, :, 1] - 2.0, atol=1e-12)
        # Marginal moments of the singular column match the covariance.
        np.testing.assert_allclose(out[:, :, 1].mean(axis=1), [1.0, 2.0, 3.0], atol=0.1)
        np.testing.assert_allclose(out[:, :, 1].std(axis=1), [1.0, 1.0, 1.0], atol=0.1)
        # Non-singular columns unaffected.
        np.testing.assert_allclose(out[:, :, 0].std(axis=1), 1.0, atol=0.1)
        np.testing.assert_allclose(out[:, :, 2].std(axis=1), np.sqrt(2.0), atol=0.1)


# ---------------------------------------------------------------------------
# 4. quantile convention consistency
# ---------------------------------------------------------------------------


class TestQuantileConsistency:
    def test_none_equals_uniform_weights(self):
        """weights=None and explicit uniform weights must agree (they used
        two different quantile conventions before)."""
        from brutus.utils.sampling import quantile

        rng = np.random.RandomState(11)
        x = rng.standard_normal(101)
        q = np.array([0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975])
        np.testing.assert_array_equal(
            quantile(x, q), quantile(x, q, weights=np.ones_like(x))
        )

    def test_midpoint_convention_values(self):
        from brutus.utils.sampling import quantile

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(quantile(x, [0.25, 0.5, 0.75]), [1.75, 3.0, 4.25])

    def test_weighted_path_unchanged(self):
        """Explicit non-uniform weights keep their historical behavior."""
        from brutus.utils.sampling import quantile

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        np.testing.assert_allclose(
            quantile(x, [0.25, 0.5, 0.75], weights=w),
            [4.0, 4.63636364, 5.0],
            rtol=1e-8,
        )

    def test_extremes_clamp_to_data_range(self):
        from brutus.utils.sampling import quantile

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert quantile(x, 0.0)[0] == 1.0
        assert quantile(x, 1.0)[0] == 5.0

    def test_error_conditions_preserved(self):
        from brutus.utils.sampling import quantile

        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Quantiles must be between"):
            quantile(x, [1.1])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            quantile(x, 0.5, weights=np.ones(2))


# ---------------------------------------------------------------------------
# 5. draw_sar batched rewrite
# ---------------------------------------------------------------------------


def _draw_sar_reference(
    scales,
    avs,
    rvs,
    covs_sar,
    ndraws=500,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    rstate=None,
    max_attempts=10000,
):
    """Pre-rewrite draw_sar (per-sample numpy multivariate_normal loop),
    kept verbatim as the distributional reference implementation."""
    if rstate is None:
        rstate = np.random

    nsamps = len(scales)
    sdraws, adraws, rdraws = np.zeros((3, nsamps, ndraws))

    for i, (s, a, r, c) in enumerate(zip(scales, avs, rvs, covs_sar)):
        s_chunks, a_chunks, r_chunks = [], [], []
        n_collected = 0
        n_attempts = 0

        while n_collected < ndraws:
            n_attempts += 1
            s_mc, a_mc, r_mc = rstate.multivariate_normal([s, a, r], c, size=ndraws).T
            inbounds = (
                (s_mc >= 0.0)
                & (a_mc >= avlim[0])
                & (a_mc <= avlim[1])
                & (r_mc >= rvlim[0])
                & (r_mc <= rvlim[1])
            )
            s_mc, a_mc, r_mc = s_mc[inbounds], a_mc[inbounds], r_mc[inbounds]
            s_chunks.append(s_mc)
            a_chunks.append(a_mc)
            r_chunks.append(r_mc)
            n_collected += len(s_mc)
            if n_attempts >= max_attempts:
                break

        if n_collected > 0:
            s_temp = np.concatenate(s_chunks)
            a_temp = np.concatenate(a_chunks)
            r_temp = np.concatenate(r_chunks)
        else:
            s_temp = a_temp = r_temp = np.array([])

        if n_collected >= ndraws:
            sdraws[i] = s_temp[:ndraws]
            adraws[i] = a_temp[:ndraws]
            rdraws[i] = r_temp[:ndraws]
        else:
            sdraws[i, :n_collected] = s_temp[:n_collected]
            adraws[i, :n_collected] = a_temp[:n_collected]
            rdraws[i, :n_collected] = r_temp[:n_collected]
            sdraws[i, n_collected:] = s
            adraws[i, n_collected:] = a
            rdraws[i, n_collected:] = r

    return sdraws, adraws, rdraws


class TestDrawSarBatched:
    # Truncation is active for rows 0 (A_V near lower bound) and 2 (A_V near
    # upper bound); row 1 is essentially untruncated.
    scales = np.array([1.0, 0.5, 2.0])
    avs = np.array([0.15, 3.0, 5.85])
    rvs = np.array([3.1, 3.3, 3.2])
    covs = np.stack(
        [
            np.diag([0.04, 0.09, 0.25]),
            np.diag([0.01, 0.04, 0.09]),
            np.diag([0.09, 0.09, 0.16]),
        ]
    )

    def test_distributional_equivalence_with_reference(self):
        """New batched draws must follow the same truncated Gaussian as the
        old per-sample loop (same distribution, different RNG stream)."""
        from brutus.utils.sampling import draw_sar

        ndraws = 8000
        ref = _draw_sar_reference(
            self.scales,
            self.avs,
            self.rvs,
            self.covs,
            ndraws=ndraws,
            rstate=np.random.RandomState(1234),
        )
        new = draw_sar(
            self.scales,
            self.avs,
            self.rvs,
            self.covs,
            ndraws=ndraws,
            rstate=np.random.RandomState(4321),
        )
        qgrid = np.array([0.05, 0.16, 0.5, 0.84, 0.95])
        for comp, (ref_c, new_c) in enumerate(zip(ref, new)):
            for row in range(len(self.scales)):
                r, n = ref_c[row], new_c[row]
                # Means within Monte-Carlo error (~6 standard errors).
                se = r.std() / np.sqrt(ndraws)
                assert abs(r.mean() - n.mean()) < 6 * se, (comp, row)
                # Standard deviations within 5%.
                assert abs(r.std() - n.std()) < 0.05 * r.std(), (comp, row)
                # Quantiles within a small fraction of the spread.
                np.testing.assert_allclose(
                    np.quantile(n, qgrid),
                    np.quantile(r, qgrid),
                    atol=0.08 * r.std(),
                    err_msg=f"component {comp}, row {row}",
                )
                # Two-sample KS as an overall distributional check.
                pval = scipy.stats.ks_2samp(r, n).pvalue
                assert pval > 1e-3, (comp, row, pval)

    def test_bounds_respected(self):
        from brutus.utils.sampling import draw_sar

        s, a, r = draw_sar(
            self.scales,
            self.avs,
            self.rvs,
            self.covs,
            ndraws=2000,
            rstate=np.random.RandomState(5),
        )
        assert np.all(s >= 0.0)
        assert np.all((a >= 0.0) & (a <= 6.0))
        assert np.all((r >= 1.0) & (r <= 8.0))
        # Truncation actually engaged for rows 0 and 2 (draws pile near the
        # bound), confirming the rejection path was exercised.
        assert a[0].min() < 0.05
        assert a[2].max() > 5.95

    def test_mean_padding_semantics(self):
        """A distribution with (essentially) zero in-bounds acceptance warns
        and pads the unfilled slots with its mean values."""
        from brutus.utils.sampling import draw_sar

        scales = np.array([1.0, 1.0])
        avs = np.array([0.5, 50.0])  # second row: hopelessly out of bounds
        rvs = np.array([3.1, 3.1])
        covs = np.stack([np.diag([0.01, 0.01, 0.01])] * 2)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            s, a, r = draw_sar(
                scales,
                avs,
                rvs,
                covs,
                ndraws=10,
                max_attempts=5,
                rstate=np.random.RandomState(0),
            )
        msgs = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
        assert any("only collected 0/10" in m and "sample 1" in m for m in msgs)
        # Padded row holds the mean values; healthy row is fully sampled.
        np.testing.assert_array_equal(s[1], np.full(10, 1.0))
        np.testing.assert_array_equal(a[1], np.full(10, 50.0))
        np.testing.assert_array_equal(r[1], np.full(10, 3.1))
        assert np.all((a[0] >= 0.0) & (a[0] <= 6.0))
        assert len(np.unique(a[0])) > 1

    def test_rstate_compatibility(self):
        """np.random module, RandomState, and Generator must all work."""
        from brutus.utils.sampling import draw_sar

        for rstate in [None, np.random.RandomState(2), np.random.default_rng(2)]:
            s, a, r = draw_sar(
                self.scales,
                self.avs,
                self.rvs,
                self.covs,
                ndraws=50,
                rstate=rstate,
            )
            assert s.shape == a.shape == r.shape == (3, 50)
            assert np.all(np.isfinite(s))

    def test_reproducible_with_seed(self):
        from brutus.utils.sampling import draw_sar

        out1 = draw_sar(
            self.scales,
            self.avs,
            self.rvs,
            self.covs,
            ndraws=100,
            rstate=np.random.RandomState(99),
        )
        out2 = draw_sar(
            self.scales,
            self.avs,
            self.rvs,
            self.covs,
            ndraws=100,
            rstate=np.random.RandomState(99),
        )
        for x1, x2 in zip(out1, out2):
            np.testing.assert_array_equal(x1, x2)

    def test_list_inputs(self):
        """Sequence (non-ndarray) inputs are accepted, as before."""
        from brutus.utils.sampling import draw_sar

        s, a, r = draw_sar(
            [1.0, 1.1],
            [0.1, 0.2],
            [3.1, 3.3],
            [np.diag([0.01, 0.01, 0.1])] * 2,
            ndraws=20,
            rstate=np.random.RandomState(6),
        )
        assert s.shape == (2, 20)


if __name__ == "__main__":
    pytest.main([__file__])
