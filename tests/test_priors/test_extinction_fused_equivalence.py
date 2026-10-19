#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Equivalence tests for the fused 3D extinction-prior kernel.

`logp_extinction` dispatches large elementwise (avs, distance) arrays to a
numba kernel (`_extinction_prior_3d_fused`) that performs one binary search
per point instead of two full `np.interp` passes plus ~8 full-size
temporaries. The numpy branch remains the fallback / NUMBA_DISABLE_JIT
path; both must agree to floating-point roundoff on all edge cases
(boundary clamping, NaN profile segments, zero/negative spreads,
truncation on/off, systematic scatter, and the normalized uniform
fallback in masked bins).
"""

import numpy as np

from brutus.priors.extinction import _extinction_prior_3d_fused, logp_extinction


class Fake3DMap:
    def __init__(self, xp, mu, sig):
        self.xp, self.mu, self.sig = xp, mu, sig

    def query(self, coord):
        return (self.xp, self.mu, self.sig)


def _numpy_reference(avs, distance, xp, mu_p, sig_p, avlim, scatter=0.2):
    """Reference: scipy's truncated normal, uniform where the map is invalid."""
    from scipy.stats import truncnorm

    sig_p = np.asarray(sig_p, dtype=float)
    with np.errstate(invalid="ignore"):
        sig_p = np.where(sig_p >= 0, np.sqrt(sig_p**2 + scatter**2), np.nan)
    av_mean = np.interp(distance, xp, mu_p)
    av_err = np.interp(distance, xp, sig_p)
    valid = np.isfinite(av_mean) & np.isfinite(av_err) & (av_err > 0)
    mu = np.where(valid, av_mean, 0.0)
    sig = np.where(valid, av_err, 1.0)
    lo, hi = avlim
    lnprior = truncnorm.logpdf(avs, (lo - mu) / sig, (hi - mu) / sig, mu, sig)
    return np.where(valid, lnprior, -np.log(hi - lo))


class TestExtinctionFusedEquivalence:
    def setup_method(self):
        self.xp = np.linspace(0.05, 5.0, 120)
        frac = self.xp / self.xp[-1]
        self.mu = 0.005 + 0.075 * frac
        self.sig = 0.01 + 0.02 * frac

    def _compare(self, mu, sig, avlim=(0.0, 20.0), seed=0, N=5000, scatter=0.2):
        rng = np.random.default_rng(seed)
        avs = rng.uniform(0.0, 0.5, N)
        # Distances extending beyond both profile ends (boundary clamping)
        dist = rng.uniform(0.001, 8.0, N)
        # Include exact grid values (interp interval edges)
        dist[: self.xp.size] = self.xp

        m = Fake3DMap(self.xp, mu, sig)
        fused = logp_extinction(
            avs, m, None, distance=dist, avlim=avlim, scatter=scatter
        )
        expected = _numpy_reference(avs, dist, self.xp, mu, sig, avlim, scatter)
        assert fused.shape == avs.shape
        assert np.allclose(fused, expected, rtol=1e-12, atol=1e-13)
        # Also confirm the in-function numpy branch agrees
        # (return_components forces it)
        numpy_lp, _ = logp_extinction(
            avs,
            m,
            None,
            distance=dist,
            avlim=avlim,
            scatter=scatter,
            return_components=True,
        )
        assert np.allclose(fused, numpy_lp, rtol=1e-12, atol=1e-13)

    def test_clean_profile(self):
        self._compare(self.mu, self.sig)

    def test_map_mean_far_above_support_matches_scipy(self):
        # A map mean of 9 mag with avlim=(0, 6): the mass on the support is
        # about 1e-26, where an erf difference cancels to 0.
        self._compare(
            np.full_like(self.mu, 9.0), np.full_like(self.sig, 0.2), avlim=(0.0, 6.0)
        )

    def test_map_mean_beyond_erfc_underflow_matches_scipy(self):
        # 40 sigma out, the mass underflows to 0 even with erfc.
        self._compare(
            np.full_like(self.mu, 14.0), np.full_like(self.sig, 0.0), avlim=(0.0, 6.0)
        )

    def test_map_mean_with_subnormal_mass_matches_scipy(self):
        # 38 sigma out (spread 0.2 mag from the scatter), erfc is subnormal.
        self._compare(
            np.full_like(self.mu, 13.6), np.full_like(self.sig, 0.0), avlim=(0.0, 6.0)
        )

    def test_nan_profile_segments(self):
        mu = self.mu.copy()
        mu[40:60] = np.nan
        sig = self.sig.copy()
        sig[100:110] = np.nan
        self._compare(mu, sig, seed=1)

    def test_zero_and_negative_sigma(self):
        sig = self.sig.copy()
        sig[10:20] = 0.0
        sig[70:75] = -0.5
        self._compare(self.mu, sig, seed=2)
        # Without scatter, zero spreads are invalid too
        self._compare(self.mu, sig, seed=2, scatter=0.0)

    def test_raw_spreads_without_scatter(self):
        self._compare(self.mu, self.sig, scatter=0.0)

    def test_nan_segments_without_scatter(self):
        mu = self.mu.copy()
        mu[40:60] = np.nan
        self._compare(mu, self.sig, seed=4, scatter=0.0)

    def test_small_arrays_use_numpy_branch(self):
        # Below the dispatch threshold the numpy branch runs; results must
        # match the standalone reference too.
        rng = np.random.default_rng(3)
        avs = rng.uniform(0.0, 0.5, 50)
        dist = rng.uniform(0.001, 8.0, 50)
        m = Fake3DMap(self.xp, self.mu, self.sig)
        lp = logp_extinction(avs, m, None, distance=dist)
        expected = _numpy_reference(avs, dist, self.xp, self.mu, self.sig, (0.0, 20.0))
        assert np.allclose(lp, expected, rtol=1e-12, atol=1e-13)

    def test_kernel_direct_boundary_clamp(self):
        # Distances outside the map must clamp to the boundary profile
        # values (np.interp semantics)
        avs = np.array([0.1, 0.1])
        dist = np.array([1e-6, 100.0])
        out = _extinction_prior_3d_fused(
            avs, dist, self.xp, self.mu, self.sig, 0.0, 20.0, -np.log(20.0)
        )
        expected = _numpy_reference(
            avs, dist, self.xp, self.mu, self.sig, (0.0, 20.0), scatter=0.0
        )
        assert np.allclose(out, expected, rtol=1e-12, atol=1e-13)
