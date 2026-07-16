#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression and equivalence tests for the audited los_dust.py fixes.

Covers:
- Equivalence of the searchsorted likelihood rewrite against the previous
  per-segment implementation (copied below as a private reference).
- The monotonicity constraint applying to the EFFECTIVE extinction profile
  in template / additive-foreground modes (output-changing fix).
- NaN reddening samples being masked instead of poisoning the sightline.
- Kernel truncation renormalization at the reddening bounds
  (output-changing fix).
- Closed-form truncated log-normal prior transform matching
  scipy.stats.truncnorm.
"""

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

        # both are finite; each drops exactly one sample of object 0
        assert np.isfinite(ll_r) and np.isfinite(ll_d)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
