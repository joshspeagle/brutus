#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for verified audit fixes in brutus.priors.

Covers:
- logp_imf normalization when the mass range does not straddle mass_break
  (previously wrong by >2x for mass_min=1.0 and NaN for mass_min=5.0).
- convert_parallax_to_scale with p_err == 0 (previously ZeroDivisionError
  for Python floats / silent delta-function for numpy scalars).
- logn_disk solar normalization including R_smooth (previously the disk
  density at the solar position was 0.91, skewing the mixture fractions).
- feh/loga plain-array priors honored on the small-N numpy path and on
  fused-kernel fallback (previously silently dropped).
- Joint (feh, age) prior correctly factorized as a single mixture (the age
  term now uses metallicity-updated membership weights).
- logp_extinction raising for dustmap objects without query() (previously
  a pathlib.Path silently disabled the dust prior).
- logp_extinction normalized over the physical A_V support (truncated
  normal) instead of the full real line.
"""

import warnings
from unittest.mock import Mock

import numpy as np

# np.trapezoid is the NumPy 2.0 name for np.trapz; CI also tests NumPy 1.x
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pytest
from scipy import integrate
from scipy.special import logsumexp

from brutus.priors import (
    convert_parallax_to_scale,
    logn_disk,
    logn_halo,
    logp_extinction,
    logp_feh,
    logp_galactic_structure,
    logp_imf,
)
from brutus.priors.astrometric import SCALE_MAX, SCALE_MIN
from brutus.utils import truncnorm_logpdf
from brutus.utils.math import galactic_to_galactocentric_cyl


class TestIMFNormalization:
    """logp_imf must integrate to 1 over [mass_min, mass_max] regardless of
    where the range sits relative to mass_break."""

    @staticmethod
    def _integral(mass_min, mass_max, **kwargs):
        def pdf(m):
            return np.exp(
                logp_imf(
                    np.atleast_1d(m),
                    mass_min=mass_min,
                    mass_max=mass_max,
                    **kwargs,
                )
            )[0]

        val, _ = integrate.quad(pdf, mass_min, mass_max, limit=200)
        return val

    def test_defaults_normalized(self):
        assert np.isclose(self._integral(0.08, 100.0), 1.0, rtol=1e-6)

    def test_mass_min_above_break_normalized(self):
        # Old code: integral was 2.185 (norm_low integrated backwards)
        assert np.isclose(self._integral(1.0, 100.0), 1.0, rtol=1e-6)

    def test_mass_min_far_above_break_not_nan(self):
        # Old code: norm < 0 -> log(norm) = NaN for every mass
        logp = logp_imf(np.array([6.0, 10.0]), mass_min=5.0)
        assert np.all(np.isfinite(logp))
        assert np.isclose(self._integral(5.0, 100.0), 1.0, rtol=1e-6)

    def test_mass_max_below_break_normalized(self):
        assert np.isclose(self._integral(0.08, 0.3), 1.0, rtol=1e-6)

    def test_binary_normalization_high_mass_min(self):
        # Binary case squares the norm; must stay finite and consistent
        logp = logp_imf(np.array([2.0]), mgrid2=np.array([1.5]), mass_min=1.0)
        assert np.all(np.isfinite(logp))

    def test_invalid_mass_range_raises(self):
        with pytest.raises(ValueError):
            logp_imf(np.array([1.0]), mass_min=2.0, mass_max=1.0)
        with pytest.raises(ValueError):
            logp_imf(np.array([1.0]), mass_min=0.0)


class TestConvertParallaxToScaleZeroError:
    """p_err <= 0 must return the uninformative fallback, not crash."""

    def test_zero_error_python_float(self):
        # Old code: ZeroDivisionError
        s_mean, s_std = convert_parallax_to_scale(1.0, 0.0)
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX

    def test_zero_error_numpy_scalar(self):
        # Old code: silent (p**2, 0.0) delta-function prior
        s_mean, s_std = convert_parallax_to_scale(np.float64(1.0), np.float64(0.0))
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX

    def test_negative_error(self):
        s_mean, s_std = convert_parallax_to_scale(1.0, -0.1)
        assert s_mean == SCALE_MIN
        assert s_std == SCALE_MAX


class TestDiskSolarNormalization:
    """logn_disk must be exactly 0 at the solar position (like logn_halo),
    so the f_thick/f_halo mixture fractions are exact locally."""

    def test_thin_disk_unity_at_solar(self):
        # Old code: exp(logn) = 0.9117 (normalized at un-smoothed R_solar)
        logn = logn_disk(np.array([8.2]), np.array([0.025]))
        assert logn[0] == 0.0

    def test_thick_disk_unity_at_solar(self):
        # Old code: exp(logn) = 0.8867
        logn = logn_disk(np.array([8.2]), np.array([0.025]), R_scale=2.0, Z_scale=0.9)
        assert logn[0] == 0.0

    def test_halo_unity_at_solar(self):
        logn = logn_halo(np.array([8.2]), np.array([0.025]))
        assert np.isclose(logn[0], 0.0)

    def test_custom_solar_position(self):
        logn = logn_disk(np.array([8.5]), np.array([0.05]), R_solar=8.5, Z_solar=0.05)
        assert logn[0] == 0.0


def _make_labels(feh, loga):
    labels = np.empty(len(feh), dtype=[("feh", float), ("loga", float)])
    labels["feh"] = feh
    labels["loga"] = loga
    return labels


class TestFehLogaArrayPaths:
    """Plain feh/loga arrays must produce the same prior as structured
    labels on EVERY code path (fused kernel, small-N numpy, fallback)."""

    COORD = (90.0, 30.0)

    def test_small_n_arrays_match_labels(self):
        # Old code: arrays were silently ignored for len(dists) <= 1000
        N = 500
        d = np.linspace(0.1, 5.0, N)
        feh = np.full(N, -0.2)
        loga = np.full(N, 9.5)

        via_arrays = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        via_labels = logp_galactic_structure(
            d, self.COORD, labels=_make_labels(feh, loga)
        )
        spatial_only = logp_galactic_structure(d, self.COORD)

        assert np.array_equal(via_arrays, via_labels)
        assert not np.allclose(via_arrays, spatial_only)

    def test_large_n_arrays_match_labels(self):
        N = 2000
        rng = np.random.default_rng(0)
        d = np.linspace(0.1, 5.0, N)
        feh = rng.normal(-0.5, 0.5, N)
        loga = rng.uniform(8.5, 10.1, N)

        via_arrays = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        via_labels = logp_galactic_structure(
            d, self.COORD, labels=_make_labels(feh, loga)
        )
        assert np.array_equal(via_arrays, via_labels)

    def test_fused_kernel_failure_falls_back_with_priors(self, monkeypatch):
        # Old code: the numpy fallback dropped the feh/loga terms entirely
        import brutus.priors.galactic as gal

        def boom(*args, **kwargs):
            raise RuntimeError("simulated kernel failure")

        N = 2000
        rng = np.random.default_rng(1)
        d = np.linspace(0.1, 5.0, N)
        feh = rng.normal(-0.5, 0.5, N)
        loga = rng.uniform(8.5, 10.1, N)

        expected = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        spatial_only = logp_galactic_structure(d, self.COORD)

        monkeypatch.setattr(gal, "_galactic_prior_fused", boom)
        with pytest.warns(RuntimeWarning, match="falling back to numpy"):
            fallback = gal.logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)

        assert np.allclose(fallback, expected, rtol=1e-10, atol=1e-12)
        assert not np.allclose(fallback, spatial_only)

    def test_float32_and_list_inputs_normalized(self):
        N = 2000
        d = np.linspace(0.1, 5.0, N)
        feh64 = np.full(N, -0.3)
        loga64 = np.full(N, 9.2)
        ref = logp_galactic_structure(d, self.COORD, feh=feh64, loga=loga64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no fallback warnings allowed
            via_f32 = logp_galactic_structure(
                d, self.COORD, feh=feh64.astype(np.float32), loga=loga64
            )
            via_list = logp_galactic_structure(
                d, self.COORD, feh=list(feh64), loga=loga64
            )
        # float32 rounding perturbs values; -0.3 is inexact in binary
        assert np.allclose(via_f32, ref, rtol=1e-6)
        assert np.array_equal(via_list, ref)

    def test_shape_mismatch_raises(self):
        d = np.linspace(0.1, 5.0, 100)
        with pytest.raises(ValueError, match="feh"):
            logp_galactic_structure(d, self.COORD, feh=np.zeros(50))
        with pytest.raises(ValueError, match="loga"):
            logp_galactic_structure(d, self.COORD, loga=np.zeros((100, 2)))

    def test_array_coordinates_use_numpy_path(self):
        # Per-point sky coordinates cannot use the fused kernel (scalar
        # trig factors); must fall through to the numpy path silently.
        N = 1500
        rng = np.random.default_rng(2)
        d = np.linspace(0.1, 5.0, N)
        ell = rng.uniform(0.0, 360.0, N)
        b = rng.uniform(-90.0, 90.0, N)
        feh = rng.normal(-0.5, 0.5, N)
        loga = rng.uniform(8.5, 10.1, N)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            logp = logp_galactic_structure(d, (ell, b), feh=feh, loga=loga)
        assert logp.shape == (N,)
        assert np.all(np.isfinite(logp))


class TestJointFehAgePrior:
    """The (feh, age) prior must be the mixture sum_c w_c p(feh|c) p(age|c),
    not the product of marginal mixtures."""

    COORD = (90.0, 30.0)

    # Default component parameters of logp_galactic_structure
    COMPONENTS = [
        # (feh_mean, feh_sigma, ln_f)
        (-0.2, 0.3, 0.0),
        (-0.7, 0.4, np.log(0.04)),
        (-1.6, 0.5, np.log(0.005)),
    ]

    def _expected_joint(self, dists, feh, loga):
        """Independent reference: logsumexp_c[lnp_c + ln p(feh|c) + ln p(age|c)]."""
        ell, b = self.COORD
        R, Z = galactic_to_galactocentric_cyl(dists, ell, b)
        vol = 2.0 * np.log(dists + 1e-300)
        age = 10.0**loga / 1e9

        disk_scales = [(2.6, 0.3), (2.0, 0.9)]
        terms = []
        for idx, (fm, fsig, ln_f) in enumerate(self.COMPONENTS):
            if idx < 2:
                R_scale, Z_scale = disk_scales[idx]
                lnp_c = logn_disk(R, Z, R_scale=R_scale, Z_scale=Z_scale)
            else:
                lnp_c = logn_halo(R, Z)
            lnp_c = lnp_c + vol + ln_f

            lp_feh = logp_feh(feh, feh_mean=fm, feh_sigma=fsig)

            age_mean = 13.8 / (1.0 + np.exp((fm + 0.5) / 0.5))
            age_sigma = np.clip((13.8 - age_mean) / 2.0, 1.0, 4.0)
            lp_age = truncnorm_logpdf(
                age,
                (0.0 - age_mean) / age_sigma,
                (13.8 - age_mean) / age_sigma,
                loc=age_mean,
                scale=age_sigma,
            )
            terms.append(lnp_c + lp_feh + lp_age)

        return logsumexp(np.vstack(terms), axis=0)

    def test_numpy_path_matches_mixture(self):
        rng = np.random.default_rng(3)
        N = 200
        d = rng.uniform(0.05, 10.0, N)
        feh = rng.normal(-0.8, 0.7, N)
        loga = rng.uniform(8.0, 10.1, N)

        logp = logp_galactic_structure(d, self.COORD, labels=_make_labels(feh, loga))
        expected = self._expected_joint(d, feh, loga)
        assert np.allclose(logp, expected, rtol=1e-10, atol=1e-12)

    def test_fused_path_matches_mixture(self):
        rng = np.random.default_rng(4)
        N = 2000
        d = rng.uniform(0.05, 10.0, N)
        feh = rng.normal(-0.8, 0.7, N)
        loga = rng.uniform(8.0, 10.1, N)

        logp = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        expected = self._expected_joint(d, feh, loga)
        assert np.allclose(logp, expected, rtol=1e-10, atol=1e-12)

    def test_metal_poor_nearby_star_favors_old_age(self):
        # Old code: age weights ignored feh, so a nearby feh=-1.6 star was
        # weighted by the (thin-disk-dominated) position-only weights and
        # 12 Gyr was DISFAVORED vs 3 Gyr by ~e^-1.2.
        d = np.array([0.3, 0.3])
        labels = _make_labels(
            np.array([-1.6, -1.6]),
            np.array([np.log10(12e9), np.log10(3e9)]),
        )
        logp = logp_galactic_structure(d, self.COORD, labels=labels)
        assert logp[0] > logp[1]


class TestExtinctionDustmapValidation:
    """Objects without query() must raise; query failures must warn."""

    def test_pathlib_path_raises(self):
        # Old code: silently returned a uniform prior, disabling the dust
        # prior for the entire fit.
        from pathlib import Path

        with pytest.raises(TypeError, match="query"):
            logp_extinction(
                np.array([0.1, 0.5]),
                Path("map.h5"),
                np.array([10.0, 20.0]),
                distance=np.array([1.0, 1.0]),
            )

    def test_string_raises(self):
        with pytest.raises(TypeError, match="query"):
            logp_extinction(np.array([0.1]), "bayestar2019_v1.h5", None, distance=1.0)

    def test_query_failure_warns_and_returns_uniform(self):
        dustmap = Mock()
        dustmap.query.side_effect = TypeError("Query failed")
        with pytest.warns(RuntimeWarning, match="query failed"):
            logp = logp_extinction(np.array([0.1, 0.2]), dustmap, None)
        assert np.allclose(logp, 0.0)


class Fake3DMap:
    """Minimal Bayestar-like 3D dust map."""

    def __init__(self, xp, mu, sig):
        self.xp, self.mu, self.sig = xp, mu, sig

    def query(self, coord):
        return (self.xp, self.mu, self.sig)


class TestExtinctionTruncatedNormalization:
    """The Gaussian dust prior must be normalized over the A_V support."""

    def setup_method(self):
        xp = np.linspace(0.05, 5.0, 60)
        mu = 0.005 + 0.075 * (xp / xp[-1])
        sig = 0.01 + 0.02 * (xp / xp[-1])
        self.map = Fake3DMap(xp, mu, sig)

    def test_prior_integrates_to_one_on_support(self):
        # Old code: at d=0.1 kpc the integral over [0, 20] was ~0.69
        # (full-real-line normalization with mu ~ 0.5 sigma above zero).
        av_grid = np.linspace(0.0, 20.0, 200001)
        for d in (0.1, 2.0):
            lp = logp_extinction(
                av_grid, self.map, None, distance=np.full_like(av_grid, d)
            )
            integral = _trapezoid(np.exp(lp), av_grid)
            assert np.isclose(integral, 1.0, rtol=1e-3)

    def test_avlim_none_gives_plain_gaussian(self):
        avs = np.array([0.1, 0.2])
        dist = np.array([1.0, 1.0])
        lp = logp_extinction(avs, self.map, None, distance=dist, avlim=None)
        mu = np.interp(1.0, self.map.xp, self.map.mu)
        sig = np.interp(1.0, self.map.xp, self.map.sig)
        expected = -0.5 * ((avs - mu) ** 2 / sig**2 + np.log(2 * np.pi * sig**2))
        assert np.allclose(lp, expected)

    def test_truncation_is_distance_dependent(self):
        # The truncation correction -log(Z(d)) must differ between a
        # near-boundary profile point and a well-separated one.
        avs = np.array([0.0, 0.0])
        dist = np.array([0.05, 5.0])
        lp_trunc = logp_extinction(avs, self.map, None, distance=dist)
        lp_plain = logp_extinction(avs, self.map, None, distance=dist, avlim=None)
        corr = lp_trunc - lp_plain
        assert corr[0] > corr[1] > 0.0

    def test_simple_2tuple_map_also_truncated(self):
        dustmap = Mock()
        dustmap.query.return_value = (0.05, 0.1)  # mu within 0.5 sigma of 0
        av_grid = np.linspace(0.0, 20.0, 200001)
        lp = logp_extinction(av_grid, dustmap, None)
        integral = _trapezoid(np.exp(lp), av_grid)
        assert np.isclose(integral, 1.0, rtol=1e-3)
