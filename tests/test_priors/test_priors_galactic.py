"""Tests of brutus.priors.galactic."""

import os
import warnings

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import logsumexp

from brutus.priors import galactic as gal
from brutus.priors import kinematic as kin
from brutus.priors import logn_disk, logn_halo, logp_galactic_structure
from brutus.utils.math import (
    R_SOLAR,
    Z_SOLAR,
    galactic_to_galactocentric,
    galactic_to_galactocentric_cyl,
)

_QUAD = dict(epsabs=1e-12, epsrel=1e-12, limit=500)


# v_c at the Sun: the default scale of the measured rotation curve.
_VC_SUN = kin.solar_velocity(R_SOLAR)[1] - kin.V_PEC_SUN
_ROT = (R_SOLAR, _VC_SUN)
# The tabulated potential scales with v_c at the Sun squared.
_PHI_SCALE = (_VC_SUN / kin._V_VC0) ** 2


def _vc(R):
    """The rotation curve and d ln v_c / d ln R (tested in the kinematic
    tests)."""
    return kin._circular_velocity(float(R), *_ROT)


def _mean_L(R, age, feh):
    """Mean L_z of a thin-disk population at R in the midplane, from the
    kinematic kernels (tested in the kinematic tests), with the default
    thin-disk density (R_scale = 2.74, R_smooth = 2)."""
    vc, dlnvc = _vc(R)
    L_refs = kin._thin_disk_L_refs(*_ROT)
    dlnnu = -R * R / (2.74 * np.hypot(R, 2.0))
    sig_R0 = kin._thin_disk_sigma0(age, feh)[0]
    fade = kin._thin_disk_fade(R)
    L = kin._thin_disk_mean_L(R, 0.0, sig_R0, vc, dlnvc, *L_refs, dlnnu, fade)
    return L, L_refs


def _kz_column(i):
    """K_z(z) at table row i, a cubic spline through the tabulated values
    in u = asinh(z / z_scale): independent of the production Hermite
    interpolation and of the tabulated integrals."""
    u = np.arange(kin._V_KZ.shape[1]) * kin._V_DU
    return CubicSpline(u, kin._V_KZ[i])


def _fz(t):
    """The thin-disk sigma_z height factor, written out: 1 + gamma_z |z|,
    with |z| held at 2 kpc through a quadratic blend over 1.5-2.5 kpc."""
    top, w = kin.THIN_DISK_Z_MAX, kin.THIN_DISK_Z_EDGE
    return 1 + kin.THIN_DISK_GAMMA_Z[1] * _clamp(t, -1.0, top, w)


def _column_integral(i, abs_z, weight):
    """int_0^|z| K_z(t) weight(t) dt along table row i, by quadrature."""
    spline = _kz_column(i)

    def integrand(t):
        return spline(np.arcsinh(t / kin._V_ZS)) * weight(t)

    return quad(integrand, 0.0, abs_z, **_QUAD)[0]


_SUMMARY = gal.galactic_prior_summary()
_F_THICK, _F_HALO = _SUMMARY["f_thick_solar"], _SUMMARY["f_halo_solar"]
_NO_SYSTEMATICS = dict(
    feh_systematic=(0.0, 0.0, 0.0), age_systematic=(0.0, 0.0, 0.0), age_floor=0.0
)


def _x_bar_reference(x, y, z):
    """The X-shaped bar of Sormani et al. (2022), eqs. 2-5, in Msun/kpc^3,
    vectorized, in the bar frame."""
    rho, x0, y0, z0, c_par, c_perp, m, r_cut, alpha, c, n, xc, yc = gal._BAR_X
    x, y, z = np.broadcast_arrays(*map(np.asarray, (x, y, z)))
    a = (
        (np.abs(x / x0) ** c_perp + np.abs(y / y0) ** c_perp) ** (c_par / c_perp)
        + np.abs(z / z0) ** c_par
    ) ** (1 / c_par)
    a_pm = [np.hypot((x + sign * c * z) / xc, y / yc) for sign in (1, -1)]
    return (
        rho
        / np.cosh(a**m)
        * np.exp(-((np.sqrt(x**2 + y**2 + z**2) / r_cut) ** 2))
        * (1 + alpha * sum(np.exp(-(ap**n)) for ap in a_pm))
    )


def _bar_reference(x, y, z):
    """The bar of Sormani et al. (2022): the X-shaped bar and the two long
    bars of eqs. 6-8."""
    out = _x_bar_reference(x, y, z)
    R = np.hypot(x, y)
    with np.errstate(divide="ignore", over="ignore"):
        for rho, x0, y0, h, c_perp, n, r_out, r_in, n_out, n_in in gal._BAR_LONG:
            a = (np.abs(x / x0) ** c_perp + np.abs(y / y0) ** c_perp) ** (1 / c_perp)
            out = out + rho / np.cosh(z / h) ** 2 * np.exp(
                -(a**n) - (R / r_out) ** n_out - (r_in / R) ** n_in
            )
    return out


def _bar_frame(d, ell, b, angle=gal.BAR_ANGLE):
    """Bar-frame position of a star at distance d toward (l, b)."""
    X, Y, Z = galactic_to_galactocentric(d, ell, b)
    ar = np.radians(angle)
    return X * np.cos(ar) - Y * np.sin(ar), X * np.sin(ar) + Y * np.cos(ar), Z


def _ln_bar_labels(feh, age, feh_sys, age_frac, age_floor=0.0):
    """ln p([Fe/H], age) of the default bar populations, with the given
    systematics."""
    out = []
    for fm, fs, am, asd in ((-0.31, 0.31, 11.0, 1.5), (0.26, 0.20, 8.0, 3.0)):
        sf = np.hypot(fs, feh_sys)
        sa = np.sqrt(asd**2 + (age_frac * am) ** 2 + age_floor**2)
        out.append(
            np.log(0.5)
            + stats.norm.logpdf(feh, fm, sf)
            + stats.truncnorm.logpdf(age, -am / sa, (13.8 - am) / sa, am, sa)
        )
    return np.logaddexp(*out)


# Keys of the thin-disk tables with the default parameters and systematics:
# R_solar, v_c at the Sun, R_thin, Rs_thin and sig_birth in kpc.
_THIN_KEY = (R_SOLAR, _VC_SUN, 2.74, 2.0, np.hypot(0.03, 0.1) / 0.075)


def _clamp(x, lo, hi, width):
    """The quadratic edge blend of kinematic._soft_clamp, written out."""
    if abs(x - lo) < width:
        return lo + (x - lo + width) ** 2 / (4 * width)
    if abs(x - hi) < width:
        return hi - (hi + width - x) ** 2 / (4 * width)
    return min(max(x, lo), hi)


class TestGalacticPriors:
    """Test Galactic structure and stellar population priors."""

    def test_logn_disk_basic(self):
        """Test basic disk number density evaluation."""
        distances = np.array([0.5, 1.0, 2.0, 5.0])  # kpc
        zs = np.array([0.0, 0.1, 0.5, 1.0])  # kpc above plane

        logn = logn_disk(distances, zs)

        # Should be finite
        assert np.all(np.isfinite(logn))

        # Should decrease with height
        logn_heights = logn_disk(np.array([1.0]), np.array([0.0, 0.2, 0.5, 1.0]))
        assert np.all(np.diff(logn_heights) < 0)  # Decreasing

    def test_logn_disk_custom_parameters(self):
        """Test disk with custom scale parameters."""
        distances = np.array([1.0, 2.0])
        zs = np.array([0.1, 0.1])

        logn_default = logn_disk(distances, zs)
        logn_custom = logn_disk(distances, zs, R_scale=3.0, Z_scale=0.4)

        # Should produce different results
        assert not np.allclose(logn_default, logn_custom)

    def test_logn_halo_basic(self):
        """Test basic halo number density evaluation."""
        distances = np.array([0.5, 1.0, 5.0, 20.0])  # kpc
        zs = np.array([0.0, 1.0, 5.0, 10.0])  # kpc

        logn = logn_halo(distances, zs)

        # Should be finite
        assert np.all(np.isfinite(logn))

    def test_logn_halo_custom_parameters(self):
        """Test halo with custom parameters."""
        distances = np.array([5.0, 10.0])
        zs = np.array([2.0, 2.0])

        logn_default = logn_halo(distances, zs)
        logn_custom = logn_halo(distances, zs, eta_inner=3.5, q=0.6)

        # Should produce different results
        assert not np.allclose(logn_default, logn_custom)

    def test_logp_galactic_structure_basic(self):
        """Test combined Galactic structure prior."""
        # Create mock coordinates
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")
        distances = np.array([1.0, 2.0, 5.0])  # kpc

        logp = logp_galactic_structure(distances, coord)

        # Should be finite
        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_logp_galactic_structure_with_components(self):
        """Test Galactic structure prior returning components."""
        coord = SkyCoord(ra=90.0, dec=45.0, unit="deg")
        distances = np.array([0.5, 1.5, 3.0])

        result = logp_galactic_structure(distances, coord, return_components=True)
        logp_total, components = result

        # Check components structure
        assert "number_density" in components
        assert isinstance(components["number_density"], list)
        assert len(components["number_density"]) == 4  # thin, thick, halo, bar

        # Each component should have right length
        for comp in components["number_density"]:
            assert len(comp) == len(distances)
            assert np.all(np.isfinite(comp))


class TestDiskSolarNormalization:
    """logn_disk must be exactly 0 at the solar position (like logn_halo),
    so the mixture fractions are exact at the Sun."""

    def test_thin_disk_unity_at_solar(self):
        # Old code: exp(logn) = 0.9117 (normalized at un-smoothed R_solar)
        logn = logn_disk(np.array([R_SOLAR]), np.array([Z_SOLAR]))
        assert logn[0] == 0.0

    def test_thick_disk_unity_at_solar(self):
        # Old code: exp(logn) = 0.8867
        logn = logn_disk(
            np.array([R_SOLAR]), np.array([Z_SOLAR]), R_scale=2.0, Z_scale=0.9
        )
        assert logn[0] == 0.0

    def test_halo_unity_at_solar(self):
        logn = logn_halo(np.array([R_SOLAR]), np.array([Z_SOLAR]))
        assert np.isclose(logn[0], 0.0)

    def test_custom_solar_position(self):
        logn = logn_disk(np.array([8.5]), np.array([0.05]), R_solar=8.5, Z_solar=0.05)
        assert logn[0] == 0.0


def _make_labels(feh, loga, dtype=float):
    labels = np.empty(len(feh), dtype=[("feh", dtype), ("loga", dtype)])
    labels["feh"] = feh
    labels["loga"] = loga
    return labels


class TestFehLogaArrayPaths:
    """Plain feh/loga arrays must produce the same prior as structured
    labels."""

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

    @pytest.mark.parametrize("N", [500, 2000])
    def test_float32_labels_match_float64_arrays(self, N):
        """Regression (#250): float32 `labels` fields were used as given, so
        the numpy path, and the fused path without JIT, computed in float32."""
        rng = np.random.default_rng(4)
        d = rng.uniform(0.1, 10.0, N)
        labels = _make_labels(
            rng.uniform(-2.0, 0.5, N), rng.uniform(8.5, 10.1, N), dtype=np.float32
        )
        via_labels = logp_galactic_structure(d, self.COORD, labels=labels)
        via_arrays = logp_galactic_structure(
            d,
            self.COORD,
            feh=labels["feh"].astype(np.float64),
            loga=labels["loga"].astype(np.float64),
        )
        assert np.array_equal(via_labels, via_arrays)

    def test_float32_and_list_inputs_normalized(self):
        N = 2000
        d = np.linspace(0.1, 5.0, N)
        feh64 = np.full(N, -0.3)
        loga64 = np.full(N, 9.2)
        ref = logp_galactic_structure(d, self.COORD, feh=feh64, loga=loga64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
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

    def test_labels_of_another_shape_raise(self):
        """A `labels` of the right size but another shape would pair each
        distance with the wrong entry."""
        d = np.ones((2, 3))
        labels = _make_labels(np.zeros(6), np.full(6, 9.5)).reshape(3, 2)
        with pytest.raises(ValueError, match="labels"):
            logp_galactic_structure(d, self.COORD, labels=labels)

    def test_per_point_sky_coordinates_give_finite_priors_without_warnings(self):
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


class TestMixedArrayAndLabels:
    """A plain feh/loga array must override only its own field: any field
    not given as an array is sourced from a structured `labels`, so mixing
    the two call styles cannot silently drop a prior term."""

    COORD = (90.0, 30.0)

    @pytest.mark.parametrize("N", [500, 2000])
    def test_feh_array_plus_labels_with_loga_keeps_age_prior(self, N):
        rng = np.random.default_rng(1)
        d = np.linspace(0.1, 5.0, N)
        feh = rng.normal(-0.5, 0.3, N)
        loga = rng.uniform(8.5, 10.1, N)

        mixed = logp_galactic_structure(
            d, self.COORD, feh=feh, labels=_make_labels(feh, loga)
        )
        both_arrays = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        feh_only = logp_galactic_structure(d, self.COORD, feh=feh)

        # the age prior from labels must be applied...
        assert np.array_equal(mixed, both_arrays)
        # ...and it must actually change the result
        assert not np.allclose(mixed, feh_only)

    def test_loga_array_plus_labels_with_feh_keeps_feh_prior(self):
        N = 800
        rng = np.random.default_rng(2)
        d = np.linspace(0.1, 5.0, N)
        feh = rng.normal(-0.5, 0.3, N)
        loga = rng.uniform(8.5, 10.1, N)

        mixed = logp_galactic_structure(
            d, self.COORD, loga=loga, labels=_make_labels(feh, loga)
        )
        both_arrays = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        assert np.array_equal(mixed, both_arrays)


class TestGalacticCoverage:
    """Tests to improve galactic.py code coverage."""

    def test_galactic_structure_tuple_coordinates(self):
        """Test galactic structure with tuple coordinates (not SkyCoord)."""
        distances = np.array([1.0, 2.0])
        coord_tuple = (180.0, 0.0)  # (l, b) in degrees

        logp = logp_galactic_structure(distances, coord_tuple)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_invalid_labels(self):
        """Test galactic structure with invalid/missing label fields."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with different field name
        labels = np.array([(0.0,), (-0.5,)], dtype=[("metallicity", "f4")])

        # Should handle missing fields gracefully
        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_components_with_labels(self):
        """Test galactic structure returning components with labels."""
        distances = np.array([1.0])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        # Create labels with both metallicity and age
        labels = np.array([(0.0, 9.0)], dtype=[("feh", "f4"), ("loga", "f4")])

        logp, components = logp_galactic_structure(
            distances, coord, labels=labels, return_components=True
        )

        # The terms of the prior are the components' densities plus the ln
        # densities of their labels.
        terms = np.add(components["number_density"], components["labels"])
        np.testing.assert_allclose(logp, np.logaddexp.reduce(terms), rtol=1e-12)


@pytest.mark.parametrize("feh", [None, np.zeros(3)])
def test_labelled_ages_out_of_reach_of_thin_disk_stars_have_no_density(feh):
    # Regression: with a spread of 0.05 Gyr, no true age below the end of
    # thin-disk star formation (12 Gyr) reaches a labelled 13 Gyr, and the
    # sum over true ages had an empty range that gave NaN.
    _, comps = logp_galactic_structure(
        np.full(3, 1.0),
        (30.0, 10.0),
        feh=feh,
        loga=np.full(3, np.log10(13e9)),
        age_systematic=(0.0, 0.15, 0.15),
        age_floor=0.05,
        return_components=True,
    )
    assert np.all(np.isfinite(comps["labels"])) and np.all(comps["labels"][0] < -1e299)


@pytest.mark.parametrize("feh", [None, np.zeros(3)])
def test_ages_beyond_the_age_of_the_universe_have_zero_prior(feh):
    lnp = logp_galactic_structure(
        np.full(3, 1.0), (30.0, 10.0), feh=feh, loga=np.full(3, 10.15)
    )
    np.testing.assert_array_equal(lnp, -1e300)


@pytest.mark.parametrize("names", [("feh",), ("loga",), ("feh", "loga")])
def test_labels_on_the_trailing_axis_match_tiled_labels(names):
    # One label per model, broadcast over 40 draws, as logpost_grid passes
    # them; the terms of each label are computed once.
    rng = np.random.default_rng(5)
    d = rng.uniform(0.05, 8.0, (40, 300))
    values = {"feh": rng.uniform(-2.0, 0.5, 300), "loga": rng.uniform(8.0, 10.1, 300)}
    per_model = {k: values[k] for k in names}
    tiled = {k: np.tile(values[k], 40) for k in names}
    got = logp_galactic_structure(d, (30.0, 10.0), **per_model)
    ref = logp_galactic_structure(d.ravel(), (30.0, 10.0), **tiled)
    assert got.shape == d.shape
    np.testing.assert_array_equal(got.ravel(), ref)


def test_tabulated_ln_ndtr_matches_scipy():
    from scipy.special import log_ndtr

    t = np.concatenate(
        [np.linspace(-40.0, 9.0, 20001), np.linspace(-37.01, -36.99, 101)]
    )
    got = np.array([gal._ln_ndtr(v) for v in t])
    np.testing.assert_allclose(got, log_ndtr(t), rtol=1e-12, atol=3e-9)


class TestLabelsValidation:
    @pytest.mark.parametrize("n", [10, 2000])
    def test_labels_length_mismatch_raises(self, n):
        labels = np.zeros(5, dtype=[("feh", float), ("loga", float)])
        with pytest.raises(ValueError, match="labels"):
            logp_galactic_structure(
                np.linspace(0.1, 5.0, n), (30.0, 10.0), labels=labels
            )

    def test_scalar_distance_with_labels_is_finite(self):
        labels = np.array([(0.0, 9.5)], dtype=[("feh", float), ("loga", float)])
        with_labels = logp_galactic_structure(1.0, (90.0, 30.0), labels=labels)
        with_feh = logp_galactic_structure(1.0, (90.0, 30.0), feh=[0.0])
        assert np.all(np.isfinite(with_labels))
        assert np.isfinite(with_feh)
        assert np.ndim(logp_galactic_structure(1.0, (90.0, 30.0))) == 0

    @pytest.mark.parametrize("shape", [(3, 2)])
    def test_labels_matching_multidimensional_dists_broadcast(self, shape):
        n = int(np.prod(shape))
        labels = np.zeros(shape, dtype=[("feh", float), ("loga", float)])
        labels["loga"] = 9.5
        lnp = logp_galactic_structure(np.full(shape, 1.0), (90.0, 30.0), labels=labels)
        flat = logp_galactic_structure(
            np.full(n, 1.0), (90.0, 30.0), labels=labels.ravel()
        )
        assert lnp.shape == shape
        np.testing.assert_allclose(lnp.ravel(), flat, rtol=1e-12)

    def test_single_label_record_with_scalar_distance(self):
        labels = np.array([(0.0, 9.5)], dtype=[("feh", float), ("loga", float)])
        lnp = logp_galactic_structure(1.0, (90.0, 30.0), labels=labels[0])
        assert np.isfinite(lnp)


class TestVerticalJeans:
    """The disk vertical profiles from the vertical Jeans equation."""

    def test_potential_reproduces_the_rebuilt_model(self):
        # K_z(R0, 1.1 kpc) / 2 pi G of the rebuilt Binney & Vasiliev (2024)
        # model is 63.9 Msun/pc^2 (66.4 in their eq. 15).
        i = round(1.0 / kin._V_DX)
        kz = _kz_column(i)(np.arcsinh(1.1 / kin._V_ZS))
        assert abs(kz / (2 * np.pi * 4.30091e-6) * 1e-6 - 63.9) < 0.1

    @pytest.mark.parametrize("abs_z", [0.0004, 0.05, 0.3, 1.0, 1.8, 2.2, 4.0, 9.9])
    @pytest.mark.parametrize("i", [0, 40, 80, 81, 200])
    def test_vertical_tables_integrate_their_force(self, abs_z, i):
        # At table radii both integrals agree with quadrature of K_z.
        x = i * kin._V_DX
        _, jeans, dphi = kin._vertical_terms(x, abs_z)
        ref = _column_integral(i, abs_z, lambda t: 1.0)
        assert np.isclose(dphi, ref, rtol=1e-4, atol=1e-3)
        ref = _column_integral(i, abs_z, lambda t: 1.0 / _fz(t) ** 2)
        assert np.isclose(jeans, ref, rtol=1e-4, atol=1e-3)

    def test_thick_disk_column_has_no_kinks_in_radius(self):
        # The clamped potential of #284 bent the column height by 0.046 kpc
        # per 0.25 kpc step at 4 and 9 kpc. The largest bend is now 0.005,
        # at 13 kpc, in the blend where the dispersions start to hold. The
        # radii start outside the inner blend, which ends at 3.8 kpc.
        x = np.arange(3.75, 20.01, 0.25) / R_SOLAR
        var = [kin.THICK_DISK_SIGMA[2] ** 2 * kin._thick_disk_scale(v)[2] for v in x]
        ln_s = np.log(np.array(var) / _PHI_SCALE)
        h = [np.exp(gal._ln_column(v, k, gal._THICK_COLUMN)) for v, k in zip(x, ln_s)]
        assert np.abs(np.diff(h, 2)).max() < 0.01

    @pytest.mark.parametrize("feh", [-1.0, 0.0])
    def test_thin_disk_falls_faster_than_z_squared_at_height(self, feh):
        # With sigma_z growing as 1 + gamma_z |z| without bound, the Jeans
        # integral converges and d ln nu / d ln |z| tends to -2. Holding |z|
        # at 2 kpc makes it fall far faster over 15-30 kpc.
        d = np.array([15.0, 30.0])
        _, comps = logp_galactic_structure(
            d,
            (0.0, 90.0),
            feh=np.full(2, feh),
            loga=np.full(2, 9.7),
            return_components=True,
        )
        thin = comps["number_density"][0] + comps["labels"][0] - 2 * np.log(d)
        assert np.diff(thin)[0] / np.log(2.0) < -10.0

    @pytest.mark.parametrize("x", [0.137, 0.5046, 1.00625, 2.31, 3.6])
    @pytest.mark.parametrize("abs_z", [0.2, 1.0, 3.0, 8.0])
    def test_vertical_terms_between_table_radii_match_quadrature(self, x, abs_z):
        # Between table radii both integrals agree with quadrature of a
        # bicubic spline of K_z in (x, u) to 2e-5; a one-row offset or a
        # wrong weight in the Catmull-Rom step misses by about 1e-2.
        u = np.arange(kin._V_KZ.shape[1]) * kin._V_DU
        rows = np.arange(kin._V_KZ.shape[0]) * kin._V_DX
        spline = RectBivariateSpline(rows, u, kin._V_KZ)

        def kz(t):
            return spline(x, np.arcsinh(t / kin._V_ZS))[0, 0]

        _, jeans, dphi = kin._vertical_terms(x, abs_z)
        ref = quad(kz, 0.0, abs_z, **_QUAD)[0]
        assert np.isclose(dphi, ref, rtol=1e-4)
        ref = quad(lambda t: kz(t) / _fz(t) ** 2, 0.0, abs_z, **_QUAD)[0]
        assert np.isclose(jeans, ref, rtol=1e-4)

    @pytest.mark.parametrize(
        "x, abs_z", [(np.nan, 1.0), (1.0, np.nan), (np.inf, 1.0), (1.0, np.inf)]
    )
    def test_non_finite_positions_give_nan(self, x, abs_z):
        # int() of a NaN or infinite index crashed the jitted kernel.
        assert np.isnan(kin._vertical_terms(x, abs_z)).all()

    def test_vertical_terms_continue_beyond_the_table_with_its_force(self):
        x = 1.0
        top = kin._V_ZS * np.sinh((kin._V_KZ.shape[1] - 1) * kin._V_DU)
        k = kin._V_KZ[round(x / kin._V_DX), -1]
        _, j0, p0 = kin._vertical_terms(x, top)
        _, j1, p1 = kin._vertical_terms(x, top + 5.0)
        assert np.isclose(p1 - p0, 5.0 * k, rtol=1e-12)
        assert np.isclose(j1 - j0, 5.0 * k / _fz(top) ** 2, rtol=1e-12)

    @pytest.mark.parametrize("abs_z", [0.1, 1.0, 3.0])
    def test_vertical_terms_have_a_continuous_radial_slope(self, abs_z):
        # Catmull-Rom interpolation in x: at a node, the one-sided slopes
        # differ by the curvature times the step, so the gap shrinks with
        # the step; a jump in the slope would not.
        def gap(x, h):
            f0, fl, fr = (kin._vertical_terms(v, abs_z)[2] for v in (x, x - h, x + h))
            return abs((fr - f0) - (f0 - fl)) / h

        for i in (1, 80, 160, 300):
            x = i * kin._V_DX
            assert gap(x, 1e-7) < 0.02 * gap(x, 1e-5) + 1e-3

    def test_feh_outside_the_fitted_range_takes_the_edge_value(self):
        for feh, edge in [(-2.5, -1.0), (1.0, 0.5), (3.0, 0.5)]:
            assert kin._thin_disk_sigma0(4.0, feh) == kin._thin_disk_sigma0(4.0, edge)

    @staticmethod
    def _scale_height(lnnu, z):
        return -1.0 / np.polyfit(z, lnnu, 1)[0]

    def test_disk_scale_heights_match_star_counts(self):
        # New result: scale heights near 0.3 and 0.9 kpc (Juric et al. 2008)
        # follow from the dispersions and the potential.
        d = np.linspace(0.3, 2.0, 35)
        _, comps = logp_galactic_structure(d, (0.0, 90.0), return_components=True)
        thin, thick = (c - 2 * np.log(d) for c in comps["number_density"][:2])
        near, far = d < 1.5, d > 0.5
        assert 0.25 < self._scale_height(thin[near], d[near]) < 0.4
        assert 0.8 < self._scale_height(thick[far], d[far]) < 1.2

    def test_younger_thin_disk_stars_lie_closer_to_the_plane(self):
        d = np.linspace(0.2, 1.5, 20)
        heights = []
        for loga in (9.0, 9.5, 10.0):
            comps = logp_galactic_structure(
                d,
                (0.0, 90.0),
                loga=np.full(d.size, loga),
                return_components=True,
            )[1]
            thin = comps["number_density"][0] + comps["labels"][0] - 2 * np.log(d)
            heights.append(self._scale_height(thin, d))
        assert heights[0] < heights[1] < heights[2]


class TestVerticalInputs:
    """Inputs of the vertical model in logp_galactic_structure."""

    def test_labels_broadcast_with_array_coordinates(self):
        # One distance seen along five sightlines: each entry must equal
        # the prior along its own sightline.
        ell = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        kw = dict(loga=np.array([9.0]), feh=np.array([0.0]))
        lnp = logp_galactic_structure(np.array([1.0]), (ell, np.full(5, 30.0)), **kw)
        each = [
            logp_galactic_structure(np.array([1.0]), (x, 30.0), **kw)[0] for x in ell
        ]
        np.testing.assert_allclose(lnp, each, rtol=1e-12)

    @pytest.mark.parametrize("name", ["feh", "loga"])
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_labels_raise(self, name, value):
        kw = dict(feh=np.array([0.0]), loga=np.array([9.0]))
        kw[name] = np.array([value])
        with pytest.raises(ValueError, match=name):
            logp_galactic_structure(np.array([1.0]), (0.0, 30.0), **kw)

    @pytest.mark.parametrize("name", ["v_circ_solar", "thick_sigma_z"])
    @pytest.mark.parametrize("value", [0.0, -2.0])
    def test_non_positive_vertical_parameters_raise(self, name, value):
        with pytest.raises(ValueError, match=name):
            logp_galactic_structure(np.array([1.0]), (0.0, 30.0), **{name: value})


def _sigma_z(R, age, feh):
    """Midplane sigma_z of a thin-disk population, with the dispersion
    relations written out and the mean L_z of the kinematic kernels."""
    L, (L_sun, L_clamp) = _mean_L(R, age, feh)
    L = _clamp(L, *L_clamp)
    a, lam = kin.THIN_DISK_ALPHA_L[1], kin.THIN_DISK_LAMBDA_L[1]
    f_L = (a * (L / L_sun) ** 2 + np.exp(-(L - L_sun) / lam)) / (1 + a)
    return (
        kin.THIN_DISK_SIGMA0[1]
        * ((age + 0.1) / 10.1) ** kin.THIN_DISK_BETA[1]
        * (1 + kin.THIN_DISK_GAMMA_FEH[1] * np.clip(feh, -1.0, 0.5))
        * f_L
    )


def _ln_profile(x, s, thin):
    """ln nu(|z|) with nu = 1 at the midplane, and ln of its column to
    100 kpc by adaptive quadrature, for s = sigma_z^2 / phi."""

    def ln_nu(z):
        ln_fz2, jeans, dphi = kin._vertical_terms(x, z)
        return -ln_fz2 - jeans / s if thin else -dphi / s

    edges = [0.0, 0.05, 0.3, 1.0, 1.5, 2.5, 5.0, 20.0, 100.0]
    column = sum(
        quad(lambda z: np.exp(ln_nu(z)), a, b, epsrel=1e-10, limit=200)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )
    return ln_nu, np.log(column)


class TestColumnsAndNormalization:
    """Each disk population's density is its surface density times
    nu(z) / 2H, and the components are normalized at the Sun."""

    @pytest.mark.parametrize("thin", [True, False])
    @pytest.mark.parametrize(
        "x, sigma", [(0.031, 4.0), (0.613, 12.5), (1.007, 31.0), (2.55, 77.0)]
    )
    def test_column_tables_match_quadrature(self, thin, x, sigma):
        # Off the table nodes in x and ln s.
        s = sigma**2 / _PHI_SCALE
        table = gal._THIN_COLUMN if thin else gal._THICK_COLUMN
        expected = _ln_profile(x, s, thin)[1]
        assert np.isclose(gal._ln_column(x, np.log(s), table), expected, atol=2e-4)

    def test_components_match_a_direct_computation(self):
        # Surface density, labels and nu / 2H from the written-out
        # dispersions and quadrature columns; the thin-disk label density
        # is tested against its defining integral below.
        rng = np.random.default_rng(3)
        d = rng.uniform(0.2, 6.0, 6)
        feh = rng.uniform(-1.2, 0.3, 6)
        age = rng.uniform(0.5, 11.5, 6)
        coord = (60.0, 25.0)
        _, comps = logp_galactic_structure(
            d,
            coord,
            feh=feh,
            loga=np.log10(age * 1e9),
            return_components=True,
            **_NO_SYSTEMATICS,
        )
        got = np.add(comps["number_density"], comps["labels"]) - 2 * np.log(d)
        R, Z = galactic_to_galactocentric_cyl(d, *coord)
        summary = gal.galactic_prior_summary(feh_systematic=(0.0, 0.0, 0.0))
        ln_d0 = -np.log(2 * summary["thin_column_height"])
        f_halo = summary["f_halo_solar"]
        thick_sn = gal._skew_normal_args("thick_feh", (-0.45, 0.22, -0.5), 0.0)
        halo_sn = gal._skew_normal_args("halo_feh", (-1.18, 0.28, -0.67), 0.0)
        # The halo core, normalized with the 10^[Fe/H] tail below -1.95.
        halo_norm = stats.skewnorm.sf(
            -1.95, halo_sn[2], *halo_sn[:2]
        ) + stats.skewnorm.pdf(-1.95, halo_sn[2], *halo_sn[:2]) / np.log(10)
        sig_birth = 0.03 / 0.075
        ln_norm = gal._thin_disk_norm_table(sig_birth)
        for i in range(d.size):
            x = R[i] / R_SOLAR
            ln_nu, ln_col = _ln_profile(
                x, _sigma_z(R[i], age[i], feh[i]) ** 2 / _PHI_SCALE, True
            )
            R_fit, ln_norm_R = gal._thin_disk_fit_radius(R[i], ln_norm)
            terms = gal._thin_disk_age_terms(age[i])
            thin = (
                logn_disk(R[i], Z_SOLAR)
                + gal._thin_disk_ln_joint(feh[i], R_fit, *terms, sig_birth)
                - ln_norm_R
                + ln_nu(abs(Z[i]))
                - np.log(2)
                - ln_col
                - ln_d0
            )
            s = kin.THICK_DISK_SIGMA[2] ** 2 * kin._thick_disk_scale(x)[2] / _PHI_SCALE
            ln_nu, ln_col = _ln_profile(x, s, False)
            mean_age = 13.0 - 4.0 * (max(feh[i], -1.0) + 1.0)
            thick = (
                logn_disk(R[i], Z_SOLAR, R_scale=1.9)
                + np.log(0.18)
                + ln_nu(abs(Z[i]))
                - np.log(2)
                - ln_col
                - ln_d0
                + stats.skewnorm.logpdf(feh[i], thick_sn[2], *thick_sn[:2])
                + stats.truncnorm.logpdf(
                    age[i], -mean_age / 0.8, (13.8 - mean_age) / 0.8, mean_age, 0.8
                )
            )
            halo = (
                logn_halo(R[i], Z[i])
                + np.log(f_halo)
                + stats.skewnorm.logpdf(feh[i], halo_sn[2], *halo_sn[:2])
                - np.log(halo_norm)
                + stats.truncnorm.logpdf(age[i], -8.0, 1.8 / 1.5, 12.0, 1.5)
            )
            ln_bar = np.log(
                _bar_reference(*_bar_frame(d[i], *coord)) / 0.030e9
            ) + _ln_bar_labels(feh[i], age[i], 0.0, 0.0)
            np.testing.assert_allclose(
                got[:, i], [thin, thick, halo, ln_bar], atol=5e-4
            )

    def test_thin_disk_density_without_labels_is_the_sum_over_populations(self):
        # Off the nodes of the summed table in x and z.
        f_L, table, ln_norm = gal._thin_disk_tables(*_THIN_KEY)
        ln_phi = np.log(_PHI_SCALE)
        for x, z in [(0.3137, 0.0123), (1.0213, 0.41), (1.62, 2.3), (2.9, 7.7)]:
            R_fit, ln_norm_R = gal._thin_disk_fit_radius(x * R_SOLAR, ln_norm)
            ln_w, ln_s, ln_col = gal._thin_disk_age_sum(
                gal._AGE_NODES, gal._AGE_LN_W, x, R_fit, ln_phi, f_L, _THIN_KEY[-1]
            )
            ln_fz2, jeans, _ = kin._vertical_terms(x, z)
            terms = ln_w - ln_norm_R - ln_col - np.log(2) - ln_fz2
            terms -= jeans * np.exp(-ln_s)
            assert np.isclose(
                gal._ln_marginal(x, z, table), logsumexp(terms), atol=2e-3
            )

    def test_feh_table_matches_the_sum_over_ages(self):
        # Off the nodes of the table in [Fe/H], x and z.
        f_L, _, ln_norm = gal._thin_disk_tables(*_THIN_KEY)
        table = gal._thin_disk_feh_table(*_THIN_KEY)
        labels = (_THIN_KEY[-1], 0.1, 0.2)
        ln_phi = np.log(_PHI_SCALE)
        for x, z, feh in [
            (0.47, 0.031, 0.21),
            (1.013, 0.62, -0.307),
            (1.71, 2.3, -0.66),
        ]:
            R_fit, ln_norm_R = gal._thin_disk_fit_radius(x * R_SOLAR, ln_norm)
            nodes = gal._thin_disk_feh_ages(
                feh, gal._AGE_NODES, gal._AGE_LN_W, _THIN_KEY[-1]
            )
            ln_fz2, jeans, _ = kin._vertical_terms(x, z)
            ref = gal._thin_disk_ln_feh_sum(nodes, R_fit, x, ln_phi, f_L, ln_fz2, jeans)
            ref -= ln_norm_R
            fl = gal._label_index(feh, 0.0, True, labels, table)
            fi = x / (kin._V_DX * gal._LABEL_STEPS[0])
            fk = np.arcsinh(z / kin._V_ZS) / (kin._V_DU * gal._LABEL_STEPS[1])
            assert np.isclose(gal._catmull_rom_3d(fl, fi, fk, table), ref, atol=4e-3)

    @pytest.mark.parametrize("age", [0.07, 2.3, 9.4, 13.1])
    def test_blurred_thin_disk_ages_match_quadrature_over_the_true_age(self, age):
        # Both the joint density with [Fe/H] and the table by age alone,
        # against the trapezoid rule on 4001 true ages.
        frac, floor = 0.1, 0.2
        f_L, _, ln_norm = gal._thin_disk_tables(*_THIN_KEY)
        table = gal._thin_disk_age_table(*_THIN_KEY, frac, floor)
        ln_phi, sig_birth = np.log(_PHI_SCALE), _THIN_KEY[-1]
        true = np.linspace(0.0, 12.0 - 1e-9, 4001)
        ln_k = [gal._ln_age_kernel(age, t, frac, floor) for t in true]
        ln_k = np.log(np.gradient(true)) + ln_k
        ln_k[[0, -1]] -= np.log(2.0)
        for x, z, feh in [(1.0, 0.05, 0.0), (0.83, 0.7, -0.3), (1.37, 1.6, -0.6)]:
            R_fit, ln_norm_R = gal._thin_disk_fit_radius(x * R_SOLAR, ln_norm)
            ln_fz2, jeans, _ = kin._vertical_terms(x, z)
            args = (x, ln_phi, f_L, ln_fz2, jeans)
            by_age = [
                gal._ln_sum_nu(
                    *gal._thin_disk_age_populations(
                        t, x, R_fit, ln_phi, f_L, sig_birth
                    ),
                    ln_fz2,
                    jeans,
                )
                for t in true
            ]
            joint = []
            for t in true:
                sig_R0, sig_z0 = kin._thin_disk_sigma0(t, feh)
                ln_s = gal._thin_disk_ln_s(
                    np.log(sig_R0), np.log(sig_z0), x, ln_phi, f_L
                )
                joint.append(
                    gal._thin_disk_ln_joint(
                        feh, R_fit, *gal._thin_disk_age_terms(t), sig_birth
                    )
                    + gal._thin_disk_ln_nu(
                        ln_s, gal._ln_column(x, ln_s, gal._THIN_COLUMN), ln_fz2, jeans
                    )
                )
            ages, ln_w = gal._true_age_nodes(age, frac, floor)
            populations = gal._thin_disk_age_sum(
                ages, ln_w, x, R_fit, ln_phi, f_L, sig_birth
            )
            nodes = gal._thin_disk_feh_ages(feh, ages, ln_w, sig_birth)
            for got, ln_f in [
                (gal._ln_sum_nu(*populations, ln_fz2, jeans), by_age),
                (gal._thin_disk_ln_feh_sum(nodes, R_fit, *args), joint),
            ]:
                # 16 true ages; the third position is e^-23 for age 0.07.
                assert np.isclose(got, logsumexp(ln_k + ln_f), atol=0.015)
            fl = gal._label_index(0.0, age, False, (0.0, frac, floor), table)
            fi = x / (kin._V_DX * gal._LABEL_STEPS[0])
            fk = np.arcsinh(z / kin._V_ZS) / (kin._V_DU * gal._LABEL_STEPS[1])
            assert np.isclose(
                gal._catmull_rom_3d(fl, fi, fk, table) + ln_norm_R,
                logsumexp(ln_k + by_age),
                atol=5e-3,
            )

    def test_fL_table_matches_the_mean_angular_momentum(self):
        # f_L of sigma_z at the population's mean L_z, from the kinematic
        # kernels, between the table nodes.
        f_L = gal._thin_disk_fL_rows(R_SOLAR, _VC_SUN, 2.74, 2.0)
        for R in [0.9, 3.03, 4.1, 8.3, 13.7, 19.6, 21.1]:
            for age, feh in [(0.2, 0.3), (4.0, -0.2), (11.0, -0.9)]:
                L, (L_sun, L_clamp) = _mean_L(R, age, feh)
                ref = np.log(kin._thin_disk_fL(L, L_sun, L_clamp)[1])
                ln_sig = np.log(kin._thin_disk_sigma0(age, feh)[0])
                k = (ln_sig - gal._FL_LNSIG0) / gal._FL_DLNSIG
                got = gal._catmull_rom_2d(R / R_SOLAR / kin._V_DX, k, f_L)
                assert np.isclose(got, ref, atol=2e-4)

    def test_potential_scales_with_the_circular_velocity(self):
        # Phi scales as v_c^2, so scaling v_circ_solar and thick_sigma_z by
        # one factor leaves the thick-disk column unchanged.
        base = gal.galactic_prior_summary()["thick_column_height"]
        scaled = gal.galactic_prior_summary(
            v_circ_solar=1.2 * _VC_SUN, thick_sigma_z=1.2 * kin.THICK_DISK_SIGMA[2]
        )["thick_column_height"]
        np.testing.assert_allclose(scaled, base, rtol=1e-10)

    def test_hot_thick_disk_column_includes_its_tail_above_10_kpc(self):
        # In the BV24 potential a hot disk is bound only logarithmically:
        # with sigma_z = 100 km/s, stopping the column at 10 kpc lost 5%.
        ln_nu, ln_col = _ln_profile(1.0, 100.0**2 / _PHI_SCALE, False)
        column = np.exp(ln_col - ln_nu(Z_SOLAR))
        summary = gal.galactic_prior_summary(thick_sigma_z=100.0)
        np.testing.assert_allclose(summary["thick_column_height"], column, rtol=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"thick_sigma_z": 35.0},
            {"R_thin": 2.5},
            {"f_thick_surface": 0.3},
            {"halo_mass": 1e9, "q_halo": 0.6},
        ],
    )
    def test_prior_at_the_sun_uses_the_derived_fractions(self, kwargs):
        # Toward the north Galactic pole at 1 mpc, every component is at
        # the Sun's position, where the thin disk is 1.
        _, comp = logp_galactic_structure(
            np.array([1e-6]), (0.0, 90.0), return_components=True, **kwargs
        )
        thin, thick, halo, bar = comp["number_density"]
        summary = gal.galactic_prior_summary(**kwargs)
        # The bar's sightline table holds its value at 1 pc below 1 pc.
        np.testing.assert_allclose(
            np.exp(bar - thin), summary["f_bar_solar"], rtol=1e-3
        )
        np.testing.assert_allclose(thin - 2 * np.log(1e-6), 0.0, atol=1e-5)
        np.testing.assert_allclose(
            np.exp(thick - thin), summary["f_thick_solar"], rtol=1e-5
        )
        np.testing.assert_allclose(
            np.exp(halo - thin), summary["f_halo_solar"], rtol=1e-5
        )

    def test_summary_rejects_keywords_that_set_no_normalization(self):
        with pytest.raises(TypeError, match="f_halo"):
            gal.galactic_prior_summary(f_halo=0.01)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"f_thick_surface": 0.0},
            {"f_thick_surface": np.inf},
            {"R_thin": np.nan},
            {"halo_mass": 0.0},
            {"eta_halo_inner": np.nan},
            {"rho_thin_solar": -0.03},
            {"thick_feh": (-0.45, 0.0, -0.5)},
            {"thick_feh": 0.5},
            {"halo_feh": -1.2},
            {"thick_feh": (-0.45, 0.22, -1.0)},
            {"halo_feh": (-1.2, 0.3)},
            {"thick_age": (13.0, -4.0, 0.0)},
            {"halo_age": (12.0, np.nan)},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            logp_galactic_structure(np.array([1.0]), (0.0, 90.0), **kwargs)


def _frankel_reference(feh, age, R, scatter=0.03):
    """p([Fe/H], age, R) of Frankel et al. (2018) by quadrature over the
    birth radius, with a birth scatter of ``scatter`` dex (None for the
    age density alone), and the onset fade beyond 8 Gyr."""
    tau_m, tau_sfr = gal.THIN_DISK_SFH
    f_m, grad, r_now, gamma = gal.THIN_DISK_ENRICHMENT
    sfr = np.exp(-(tau_m - age) / tau_sfr - 0.5 * max(age - 8.0, 0.0) ** 2)
    rexp = 3 * (1 - 0.42 * age / 8)
    s = 3.6 * np.sqrt(age / 8)
    f = (1 - age / tau_m) ** gamma

    def integrand(r0):
        kernel = stats.norm.pdf(R, r0, s) / stats.norm.cdf(r0 / s)
        feh0 = f_m + grad * r0 - (f_m + grad * r_now) * f
        birth = 1.0 if scatter is None else stats.norm.pdf(feh, feh0, scatter)
        return np.exp(-r0 / rexp) / rexp * kernel * birth

    return sfr * quad(integrand, 0.0, 60.0, points=[R], epsrel=1e-10, limit=400)[0]


class TestThinDiskLabels:
    """The thin-disk ages and [Fe/H] of Frankel et al. (2018)."""

    @pytest.mark.parametrize(
        "feh, age, R, rtol",
        [
            (0.1, 0.3, 8.0, 1e-6),
            (-0.2, 2.0, 11.0, 1e-4),
            (0.0, 4.0, 6.0, 1e-3),
            # Born near R0 = 0, where the kernel normalization is
            # approximated.
            (0.1, 7.5, 8.3, 0.03),
            (0.05, 6.0, 5.0, 0.03),
            (-0.4, 9.5, 8.0, 0.03),
        ],
    )
    def test_joint_density_matches_its_integral_over_birth_radius(
        self, feh, age, R, rtol
    ):
        terms = gal._thin_disk_age_terms(age)
        got = np.exp(gal._thin_disk_ln_joint(feh, R, *terms, 0.03 / 0.075))
        assert np.isclose(got, _frankel_reference(feh, age, R), rtol=rtol, atol=0)

    @pytest.mark.parametrize(
        "age, R", [(0.3, 8.0), (4.0, 6.0), (7.9, 5.0), (10.0, 9.0)]
    )
    @pytest.mark.parametrize("sig_birth", [0.4, 1.4])
    def test_feh_nodes_sum_to_the_integral_over_feh(self, age, R, sig_birth):
        terms = gal._thin_disk_age_terms(age)
        fehs, ln_w = np.empty(gal._FEH_NODES), np.empty(gal._FEH_NODES)
        gal._thin_disk_feh_nodes(R, terms, sig_birth, fehs, ln_w)
        ref = quad(
            lambda f: np.exp(gal._thin_disk_ln_joint(f, R, *terms, sig_birth)),
            -3.0,
            2.0,
            points=[fehs[np.argmax(ln_w)]],
            **_QUAD,
        )[0]
        assert np.isclose(np.exp(logsumexp(ln_w)), ref, rtol=1e-4)

    @pytest.mark.parametrize("age, R", [(4.0, 6.0), (7.9, 5.0), (10.0, 9.0)])
    def test_age_density_with_the_default_scatter_is_within_4_percent(self, age, R):
        # With the 0.1 dex systematic, the kernel normalization taken at the
        # mean birth radius misses the exact integral by up to 3.2%.
        fehs, ln_w = np.empty(gal._FEH_NODES), np.empty(gal._FEH_NODES)
        gal._thin_disk_feh_nodes(R, gal._thin_disk_age_terms(age), 1.39, fehs, ln_w)
        ref = _frankel_reference(0.0, age, R, scatter=None)
        assert np.isclose(np.exp(logsumexp(ln_w)), ref, rtol=0.04)

    def test_labels_keep_their_edge_values_outside_the_fitted_radii(self):
        ln_norm = gal._thin_disk_norm_table(1.4)
        lo, hi = gal.THIN_DISK_FIT_RADII
        for inside, outside in [(lo - 1.0, 0.0), (hi + 1.0, 30.0)]:
            assert gal._thin_disk_fit_radius(
                outside, ln_norm
            ) == gal._thin_disk_fit_radius(inside, ln_norm)

    def test_star_formation_fades_beyond_8_gyr_and_stops_at_12_gyr(self):
        # A half-Gaussian of width 1 Gyr times the exponential star
        # formation history, continuous at the onset.
        def ln_sfr(age):
            return gal._thin_disk_age_terms(age)[0]

        assert ln_sfr(8.0) == pytest.approx(-4.0 / 5.9)
        assert ln_sfr(10.0) == pytest.approx(-2.0 / 5.9 - 2.0)
        assert ln_sfr(12.0) == -1e300

    def test_thin_disk_age_density_is_smooth_across_the_onset(self):
        # The labelled-age density at the Sun has no step at 8 Gyr and
        # falls beyond it.
        ages = np.linspace(6.0, 13.0, 701)
        _, comps = logp_galactic_structure(
            np.full(ages.size, 1e-3),
            (0.0, 90.0),
            loga=np.log10(ages * 1e9),
            return_components=True,
        )
        ln_p = comps["labels"][0]
        assert np.abs(np.diff(ln_p)).max() < 0.03
        assert np.all(np.diff(ln_p[ages > 8.5]) < 0)

    def test_local_feh_matches_the_solar_neighbourhood(self):
        # All components: -0.06 +/- 0.22 (GCS) and +0.01 +/- 0.20 (APOGEE),
        # with the 0.1 dex systematic added in quadrature.
        fehs = np.linspace(-3.0, 1.0, 2001)
        lnp = logp_galactic_structure(np.full(fehs.size, 1e-3), (0.0, 0.0), feh=fehs)
        p = np.exp(lnp - lnp.max())
        p /= np.trapezoid(p, fehs)
        mean = np.trapezoid(p * fehs, fehs)
        sd = np.sqrt(np.trapezoid(p * (fehs - mean) ** 2, fehs))
        assert -0.06 < mean < 0.01 and 0.21 < sd < 0.25

    def test_young_stars_follow_the_ism_gradient(self):
        fehs = np.linspace(-1.5, 1.0, 2501)
        radii = np.array([6.0, 8.0, 10.0, 12.0])
        terms = gal._thin_disk_age_terms(0.2)
        means = []
        for R in radii:
            p = np.exp([gal._thin_disk_ln_joint(f, R, *terms, 0.4) for f in fehs])
            means.append(np.trapezoid(p * fehs, fehs) / np.trapezoid(p, fehs))
        assert np.allclose(np.diff(means) / 2.0, -0.075, atol=0.005)

    @pytest.mark.parametrize("systematic", [0.0, 0.1, 0.25])
    def test_feh_systematic_adds_to_the_birth_scatter(self, systematic):
        # The youngest stars have the birth scatter in [Fe/H], with the
        # systematic added in quadrature.
        sig_birth = gal._label_args(
            (-0.45, 0.22, -0.5),
            (13.0, -4.0, 0.8),
            (-1.18, 0.28, -0.67, -1.95, 1.0),
            (12.0, 1.5),
            (0.5, -0.31, 0.31, 0.26, 0.20),
            (11.0, 1.5, 8.0, 3.0),
            (systematic, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
        )[0][0]
        fehs = np.linspace(-1.5, 1.5, 3001)
        terms = gal._thin_disk_age_terms(1e-6)
        p = np.exp([gal._thin_disk_ln_joint(f, 8.0, *terms, sig_birth) for f in fehs])
        mean = np.trapezoid(p * fehs, fehs) / np.trapezoid(p, fehs)
        sd = np.sqrt(np.trapezoid(p * (fehs - mean) ** 2, fehs) / np.trapezoid(p, fehs))
        assert sd == pytest.approx(np.hypot(0.03, systematic), rel=1e-3)


class TestThickDiskAndHaloLabels:
    """Skew-normal [Fe/H] and ages tied to [Fe/H] (thick disk) or not
    (halo)."""

    @pytest.mark.parametrize(
        "moments", [(-0.45, 0.22, -0.5), (0.3, 1.5, 0.9), (-1.0, 0.1, 0.0)]
    )
    def test_skew_normal_has_the_requested_moments(self, moments):
        xi, omega, alpha, *_ = gal._skew_normal_args("thick_feh", moments, 0.0)
        got = stats.skewnorm.stats(alpha, xi, omega, moments="mvs")
        np.testing.assert_allclose(
            [got[0], np.sqrt(got[1]), got[2]], moments, rtol=1e-10, atol=1e-12
        )
        v = xi + omega * np.linspace(-3.0, 3.0, 7)
        # ln Phi is tabulated to within 3e-9.
        np.testing.assert_allclose(
            [gal._ln_skew_normal(u, xi, omega, alpha) for u in v],
            stats.skewnorm.logpdf(v, alpha, xi, omega),
            rtol=0,
            atol=1e-8,
        )

    @pytest.mark.parametrize("blur", [0.05, 0.15])
    def test_blurred_skew_normal_is_its_convolution_with_a_normal(self, blur):
        xi, omega, alpha, *_ = gal._skew_normal_args(
            "thick_feh", (-0.45, 0.22, -0.5), 0
        )
        args = gal._skew_normal_args("thick_feh", (-0.45, 0.22, -0.5), blur)
        for v in [-1.3, -0.6, -0.2, 0.3]:
            ref = quad(
                lambda u: stats.skewnorm.pdf(u, alpha, xi, omega)
                * stats.norm.pdf(v, u, blur),
                -4.0,
                2.0,
                **_QUAD,
            )[0]
            assert np.isclose(np.exp(gal._ln_skew_normal(v, *args[:3])), ref, rtol=1e-8)

    def test_default_halo_feh_reproduces_the_naidu_percentiles(self):
        # Core and tail before the systematic, normalized on its grid.
        xi, omega, alpha, *_ = gal._skew_normal_args(
            "halo_feh", (-1.18, 0.28, -0.67), 0.0
        )
        ln_p = gal._halo_feh_table(xi, omega, alpha, -1.95, 1.0, 0.0)
        fehs = gal._HALO_FEHS
        cdf = np.cumsum(np.exp(ln_p)) * gal._HALO_DFEH
        np.testing.assert_allclose(
            np.interp([0.16, 0.5, 0.84], cdf, fehs), [-1.48, -1.15, -0.91], atol=0.01
        )

    @pytest.mark.parametrize("blur", [0.0, 0.15])
    def test_halo_tail_falls_as_a_power_of_ten_and_keeps_the_mass(self, blur):
        # 10^[Fe/H] below the break (Youakim et al. 2020), and a normalized
        # density with or without the systematic.
        xi, omega, alpha, *_ = gal._skew_normal_args(
            "halo_feh", (-1.18, 0.28, -0.67), 0.0
        )
        ln_p = gal._halo_feh_table(xi, omega, alpha, -1.95, 1.0, blur)
        fehs = gal._HALO_FEHS
        i, j = np.searchsorted(fehs, [-4.0, -3.0])
        assert (ln_p[j] - ln_p[i]) / np.log(10) == pytest.approx(1.0, abs=1e-6)
        assert np.trapezoid(np.exp(ln_p), fehs) == pytest.approx(1.0, abs=1e-5)

    def test_thick_disk_and_halo_ages_include_the_systematics(self):
        # Standard deviations: the intrinsic spread, 15% of the mean age
        # and the 0.2 Gyr floor, in quadrature.
        ages = np.array([4.0, 9.0, 12.5])
        _, comps = logp_galactic_structure(
            np.full(3, 1.0),
            (0.0, 90.0),
            feh=np.full(3, -0.5),
            loga=np.log10(ages * 1e9),
            return_components=True,
        )
        _, no_age = logp_galactic_structure(
            np.full(3, 1.0), (0.0, 90.0), feh=np.full(3, -0.5), return_components=True
        )
        got = np.subtract(comps["labels"], no_age["labels"])[1:]
        for mean, sd, row in [(11.0, 0.8, got[0]), (12.0, 1.5, got[1])]:
            sd = np.sqrt(sd**2 + (0.15 * mean) ** 2 + 0.2**2)
            ref = stats.truncnorm.logpdf(ages, -mean / sd, (13.8 - mean) / sd, mean, sd)
            np.testing.assert_allclose(row, ref, atol=1e-10)

    @pytest.mark.parametrize("feh, peak", [(0.0, 9.0), (-1.0, 13.0), (-1.5, 13.0)])
    def test_thick_disk_ages_follow_the_old_sequence(self, feh, peak):
        # Xiang & Rix (2022): 13 Gyr at [Fe/H] = -1 to about 7 Gyr at +0.5,
        # held below -1, where their sequence starts.
        ages = np.linspace(0.01, 13.8, 1380)
        _, comps = logp_galactic_structure(
            np.full(ages.size, 1.0),
            (0.0, 90.0),
            feh=np.full(ages.size, feh),
            loga=np.log10(ages * 1e9),
            return_components=True,
        )
        assert abs(ages[np.argmax(comps["labels"][1])] - peak) < 0.02


@pytest.mark.slow
@pytest.mark.parametrize(
    "coord, d, systematics",
    [
        ((0.0, 0.0), 1e-3, {}),
        ((90.0, 60.0), 1.5, {}),
        ((180.0, 5.0), 6.0, {}),
        ((90.0, 60.0), 1.5, _NO_SYSTEMATICS),
        ((30.0, 0.0), 5.0, {}),  # in the long bar
    ],
)
def test_label_distributions_sum_to_the_density_without_labels(coord, d, systematics):
    # The labels of each component integrate to 1 at every position, and a
    # density with one label is the joint density summed over the other.
    # Gauss-Legendre panels: 8 points per 0.1 dex down to the halo tail, and
    # 16 per age panel, twice the production nodes, to MAX_AGE.
    fehs, w_feh = gal._gauss_legendre_panels(
        np.concatenate(([-6.0, -4.0], np.arange(-3.0, 1.21, 0.1))), 8
    )
    ages, w_age = gal._gauss_legendre_panels(gal._AGE_EDGES + [13.8], 16)
    F, A = np.meshgrid(fehs, ages, indexing="ij")

    # A component with no density at the position (the bar, far from the
    # centre) has no label distribution there.
    _, comps = logp_galactic_structure(np.array([d]), coord, return_components=True)
    present = np.ravel(comps["number_density"]) > gal.LOG_ZERO / 2

    def labels(**kw):
        n = len(next(iter(kw.values())))
        comps = logp_galactic_structure(
            np.full(n, d), coord, return_components=True, **systematics, **kw
        )[1]
        return np.exp(comps["labels"])[present]

    joint = labels(feh=F.ravel(), loga=np.log10(A.ravel() * 1e9))
    joint = joint.reshape((-1,) + F.shape)
    with_feh = labels(feh=fehs)
    with_age = labels(loga=np.log10(ages * 1e9))
    np.testing.assert_allclose(w_feh @ joint @ w_age, 1.0, atol=0.003)
    np.testing.assert_allclose(with_feh @ w_feh, 1.0, atol=0.003)
    np.testing.assert_allclose(with_age @ w_age, 1.0, atol=0.003)
    for marginal, summed in [(with_feh, joint @ w_age), (with_age, w_feh @ joint)]:
        keep = marginal > 1e-3 * marginal.max(axis=1, keepdims=True)
        np.testing.assert_allclose(summed[keep], marginal[keep], rtol=0.01)


class TestHaloShapeAndMass:
    """The halo is the Amarante et al. (2024) broken power law, normalized
    by the Lane et al. (2023) mass over 2 to 70 kpc."""

    def test_halo_matches_the_broken_power_law(self):
        # Points inside and beyond the break.
        R = np.array([4.0, R_SOLAR, 15.0, 25.0, 10.0, 40.0])
        Z = np.array([3.0, Z_SOLAR, 12.0, 5.0, 35.0, 20.0])
        m = np.sqrt(R**2 + (Z / 0.77) ** 2 + 4.0)
        ln_n = np.where(
            m < 19.1, -2.85 * np.log(m), -2.85 * np.log(19.1) - 4.55 * np.log(m / 19.1)
        )
        m_sun = np.sqrt(R_SOLAR**2 + (Z_SOLAR / 0.77) ** 2 + 4.0)
        np.testing.assert_allclose(
            logn_halo(R, Z), ln_n + 2.85 * np.log(m_sun), rtol=1e-12
        )

    def test_halo_density_integrates_to_the_halo_mass(self):
        # An adaptive integral in cylindrical (R, Z), independent of the
        # Gauss-Legendre nodes in (ln r, cos theta).
        rho = _SUMMARY["halo_density_solar"] * 1e9  # Msun / kpc^3

        def column(R):
            z_lo = np.sqrt(max(4.0 - R**2, 0.0))
            z_hi = np.sqrt(70.0**2 - R**2)
            return quad(
                lambda z: np.exp(logn_halo(R, z)), z_lo, z_hi, epsrel=1e-5, limit=200
            )[0]

        mass = (
            4.0
            * np.pi
            * rho
            * quad(
                lambda R: R * column(R), 0.0, 70.0, points=[2.0], epsrel=1e-5, limit=200
            )[0]
        )
        np.testing.assert_allclose(mass, 7.5e8, rtol=1e-4)


def test_thick_disk_has_one_near_side_peak_toward_the_galactic_centre():
    # Inside 3 kpc the BV24 high-alpha sigma_z rises to 119 km/s. Without
    # the inner hold, the thick disk toward (l, b) = (0, 30) deg had two
    # more distance peaks, at 7.8 and 10.2 kpc, above the Galactic centre.
    d = np.linspace(0.05, 12.0, 4000)
    _, comp = logp_galactic_structure(d, (0.0, 30.0), return_components=True)
    lnp = comp["number_density"][1]
    peaks = d[1:-1][(lnp[1:-1] > lnp[:-2]) & (lnp[1:-1] > lnp[2:])]
    assert peaks.size == 1 and peaks[0] < 6.0


class TestBar:
    """The bar is the Sormani et al. (2022) density, turned by bar_angle
    and scaled to bar_mass, with position-independent labels."""

    def test_jitted_density_matches_the_equations_of_the_paper(self):
        rng = np.random.default_rng(5)
        pts = rng.uniform(-6.0, 6.0, (400, 3)) * [1.0, 1.0, 0.4]
        pts[0] = (0.0, 0.0, 0.0)  # the centre, where R = 0
        got = [gal._bar_density(*p) for p in pts]
        np.testing.assert_allclose(got, _bar_reference(*pts.T), rtol=1e-12)

    def test_masses_are_those_of_the_paper(self):
        # Midpoint rule in one octant, 0.05 kpc steps.
        h = 0.05
        x = np.arange(h / 2, 9.0, h)
        z = np.arange(h / 2, 4.0, h)
        X, Y, Z = np.meshgrid(x, x, z, indexing="ij", sparse=True)
        mass = 8 * h**3 * _bar_reference(X, Y, Z).sum()
        np.testing.assert_allclose(mass, gal.BAR_MASS, rtol=2e-4)
        # The X-shaped bar alone is 1.28e10 in Table 1 of the paper.
        x_only = 8 * h**3 * _x_bar_reference(X, Y, Z).sum()
        assert abs(x_only - 1.28e10) < 0.005e10

    def test_prior_matches_the_density_scaled_by_its_mass(self):
        # Several sightlines: the density is evaluated at each point.
        d = np.linspace(4.0, 12.0, 9)
        ell, b = np.linspace(-8.0, 8.0, 9), np.full(9, -3.0)
        _, comps = logp_galactic_structure(
            d, (ell, b), bar_mass=2 * gal.BAR_MASS, return_components=True
        )
        expected = np.log(
            2 * _bar_reference(*_bar_frame(d, ell, b)) / 0.030e9
        ) + 2 * np.log(d)
        np.testing.assert_allclose(comps["number_density"][3], expected, rtol=1e-12)

    @pytest.mark.parametrize("coord", [(0.0, 0.0), (-6.0, 3.0), (10.0, 5.0)])
    def test_sightline_table_matches_the_density_at_each_point(self, coord):
        # One sightline interpolates in the table; several sightlines
        # evaluate the density at each point. A second sightline far away
        # makes the call one with several.
        d = np.exp(np.random.default_rng(2).uniform(np.log(1e-3), np.log(25), 4000))
        _, table = logp_galactic_structure(d, coord, return_components=True)
        ell, b = np.full(d.size + 1, coord[0]), np.full(d.size + 1, coord[1])
        ell[-1] = 180.0
        _, exact = logp_galactic_structure(
            np.append(d, 1.0), (ell, b), return_components=True
        )
        exact = {k: [v[:-1] for v in vs] for k, vs in exact.items()}
        bar = exact["number_density"][3] - 2 * np.log(d)
        keep = bar > np.log(1e3 / 0.030e9)  # above 1e3 Msun/kpc^3
        np.testing.assert_allclose(
            table["number_density"][3][keep],
            exact["number_density"][3][keep],
            atol=2e-3,
        )

    def test_one_sightline_as_an_array_gives_the_scalar_result(self):
        d = np.linspace(0.5, 15.0, 50)
        scalar = logp_galactic_structure(d, (5.0, -3.0))
        array = logp_galactic_structure(d, (np.full(50, 5.0), np.full(50, -3.0)))
        np.testing.assert_array_equal(array, scalar)

    def test_density_next_to_the_axis_does_not_overflow(self):
        # Regression: the long-bar cutoffs overflowed without JIT at R ~ 0.
        got = gal._bar_density(1e-12, 0.0, 0.1)
        np.testing.assert_allclose(got, _x_bar_reference(1e-12, 0.0, 0.1), rtol=1e-12)

    @pytest.mark.skipif(
        os.environ.get("NUMBA_DISABLE_JIT") == "1",
        reason="without JIT, log of a negative distance raises",
    )
    def test_negative_distance_gives_nan_inside_the_table(self):
        # Regression: a NaN table index read outside the table.
        assert np.isnan(logp_galactic_structure(np.array([-3.0]), (0.0, 4.0)))[0]

    def test_near_end_is_at_positive_longitude(self):
        d = np.linspace(4.0, 13.0, 901)
        peak = {}
        for ell in (6.0, -6.0):
            _, comps = logp_galactic_structure(d, (ell, 3.0), return_components=True)
            peak[ell] = d[np.argmax(comps["number_density"][3])]
        assert peak[6.0] < peak[-6.0] - 0.5

    def test_zero_mass_gives_the_three_component_prior(self):
        d = np.linspace(0.1, 15.0, 50)
        coord = (0.0, 4.0)
        feh, loga = np.full(50, 0.1), np.full(50, 9.9)
        lp, comps = logp_galactic_structure(
            d, coord, feh=feh, loga=loga, bar_mass=0.0, return_components=True
        )
        terms = np.add(comps["number_density"], comps["labels"])
        assert np.all(terms[3] <= gal.LOG_ZERO / 2)
        np.testing.assert_allclose(
            lp, logsumexp(terms[:3], axis=0), rtol=1e-12, atol=1e-15
        )

    def test_labels_are_the_mixture_with_the_thick_disk_systematics(self):
        rng = np.random.default_rng(8)
        feh = rng.uniform(-1.0, 0.5, 20)
        age = rng.uniform(1.0, 13.0, 20)
        _, comps = logp_galactic_structure(
            np.full(20, 8.0),
            (0.0, 4.0),
            feh=feh,
            loga=np.log10(age * 1e9),
            return_components=True,
        )
        expected = _ln_bar_labels(feh, age, 0.15, 0.15, 0.2)
        np.testing.assert_allclose(comps["labels"][3], expected, atol=1e-9)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bar_mass": -1.0},
            {"bar_angle": np.nan},
            {"bar_feh": (1.5, -0.3, 0.3, 0.3, 0.2)},
            {"bar_feh": (0.5, -0.3, 0.0, 0.3, 0.2)},
            {"bar_age": (11.0, 1.5, 8.0)},
        ],
    )
    def test_invalid_bar_keywords_raise(self, kwargs):
        with pytest.raises(ValueError, match="bar_"):
            logp_galactic_structure(np.array([8.0]), (0.0, 4.0), **kwargs)
