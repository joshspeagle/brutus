"""Tests of brutus.priors.galactic."""

import warnings

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import logsumexp

from brutus.priors import galactic as gal
from brutus.priors import kinematic as kin
from brutus.priors import (
    logn_disk,
    logn_halo,
    logp_age_from_feh,
    logp_feh,
    logp_galactic_structure,
)
from brutus.utils import truncnorm_logpdf
from brutus.utils.math import galactic_to_galactocentric_cyl

_QUAD = dict(epsabs=1e-12, epsrel=1e-12, limit=500)


# v_c at the Sun: the default scale of the measured rotation curve.
_VC_SUN = kin.solar_velocity(8.2)[1] - kin.V_PEC_SUN
_ROT = (8.2, _VC_SUN)
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


_AGE_ARGS = (13.8, 0.0, -0.5, 0.5, 2.0, 4.0, 1.0)
_SUMMARY = gal.galactic_prior_summary()
_F_THICK, _F_HALO = _SUMMARY["f_thick_solar"], _SUMMARY["f_halo_solar"]
_COLUMNS = gal._normalizations(**gal._NORMALIZATION_DEFAULTS)[5]


def _ln_columns(R):
    """ln(h(R) / h(R_sun)) of the thin and thick disks, from the production
    tables (tested against direct integrals in
    TestThickDiskSurfaceNormalization)."""
    x = np.asarray(R, dtype=float) / 8.2
    return [gal._ln_column_array(x, table) for table in _COLUMNS]


def _no_age_vertical(R, abs_Z, feh, age_args=_AGE_ARGS):
    """Thin- and thick-disk ln nu(|Z|) - ln nu(|Z_sun|) without an age,
    from the production kernel, with the thin-disk age prior at
    feh_thin = -0.2 and the default thick-disk sigma_z."""
    nodes = gal._thin_age_nodes(-0.2, age_args)
    R, abs_Z, feh = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (R, abs_Z, feh))
    var_z = kin.THICK_DISK_SIGMA[2] ** 2
    args = (R, abs_Z, 0.025, np.empty(0), feh, False, _ROT, (2.74, 2.0), var_z)
    return gal._disk_vertical_array(*args, *nodes)


def _clamp(x, lo, hi, width):
    """The quadratic edge blend of kinematic._soft_clamp, written out."""
    if abs(x - lo) < width:
        return lo + (x - lo + width) ** 2 / (4 * width)
    if abs(x - hi) < width:
        return hi - (hi + width - x) ** 2 / (4 * width)
    return min(max(x, lo), hi)


def _vertical_reference(R, Z, age, feh):
    """ln nu(|Z|) - ln nu(|Z_sun|) of the thin and thick disks, from the
    dispersion relations written out and the vertical potential of
    ``kin._vertical_terms`` (tested against quadrature on its own)."""
    thin = np.empty(len(R))
    thick = np.empty(len(R))
    for i in range(len(R)):
        terms = kin._vertical_terms(R[i] / 8.2, abs(Z[i]))
        terms_sun = kin._vertical_terms(R[i] / 8.2, 0.025)
        L, (L_sun, L_clamp) = _mean_L(R[i], age[i], feh[i])
        L = _clamp(L, *L_clamp)
        a, lam = kin.THIN_DISK_ALPHA_L[1], kin.THIN_DISK_LAMBDA_L[1]
        f_L = (a * (L / L_sun) ** 2 + np.exp(-(L - L_sun) / lam)) / (1 + a)
        sig = (
            kin.THIN_DISK_SIGMA0[1]
            * ((age[i] + 0.1) / 10.1) ** kin.THIN_DISK_BETA[1]
            * (1 + kin.THIN_DISK_GAMMA_FEH[1] * np.clip(feh[i], -1.0, 0.5))
            * f_L
        )
        thin[i] = (
            -(terms[0] - terms_sun[0]) - (terms[1] - terms_sun[1]) * _PHI_SCALE / sig**2
        )
        var = kin.THICK_DISK_SIGMA[2] ** 2 * kin._thick_disk_scale(R[i] / 8.2)[2]
        thick[i] = -(terms[2] - terms_sun[2]) * _PHI_SCALE / var
    return thin, thick


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

    def test_logp_feh_basic(self):
        """Test metallicity prior evaluation."""
        fehs = np.array([-2.0, -1.0, 0.0, 0.5])

        # Basic evaluation with default parameters
        logp = logp_feh(fehs)
        assert np.all(np.isfinite(logp))
        assert len(logp) == len(fehs)

    def test_logp_feh_custom_parameters(self):
        """Test metallicity prior with custom parameters."""
        fehs = np.array([-1.0, 0.0])

        logp_default = logp_feh(fehs)
        logp_custom = logp_feh(fehs, feh_mean=-0.1, feh_sigma=0.4)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)

    def test_logp_age_from_feh_basic(self):
        """Test age-metallicity relation."""
        ages = np.array([1.0, 5.0, 10.0, 13.0])  # Gyr
        fehs = np.array([-0.5, -0.2, 0.0, -0.1])

        logp = logp_age_from_feh(ages, fehs)

        # Should be finite
        assert np.all(np.isfinite(logp))

    def test_logp_age_from_feh_custom_parameters(self):
        """Test age-metallicity relation with custom parameters."""
        ages = np.array([5.0, 8.0])

        logp_default = logp_age_from_feh(ages)
        logp_custom = logp_age_from_feh(ages, feh_mean=-0.1, max_age=12.0)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)

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
        assert len(components["number_density"]) == 3  # thin, thick, halo components

        # Each component should have right length
        for comp in components.values():
            assert len(comp) == len(distances)
            assert np.all(np.isfinite(comp))


class TestDiskSolarNormalization:
    """logn_disk must be exactly 0 at the solar position (like logn_halo),
    so the mixture fractions are exact at the Sun."""

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


def _make_labels(feh, loga, dtype=float):
    labels = np.empty(len(feh), dtype=[("feh", dtype), ("loga", dtype)])
    labels["feh"] = feh
    labels["loga"] = loga
    return labels


class TestFehLogaArrayPaths:
    """Plain feh/loga arrays must produce the same prior as structured
    labels on every code path (fused kernel and small-N numpy)."""

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


class TestMixedArrayAndLabels:
    """A plain feh/loga array must override only its own field: any field
    not given as an array is sourced from a structured `labels`, so mixing
    the two call styles cannot silently drop a prior term."""

    COORD = (90.0, 30.0)

    @pytest.mark.parametrize("N", [500, 2000])  # small-N and fused paths
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


class TestJointFehAgePrior:
    """The (feh, age) prior must be the mixture sum_c w_c p(feh|c) p(age|c),
    not the product of marginal mixtures."""

    COORD = (90.0, 30.0)

    # Default component parameters of logp_galactic_structure
    COMPONENTS = [
        # (feh_mean, feh_sigma, ln_f)
        (-0.2, 0.3, 0.0),
        (-0.7, 0.4, np.log(_F_THICK)),
        (-1.6, 0.5, np.log(_F_HALO)),
    ]

    def _expected_joint(self, dists, feh, loga):
        """Independent reference: logsumexp_c[lnp_c + ln p(feh|c) + ln p(age|c)].

        The vertical integrals are done by quadrature here and from a
        1 pc table in the production code.
        """
        ell, b = self.COORD
        R, Z = galactic_to_galactocentric_cyl(dists, ell, b)
        vol = 2.0 * np.log(dists + 1e-300)
        age = 10.0**loga / 1e9

        vertical = _vertical_reference(R, Z, age, feh)
        columns = _ln_columns(R)
        terms = []
        for idx, (fm, fsig, ln_f) in enumerate(self.COMPONENTS):
            if idx < 2:
                R_scale = (2.74, 1.9)[idx]
                lnp_c = logn_disk(R, 0.025, R_scale=R_scale) + vertical[idx]
                lnp_c -= columns[idx]
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
        np.testing.assert_allclose(logp, expected, rtol=1e-9, atol=1e-9)

    def test_fused_path_matches_mixture(self):
        rng = np.random.default_rng(4)
        N = 1200
        d = rng.uniform(0.05, 10.0, N)
        feh = rng.normal(-0.8, 0.7, N)
        loga = rng.uniform(8.0, 10.1, N)

        logp = logp_galactic_structure(d, self.COORD, feh=feh, loga=loga)
        expected = self._expected_joint(d, feh, loga)
        np.testing.assert_allclose(logp, expected, rtol=1e-9, atol=1e-9)

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


class TestGalacticCoverage:
    """Tests to improve galactic.py code coverage."""

    def test_galactic_structure_tuple_coordinates(self):
        """Test galactic structure with tuple coordinates (not SkyCoord)."""
        distances = np.array([1.0, 2.0])
        coord_tuple = (180.0, 0.0)  # (l, b) in degrees

        logp = logp_galactic_structure(distances, coord_tuple)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_feh(self):
        """Test galactic structure with metallicity labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with metallicity data
        labels = np.array([(0.0,), (-0.5,)], dtype=[("feh", "f4")])

        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_age(self):
        """Test galactic structure with age labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with log(age) data
        labels = np.array(
            [(9.0,), (9.5,)], dtype=[("loga", "f4")]
        )  # log10(age in years)

        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_both(self):
        """Test galactic structure with both metallicity and age labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with both feh and loga
        labels = np.array(
            [(0.0, 9.0), (-0.5, 9.5)], dtype=[("feh", "f4"), ("loga", "f4")]
        )

        logp = logp_galactic_structure(distances, coord, labels=labels)

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

        # Should have all component types
        assert "number_density" in components
        assert "feh" in components
        assert "age" in components

        # Each component should have three subcomponents (thin, thick, halo)
        assert len(components["number_density"]) == 3
        assert len(components["feh"]) == 3
        assert len(components["age"]) == 3

    def test_logp_age_from_feh_parameter_matching(self):
        """Test age-metallicity function with correct parameter names."""
        ages = np.array([5.0, 8.0])

        # Check actual function signature parameters
        logp_default = logp_age_from_feh(ages)
        logp_custom = logp_age_from_feh(ages, feh_mean=-0.1, max_age=12.0, min_age=0.5)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)
        assert np.all(np.isfinite(logp_default))
        assert np.all(np.isfinite(logp_custom))


class TestLegacyImportFallback:
    """Test legacy import fallback (difficult to test without modifying imports)."""

    def test_scipy_import_works(self):
        """Test that scipy.special.logsumexp is available."""
        # This implicitly tests that the import succeeded
        from brutus.priors.galactic import logp_galactic_structure

        distances = np.array([1.0, 2.0, 3.0])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        logp = logp_galactic_structure(distances, coord)

        # The fact that this works means logsumexp was imported correctly
        assert np.all(np.isfinite(logp))
        assert len(logp) == 3


class TestLogZeroBeyondMaxAge:
    def test_fused_and_numpy_paths_agree_beyond_max_age(self):
        # Arrays of more than 1000 distances take the fused kernel.
        for n in (1000, 1001):
            lnp = logp_galactic_structure(
                np.full(n, 1.0),
                (30.0, 10.0),
                feh=np.zeros(n),
                loga=np.full(n, 10.2),
            )
            np.testing.assert_array_equal(lnp, -1e300)

    def test_age_prior_accepts_scalar_age(self):
        from brutus.priors.galactic import logp_age_from_feh

        lnp = logp_age_from_feh(5.0)
        assert np.isfinite(lnp)
        assert lnp == logp_age_from_feh(np.array([5.0]))[0]


class TestLabelsValidation:
    @pytest.mark.parametrize("n", [10, 2000])
    def test_labels_length_mismatch_raises(self, n):
        # 2000 distances take the fused kernel, which would read past labels.
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

    @pytest.mark.parametrize("shape", [(3, 2), (600, 2)])
    def test_labels_matching_multidimensional_dists_broadcast(self, shape):
        # 1200 distances would take the fused kernel, which is 1-D only.
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

    def test_thick_disk_column_height_has_no_kinks_in_radius(self):
        # The clamped potential of #284 bent the column height by 0.046 kpc
        # per 0.25 kpc step at 4 and 9 kpc. The largest bend is now 0.005,
        # at 12.75 kpc, in the blend where the dispersions start to hold.
        z = gal._COLUMN_Z
        R = np.arange(3.0, 20.01, 0.25)
        n = R.size * z.size
        thick = _no_age_vertical(
            np.repeat(R, z.size), np.tile(z, R.size), np.full(n, -0.2)
        )[1]
        h = np.exp(thick).reshape(R.size, -1) @ gal._COLUMN_W
        assert np.abs(np.diff(h, 2)).max() < 0.01

    @pytest.mark.parametrize("feh", [-1.0, 0.0])
    def test_thin_disk_falls_faster_than_z_squared_at_height(self, feh):
        # With sigma_z growing as 1 + gamma_z |z| without bound, the Jeans
        # integral converges and d ln nu / d ln |z| tends to -2. Holding |z|
        # at 2 kpc gives -14 ([Fe/H] = -1) and -32 ([Fe/H] = 0) over
        # 15-30 kpc.
        Z = np.array([15.0, 30.0])
        ln_nu = _no_age_vertical(np.full(2, 8.2), Z, np.full(2, feh))[0]
        assert np.diff(ln_nu)[0] / np.log(2.0) < -10.0

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

    def test_midplane_prior_is_the_surface_density_over_the_column(self):
        # At Z = Z_sun (b = 0) both vertical terms vanish, so the prior is
        # the exponential surface density over the column height.
        d = np.linspace(0.1, 15.0, 50)
        rng = np.random.default_rng(5)
        kw = dict(feh=rng.normal(-0.3, 0.4, 50), loga=rng.uniform(8.5, 10.1, 50))
        R, _ = galactic_to_galactocentric_cyl(d, 30.0, 0.0)
        vol = 2.0 * np.log(d)
        lnp, comps = logp_galactic_structure(
            d, (30.0, 0.0), return_components=True, **kw
        )
        thin, thick, halo = comps["number_density"]
        col_thin, col_thick = _ln_columns(R)
        expected_thin = logn_disk(R, 0.025) - col_thin + vol
        np.testing.assert_allclose(thin, expected_thin, rtol=1e-12)
        np.testing.assert_allclose(
            thick,
            logn_disk(R, 0.025, R_scale=1.9) - col_thick + vol + np.log(_F_THICK),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            halo, logn_halo(R, np.full_like(R, 0.025)) + vol + np.log(_F_HALO)
        )
        fused = logp_galactic_structure(d, (30.0, 0.0), **kw)
        np.testing.assert_allclose(fused, lnp, rtol=1e-12)
        # Without an age, the thin-disk age average is 0 at |Z_sun| too.
        no_age = logp_galactic_structure(d, (30.0, 0.0), return_components=True)
        thin, thick, _ = no_age[1]["number_density"]
        np.testing.assert_allclose(thin, expected_thin, rtol=1e-12)

    @pytest.mark.parametrize("feh", [-0.8, -0.2, 0.3])
    @pytest.mark.parametrize(
        "age_args", [_AGE_ARGS, (13.8, 0.0, -0.5, 0.5, 2.0, 0.5, 0.3)]
    )
    def test_age_average_matches_quadrature_over_ages(self, feh, age_args):
        # Without an age, the thin disk averages over the age prior of the
        # thin disk (at feh_thin), with the star's own [Fe/H] in sigma_z.
        # The second prior is narrow (sigma of 0.5 Gyr), where the nodes
        # must sit on its support.
        def prior(age):
            return np.exp(logp_age_from_feh(np.array([age]), -0.2, *age_args)[0])

        for R, abs_Z in [(4.0, 0.5), (8.25, 0.3), (8.25, 2.0), (15.0, 1.0), (8.2, 6.0)]:

            def weighted(age):
                args = np.array([R]), np.array([abs_Z]), np.array([age])
                return prior(age) * np.exp(
                    _vertical_reference(*args, np.array([feh]))[0][0]
                )

            kw = dict(epsrel=1e-10, limit=200, points=[1.0, 4.0, 8.0, 12.0])
            expected = np.log(
                quad(weighted, 0, 13.8, **kw)[0] / quad(prior, 0, 13.8, **kw)[0]
            )
            no_age = _no_age_vertical(R, abs_Z, feh, age_args)[0][0]
            assert np.isclose(no_age, expected, rtol=0, atol=1e-6)

    def test_feh_outside_the_fitted_range_takes_the_edge_value(self):
        for feh, edge in [(-2.5, -1.0), (1.0, 0.5), (3.0, 0.5)]:
            assert kin._thin_disk_sigma0(4.0, feh) == kin._thin_disk_sigma0(4.0, edge)

    @staticmethod
    def _scale_height(lnnu, z):
        return -1.0 / np.polyfit(z, lnnu, 1)[0]

    def test_disk_scale_heights_match_star_counts(self):
        # New result: scale heights near 0.3 and 0.9 kpc (Juric et al. 2008)
        # follow from the dispersions and the potential.
        z = np.linspace(0.3, 1.5, 25)
        thin = _no_age_vertical(np.full(z.size, 8.2), z, np.full(z.size, -0.2))[0]
        assert 0.25 < self._scale_height(thin, z) < 0.4
        z = np.linspace(0.5, 2.0, 25)
        zero = kin._vertical_terms(1.0, 0.0)
        var_z = kin.THICK_DISK_SIGMA[2] ** 2
        thick = [
            kin._thick_disk_vertical(kin._vertical_terms(1.0, x), zero, var_z)
            for x in z
        ]
        assert 0.8 < self._scale_height(thick, z) < 1.2

    def test_younger_thin_disk_stars_lie_closer_to_the_plane(self):
        d = np.linspace(0.2, 1.5, 20)
        heights = []
        for loga in (9.0, 9.5, 10.0):
            comps = logp_galactic_structure(
                d,
                (0.0, 90.0),
                loga=np.full(d.size, loga),
                return_components=True,
            )[1]["number_density"]
            heights.append(self._scale_height(comps[0] - 2 * np.log(d), d))
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


class TestThickDiskSurfaceNormalization:
    """The thick disk is normalized by its surface density at the Sun."""

    @pytest.mark.parametrize("R", [4.1, 19.7])
    def test_column_heights_between_radial_nodes_match_direct_integrals(self, R):
        # The radial term divides by h(R) / h(R_sun), interpolated from a
        # table at 0.1 R_sun steps. The reference integrates the profiles
        # at R itself, with a grid of [Fe/H] for the thin disk and 16-point
        # Gauss-Legendre panels in z (8 points in production).
        z, w_z = gal._gauss_legendre_panels(gal._COLUMN_EDGES, 16)
        fehs = np.linspace(-1.4, 1.0, 17)
        w_feh = np.exp(-0.5 * ((fehs + 0.2) / 0.3) ** 2)
        R_z = np.full(z.size, R)
        thin = 0.0
        for feh, wf in zip(fehs, w_feh):
            profile = _no_age_vertical(R_z, z, np.full(z.size, feh))[0]
            thin += wf * (np.exp(profile) @ w_z) / w_feh.sum()
        thick = np.exp(_no_age_vertical(R_z, z, R_z)[1]) @ w_z
        col_thin, col_thick = (np.exp(c[0]) for c in _ln_columns(np.array([R])))
        np.testing.assert_allclose(
            _SUMMARY["thin_column_height"] * col_thin, thin, rtol=5e-4
        )
        np.testing.assert_allclose(
            _SUMMARY["thick_column_height"] * col_thick, thick, rtol=1e-4
        )

    def test_summary_column_heights_match_a_dense_integral(self):
        # A reference over log-spaced ages and a grid of [Fe/H] with the
        # per-age kernel, independent of the age and [Fe/H] quadrature nodes.
        summary = gal.galactic_prior_summary()
        z = np.concatenate((np.linspace(0.0, 10.0, 251), np.linspace(10.5, 100.0, 180)))
        ages = np.geomspace(0.01, 13.8, 80)
        p_age = np.exp(gal.logp_age_from_feh(ages, -0.2))
        fehs = np.linspace(-1.4, 1.0, 17)
        w_feh = np.exp(-0.5 * ((fehs + 0.2) / 0.3) ** 2)
        n = ages.size * z.size
        loga = np.repeat(np.log10(ages * 1e9), z.size)
        thin = 0.0
        for feh, wf in zip(fehs, w_feh):
            prof = gal._disk_vertical_array(
                np.full(n, 8.2),
                np.tile(z, ages.size),
                0.025,
                loga,
                np.full(n, feh),
                True,
                _ROT,
                (2.74, 2.0),
                kin.THICK_DISK_SIGMA[2] ** 2,
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )
            column = np.trapezoid(np.exp(prof[0]).reshape(ages.size, -1), z, axis=1)
            thin += wf * np.trapezoid(p_age * column, ages) / np.trapezoid(p_age, ages)
        thin /= w_feh.sum()
        thick = np.trapezoid(np.exp(prof[1][: z.size]), z)
        np.testing.assert_allclose(summary["thin_column_height"], thin, rtol=5e-4)
        np.testing.assert_allclose(summary["thick_column_height"], thick, rtol=1e-6)
        np.testing.assert_allclose(
            summary["f_thick_solar"], 0.18 * thin / thick, rtol=5e-4
        )

    def test_potential_scales_with_the_circular_velocity(self):
        # Phi scales as v_c^2, so scaling v_circ_solar and thick_sigma_z by
        # one factor leaves the thick-disk column unchanged.
        base = gal.galactic_prior_summary()["thick_column_height"]
        scaled = gal.galactic_prior_summary(
            v_circ_solar=1.2 * _VC_SUN, thick_sigma_z=1.2 * kin.THICK_DISK_SIGMA[2]
        )["thick_column_height"]
        np.testing.assert_allclose(scaled, base, rtol=1e-12)

    def test_hot_thick_disk_column_includes_its_tail_above_10_kpc(self):
        # In the BV24 potential a hot disk is bound only logarithmically:
        # with sigma_z = 100 km/s, stopping the column at 10 kpc lost 5%.
        z = np.concatenate((np.linspace(0.0, 10.0, 501), np.linspace(10.1, 100.0, 900)))
        terms0 = kin._vertical_terms(1.0, 0.025)
        thick = [
            kin._thick_disk_vertical(
                kin._vertical_terms(1.0, t), terms0, 100.0**2 / _PHI_SCALE
            )
            for t in z
        ]
        column = np.trapezoid(np.exp(thick), z)
        summary = gal.galactic_prior_summary(thick_sigma_z=100.0)
        np.testing.assert_allclose(summary["thick_column_height"], column, rtol=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"thick_sigma_z": 35.0},
            {"feh_thin": -0.5},
            {"f_thick_surface": 0.3},
            {"halo_mass": 1e9, "q_halo": 0.6},
        ],
    )
    def test_prior_at_the_sun_uses_the_derived_fractions(self, kwargs):
        # Toward the north Galactic pole at 1 mpc, every component is at
        # the Sun's position, where each profile is 1.
        _, comp = logp_galactic_structure(
            np.array([1e-6]), (0.0, 90.0), return_components=True, **kwargs
        )
        thin, thick, halo = comp["number_density"]
        summary = gal.galactic_prior_summary(**kwargs)
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
            {"feh_thin_sigma": np.nan},
            {"halo_mass": 0.0},
            {"eta_halo_inner": np.nan},
            {"rho_thin_solar": -0.03},
        ],
    )
    def test_invalid_normalization_inputs_raise(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            logp_galactic_structure(np.array([1.0]), (0.0, 90.0), **kwargs)


class TestHaloShapeAndMass:
    """The halo is the Amarante et al. (2024) broken power law, normalized
    by the Lane et al. (2023) mass over 2 to 70 kpc."""

    def test_halo_matches_the_broken_power_law(self):
        # Points inside and beyond the break.
        R = np.array([4.0, 8.2, 15.0, 25.0, 10.0, 40.0])
        Z = np.array([3.0, 0.025, 12.0, 5.0, 35.0, 20.0])
        m = np.sqrt(R**2 + (Z / 0.77) ** 2 + 4.0)
        ln_n = np.where(
            m < 19.1, -2.85 * np.log(m), -2.85 * np.log(19.1) - 4.55 * np.log(m / 19.1)
        )
        m_sun = np.sqrt(8.2**2 + (0.025 / 0.77) ** 2 + 4.0)
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
