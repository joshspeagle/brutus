#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the velocity moments and their sky projection."""

import os

import numpy as np
import pytest

from brutus.priors import kinematic as kin
from brutus.priors.galactic import logn_disk, logp_age_from_feh
from brutus.priors.kinematic import (
    HALO_DELTA_LOBE,
    HALO_F_ISO,
    HALO_R,
    HALO_SIGMA_R_ISO,
    HALO_SIGMA_R_LOBE,
    HALO_SIGMA_T_ISO,
    HALO_SIGMA_T_LOBE,
    PM_SGRA,
    THICK_DISK_R_RANGE,
    THIN_DISK_R_RANGE,
    THIN_DISK_SIGMA0,
    U_SUN,
    V_PEC_SUN,
    VC_SOURCES,
    halo_velocity_moments,
    solar_velocity,
    thick_disk_velocity_moments,
    thin_disk_velocity_moments,
    velocity_to_observables,
)
from brutus.utils.math import galactic_to_galactocentric_cyl

R_SOLAR = 8.2
# kpc / (km/s) in years, and mas per radian, from the definitions of the
# parsec, the astronomical unit and the Julian year.
KPC_KM = 1000.0 * 648000.0 / np.pi * 149597870.7
YR_S = 365.25 * 86400.0
MAS_PER_RAD = 180.0 / np.pi * 3600e3


def _vc_solar(R_solar=R_SOLAR):
    return solar_velocity(R_solar)[1] - V_PEC_SUN


def test_star_at_rest_at_galactic_centre_shows_sgra_proper_motion():
    obs, _ = velocity_to_observables(
        np.array([R_SOLAR]), (0.0, 0.0), np.zeros((1, 3)), np.zeros((1, 3, 3))
    )
    np.testing.assert_allclose(obs[0], [PM_SGRA[0], PM_SGRA[1], -U_SUN], rtol=1e-12)


def _sky_motion(dist, ell, b, v_cyl, v_sun):
    """Proper motions and radial velocity by moving the star on the sky."""
    ell, b = np.radians(ell), np.radians(b)
    pos = dist * np.array([np.cos(b) * np.cos(ell), np.cos(b) * np.sin(ell), np.sin(b)])
    phi = np.arctan2(pos[1], pos[0] - R_SOLAR)
    e_R = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_phi = np.array([np.sin(phi), -np.cos(phi), 0.0])
    v_h = v_cyl[0] * e_R + v_cyl[1] * e_phi + np.array([0, 0, v_cyl[2]]) - v_sun

    def lb(p):
        return np.arctan2(p[1], p[0]), np.arctan2(p[2], np.hypot(p[0], p[1]))

    # Time in units of kpc / (km/s), so pos + v_h * dt is in kpc.
    dt = 1e-7 * dist / np.linalg.norm(v_h)
    l1, b1 = lb(pos - v_h * dt)
    l2, b2 = lb(pos + v_h * dt)
    to_mas_yr = MAS_PER_RAD * YR_S / KPC_KM
    dl = np.angle(np.exp(1j * (l2 - l1)))
    mu_l = np.cos(b) * dl / (2 * dt) * to_mas_yr
    mu_b = (b2 - b1) / (2 * dt) * to_mas_yr
    return np.array([mu_l, mu_b, pos @ v_h / dist])


def test_projection_matches_the_motion_of_the_star_on_the_sky():
    rng = np.random.default_rng(42)
    v_sun = solar_velocity(R_SOLAR)
    for _ in range(20):
        ell, b = rng.uniform(0, 360), rng.uniform(-80, 80)
        dist = rng.uniform(0.05, 20.0)
        v_cyl = rng.normal(0.0, 150.0, 3)
        obs, _ = velocity_to_observables(
            np.array([dist]), (ell, b), v_cyl[None], np.zeros((1, 3, 3))
        )
        np.testing.assert_allclose(
            obs[0], _sky_motion(dist, ell, b, v_cyl, v_sun), rtol=1e-6, atol=1e-9
        )


def test_projected_covariance_uses_the_projection_of_the_mean():
    rng = np.random.default_rng(1)
    dists = rng.uniform(0.1, 10.0, 5)
    coord = (123.0, -35.0)
    # The (v_R, v_z) block may be correlated; v_phi is independent of both.
    cov = np.zeros((5, 3, 3))
    a = rng.normal(size=(5, 2, 2))
    cov[np.ix_(range(5), [0, 2], [0, 2])] = a @ np.transpose(a, (0, 2, 1)) * 1e3
    cov[:, 1, 1] = rng.uniform(100.0, 1e4, 5)
    base, obs_cov = velocity_to_observables(dists, coord, np.zeros((5, 3)), cov)
    # The map is affine in the mean, so its Jacobian comes from unit steps.
    jac = np.empty((5, 3, 3))
    for k in range(3):
        step = np.zeros((5, 3))
        step[:, k] = 1.0
        jac[:, :, k] = velocity_to_observables(dists, coord, step, cov)[0] - base
    expected = jac @ cov @ np.transpose(jac, (0, 2, 1))
    np.testing.assert_allclose(obs_cov, expected, rtol=1e-9, atol=1e-12)


def test_rigid_rotation_shows_one_proper_motion_and_no_radial_velocity():
    v_circ = 230.0
    v_sun = np.array([0.0, v_circ, 0.0])
    for ell in (10.0, 90.0, 180.0, 300.0):
        dists = np.array([0.5, 2.0, 6.0])
        R, _ = galactic_to_galactocentric_cyl(dists, ell, 0.0, R_solar=R_SOLAR)
        mean = np.zeros((3, 3))
        mean[:, 1] = v_circ * R / R_SOLAR
        obs, _ = velocity_to_observables(
            dists, (ell, 0.0), mean, np.zeros((3, 3, 3)), v_sun=v_sun
        )
        omega = -v_circ / (4.740470463533348 * R_SOLAR)
        np.testing.assert_allclose(obs[:, 0], omega, rtol=1e-12)
        np.testing.assert_allclose(obs[:, 1:], 0.0, atol=1e-10)


def _vc(R, R_solar=R_SOLAR, v_circ_solar=None):
    """The rotation curve kernel at radii R."""
    vc = _vc_solar(R_solar) if v_circ_solar is None else v_circ_solar
    return np.array(
        [kin._circular_velocity(r, R_solar, vc)[0] for r in np.atleast_1d(R)]
    )


def _measured_curve(source, R_range, max_err=np.inf):
    """Radii scaled to R_SOLAR and v_c / v_c,sun of one source in the data."""
    path = os.path.join(os.path.dirname(kin.__file__), "rotation_curve.dat")
    R, v, err, src = np.loadtxt(path).T
    R_sun, vc_sun = VC_SOURCES[source]
    keep = (src == source) & (err <= max_err)
    keep &= (R >= R_range[0]) & (R <= R_range[1])
    return R[keep] * R_SOLAR / R_sun, v[keep] / vc_sun


def test_rotation_curve_follows_the_measured_circular_velocities():
    # Eilers et al. (2019) bins with errors under 5 km/s, 5.3-21 kpc.
    R, shape = _measured_curve(1, (5.3, 21.3), max_err=5.0)
    rel = _vc(R) / _vc_solar() / shape - 1
    assert np.abs(rel).max() < 0.03 and np.sqrt(np.mean(rel**2)) < 0.015
    # Sofue & Kohno (2025) at 0.1-4.5 kpc, smoothed over the bar's wiggles.
    R, shape = _measured_curve(0, (0.1, 4.5))
    rel = _vc(R) / _vc_solar() / shape - 1
    assert np.sqrt(np.mean(rel**2)) < 0.05


@pytest.mark.parametrize("R_solar, v_circ_solar", [(8.0, 230.0), (8.5, 245.0)])
def test_rotation_curve_scales_with_the_solar_radius_and_velocity(
    R_solar, v_circ_solar
):
    R = np.array([1.0, 4.0, 8.0, 15.0, 30.0])
    expected = v_circ_solar / _vc_solar() * _vc(R * R_SOLAR / R_solar)
    np.testing.assert_allclose(_vc(R, R_solar, v_circ_solar), expected, rtol=1e-12)
    np.testing.assert_allclose(_vc(R_solar, R_solar, v_circ_solar), v_circ_solar)


def test_rotation_curve_slope_is_the_derivative_of_its_values():
    R = np.linspace(0.2, 30.0, 300)
    dlnvc = np.array([kin._circular_velocity(r, R_SOLAR, _vc_solar())[1] for r in R])
    np.testing.assert_allclose(dlnvc, _dln_dlnR(_vc, R), atol=1e-6)


def _thin(R, age=5.0, feh=0.0, **kw):
    R = np.atleast_1d(np.asarray(R, dtype=float))
    return thin_disk_velocity_moments(R, np.zeros_like(R), age, feh, **kw)


def _dln_dlnR(f, R, h=1e-5):
    """Central difference of d ln f / d ln R."""
    return (np.log(f(R * (1 + h))) - np.log(f(R * (1 - h)))) / (2 * h)


def test_thin_disk_dispersions_use_the_mean_angular_momentum():
    # Sharma et al. (2021): sigma = sigma_0 f_L(L_z) at 10 Gyr, [Fe/H] = 0
    # and z = 0, with L_z the population's mean R v_phi (to 0.06%).
    # Near the edges of THIN_DISK_R_RANGE the fixed point once oscillated.
    R = np.array([3.05, 4.0, R_SOLAR, 14.0, 19.95])
    mean, cov = _thin(R, age=10.0)
    vc = _vc(R)
    L_solar = R_SOLAR * _vc_solar()
    L_range = tuple(r * _vc(r)[0] for r in THIN_DISK_R_RANGE)
    for i in range(len(R)):
        f_R, f_z, _ = kin._thin_disk_fL(R[i] * mean[i, 1], L_solar, L_range)
        np.testing.assert_allclose(
            np.sqrt(cov[i, [0, 2], [0, 2]]),
            np.multiply(THIN_DISK_SIGMA0, (f_R, f_z)),
            rtol=1e-3,
        )
        # The circular L_z = R v_c gives other dispersions.
        f_c = kin._thin_disk_fL(R[i] * vc[i], L_solar, L_range)[0]
        assert abs(f_c / f_R - 1) > 5e-3


def test_thin_disk_population_matches_the_measured_low_alpha_disk():
    # Average over the local thin-disk age prior of brutus at [Fe/H] = -0.2.
    # Anguiano et al. (2020, Table 1), low-alpha disk: lag 240 - 229.43 =
    # 10.6 km/s, sigma_phi / sigma_R = 0.66, sigma_z / sigma_R = 0.49.
    ages = np.linspace(0.05, 13.8, 400)
    w = np.exp(logp_age_from_feh(ages, feh_mean=-0.2))
    w /= w.sum()
    mean, cov = thin_disk_velocity_moments(
        np.full_like(ages, R_SOLAR), np.zeros_like(ages), ages, -0.2
    )
    vphi = w @ mean[:, 1]
    var = w @ np.diagonal(cov, axis1=1, axis2=2)
    var[1] += w @ (mean[:, 1] - vphi) ** 2
    sig = np.sqrt(var)
    assert abs(_vc_solar() - vphi - 10.6) < 3.0
    assert abs(sig[1] / sig[0] - 0.66) < 0.05
    assert abs(sig[2] / sig[0] - 0.49) < 0.05


def test_thick_disk_rotation_matches_the_measured_high_alpha_lag():
    # Anguiano et al. (2020, Table 1), high-alpha disk: lag 240 - 191.82 =
    # 48.2 km/s. The model predicts it; nothing is fitted to it.
    mean, _ = thick_disk_velocity_moments(np.array([R_SOLAR]))
    assert abs(_vc_solar() - mean[0, 1] - 48.2) < 3.0


@pytest.mark.parametrize("disk", ["thin", "thick"])
def test_disk_rotation_solves_the_jeans_equation_for_the_logn_disk_density(disk):
    R = np.linspace(4.5, 12.5, 17)
    density = dict(R_scale=2.3, R_smooth=1.5)
    if disk == "thin":

        def moments(R):
            return _thin(R, age=7.0, feh=-0.3, **density)

    else:

        def moments(R):
            return thick_disk_velocity_moments(R, R_sigma=9.0, **density)

    mean, cov = moments(R)
    dlnnu = _dln_dlnR(lambda x: np.exp(logn_disk(x, 0.0, **density)), R)
    dlnvar = _dln_dlnR(lambda x: moments(x)[1][:, 0, 0], R)
    # Binney & Tremaine (2008), eq. 4.228, with no tilt term.
    rhs = cov[:, 1, 1] - cov[:, 0, 0] * (1 + dlnnu + dlnvar)
    jeans = np.sqrt(_vc(R) ** 2 - rhs)
    # The thin disk takes d ln L_z / d ln R of a circular orbit for its
    # mean L_z, which moves v_phi by up to 2.2 km/s at 3-20 kpc.
    np.testing.assert_allclose(
        mean[:, 1], jeans, rtol=1e-7, atol=2.5 if disk == "thin" else 0.0
    )
    np.testing.assert_allclose(mean[:, [0, 2]], 0.0)


def test_thin_disk_azimuthal_dispersion_follows_the_epicyclic_ratio():
    R = np.linspace(3.0, 20.0, 18)
    _, cov = _thin(R)
    dlnvc = _dln_dlnR(_vc, R)
    np.testing.assert_allclose(cov[:, 1, 1] / cov[:, 0, 0], (1 + dlnvc) / 2, rtol=1e-6)


@pytest.mark.parametrize("disk", ["thin", "thick"])
def test_disk_rotation_stays_below_circular_and_vanishes_at_the_centre(disk):
    R = np.linspace(0.0, 4.0, 41)
    mean = (_thin(R) if disk == "thin" else thick_disk_velocity_moments(R))[0]
    assert mean[0, 1] == 0.0
    assert np.all(mean[:, 1] <= _vc(R) * (1 + 1e-12))


def test_disk_dispersions_keep_their_edge_values_outside_the_fitted_radii():
    # The thin disk is clamped in L_z, at the L_z of circular orbits at the
    # edges of THIN_DISK_R_RANGE; its mean L_z lies below them, so the
    # clamp holds inside 3 kpc and from a little beyond 20 kpc.
    lo, hi = THIN_DISK_R_RANGE
    _, cov = _thin(np.array([0.5 * lo, 0.25 * lo, 1.5 * hi, 3 * hi]))
    np.testing.assert_allclose(cov[1, [0, 2], [0, 2]], cov[0, [0, 2], [0, 2]])
    np.testing.assert_allclose(cov[3, [0, 2], [0, 2]], cov[2, [0, 2], [0, 2]])
    lo, hi = THICK_DISK_R_RANGE
    _, cov = thick_disk_velocity_moments(np.array([lo, 0.5 * lo, hi, 10 * hi]))
    np.testing.assert_allclose(cov[1], cov[0])
    np.testing.assert_allclose(cov[3], cov[2])


def test_velocity_ellipsoids_stay_positive_definite_at_all_radii():
    R = np.linspace(0.0, 300.0, 3001)
    Z = 0.3 * R
    for _, cov in (_thin(R), thick_disk_velocity_moments(R)):
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    _, _, cov = halo_velocity_moments(R, Z)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_rotation_curve_is_flat_beyond_the_measured_radii():
    # The table ends at 37 kpc, where its slope is below 1e-8.
    R = np.array([40.0, 300.0])
    np.testing.assert_allclose(_vc(R), _vc(R[0])[0])
    _, cov = _thin(R)
    # sigma_phi^2 / sigma_R^2 = (1 + d ln v_c / d ln R) / 2 = 1/2 where flat.
    np.testing.assert_allclose(cov[:, 1, 1] / cov[:, 0, 0], 0.5, rtol=1e-6)


def test_disk_does_not_rotate_where_the_jeans_equation_has_no_solution():
    mean, _ = thick_disk_velocity_moments(
        np.array([R_SOLAR]), sigma_solar=(300.0, 10.0, 10.0)
    )
    assert mean[0, 1] == 0.0


def test_halo_components_reproduce_the_lancaster_table_at_the_bin_centres():
    # In the plane (Z = 0), v_R is v_r and v_z is v_theta.
    R = np.array(HALO_R)
    weight, mean, cov = halo_velocity_moments(R, np.zeros_like(R))
    f = np.array(HALO_F_ISO)
    np.testing.assert_allclose(weight, np.stack([f, (1 - f) / 2, (1 - f) / 2], 1))
    np.testing.assert_allclose(mean[:, 0], 0.0)
    for k, sign in ((1, 1.0), (2, -1.0)):
        np.testing.assert_allclose(mean[:, k, 0], sign * np.array(HALO_DELTA_LOBE))
        # Lancaster et al. print mu_vphi = (-4.49, 11.54, 14.13, -27.39)
        # with v_phi < 0 for disc rotation; here v_phi > 0 is rotation.
        np.testing.assert_allclose(mean[:, k, 1], [4.49, -11.54, -14.13, 27.39])
        np.testing.assert_allclose(mean[:, k, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(cov[:, 0, 0, 0], np.square(HALO_SIGMA_R_ISO))
    np.testing.assert_allclose(cov[:, 0, 2, 2], np.square(HALO_SIGMA_T_ISO))
    np.testing.assert_allclose(cov[:, 1, 0, 0], np.square(HALO_SIGMA_R_LOBE))
    np.testing.assert_allclose(cov[:, 2, 1, 1], np.square(HALO_SIGMA_T_LOBE))


def test_halo_components_keep_their_spherical_ellipsoids_off_the_plane():
    rng = np.random.default_rng(3)
    R = np.concatenate([[0.0, 3.0], rng.uniform(0.0, 60.0, 10)])
    Z = np.concatenate([[5.0, 0.0], rng.uniform(-60.0, 60.0, 10)])
    _, mean, cov = halo_velocity_moments(R, Z)
    r = np.hypot(R, Z)
    unit_r = np.stack([R / r, Z / r], axis=1)
    for k in range(3):
        rz = cov[:, k][:, [0, 2]][:, :, [0, 2]]
        # The (R, z) block has eigenvalues (sigma_t^2, sigma_r^2), with the
        # larger one along the radial direction; sigma_phi = sigma_t.
        eig = np.linalg.eigvalsh(rz)
        np.testing.assert_allclose(eig[:, 0], cov[:, k, 1, 1], rtol=1e-10)
        radial = np.einsum("ni,nij,nj->n", unit_r, rz, unit_r)
        np.testing.assert_allclose(radial, eig[:, 1], rtol=1e-10)
        # The lobe means point along the radial direction.
        np.testing.assert_allclose(
            mean[:, k, 0] * unit_r[:, 1], mean[:, k, 2] * unit_r[:, 0], atol=1e-9
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: velocity_to_observables(
            np.array([0.0]), (0, 0), np.zeros((1, 3)), np.zeros((1, 3, 3))
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 0), np.zeros((2, 3)), np.zeros((1, 3, 3))
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 0), np.zeros((1, 3)), np.ones((1, 3, 3))
        ),
        lambda: velocity_to_observables(
            np.array([1.0]),
            (0, 0),
            np.zeros((1, 3)),
            np.array([[[1.0, 0, 5], [0, 1, 0], [-5, 0, 1]]]),
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 0), np.zeros((1, 3)), np.zeros((1, 3, 3)), R_solar=-8.2
        ),
        lambda: thick_disk_velocity_moments(np.array([8.0]), R_sigma=0.0),
        lambda: thick_disk_velocity_moments(np.array([-1.0])),
        lambda: _thin(8.0, age=-1.0),
        lambda: _thin(8.0, feh=np.nan),
        lambda: _thin(8.0, v_circ_solar=-5.0),
        lambda: _thin(8.0, v_circ_solar=0.0),
        lambda: thick_disk_velocity_moments(np.array([8.0]), v_circ_solar=np.nan),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 0), np.zeros((1, 3)), -np.eye(3)[None]
        ),
        lambda: velocity_to_observables(
            np.array([1.0]),
            (0, 0),
            np.zeros((1, 3)),
            np.array([[[1.0, 0, 5], [0, 1, 0], [5, 0, 1]]]),
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (np.nan, 0), np.zeros((1, 3)), np.zeros((1, 3, 3))
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 95.0), np.zeros((1, 3)), np.zeros((1, 3, 3))
        ),
        lambda: halo_velocity_moments(np.array([1.0]), np.array([np.nan])),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
