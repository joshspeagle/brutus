#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the velocity moments and their sky projection."""

import numpy as np
import pytest

from brutus.priors.galactic import logn_disk
from brutus.priors.kinematic import (
    DVCIRC_DR,
    PM_SGRA,
    THICK_DISK_R_SIGMA,
    THICK_DISK_SIGMA,
    U_SUN,
    V_PEC_SUN,
    disk_velocity_moments,
    halo_velocity_moments,
    solar_velocity,
    velocity_to_observables,
)
from brutus.utils.math import galactic_to_galactocentric_cyl

R_SOLAR = 8.2
# kpc / (km/s) in years, and mas per radian, from the definitions of the
# parsec, the astronomical unit and the Julian year.
KPC_KM = 1000.0 * 648000.0 / np.pi * 149597870.7
YR_S = 365.25 * 86400.0
MAS_PER_RAD = 180.0 / np.pi * 3600e3


def _circular_velocity(R_solar=R_SOLAR):
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


@pytest.mark.parametrize(
    "kwargs, lag",
    [
        ({}, 15.0),
        (
            dict(sigma_solar=THICK_DISK_SIGMA, R_scale=2.0, R_sigma=THICK_DISK_R_SIGMA),
            46.0,
        ),
    ],
    ids=["thin", "thick"],
)
def test_disk_parameters_give_the_measured_solar_lag(kwargs, lag):
    # Lags from Bensby et al. (2014).
    mean, _ = disk_velocity_moments(np.array([R_SOLAR]), **kwargs)
    assert abs(_circular_velocity() - mean[0, 1] - lag) < 0.5


def test_disk_rotation_solves_the_jeans_equation_for_the_logn_disk_density():
    R = np.linspace(3.0, 20.0, 18)
    sigma = (40.0, 25.0, 20.0)
    density = dict(R_scale=2.3, R_smooth=1.5)
    kw = dict(R_sigma=4.0, **density)
    mean, cov = disk_velocity_moments(R, sigma_solar=sigma, **kw)

    h = 1e-5
    dlnnu = (
        logn_disk(R * (1 + h), 0.0, **density) - logn_disk(R * (1 - h), 0.0, **density)
    ) / (2 * h)
    var_R = cov[:, 0, 0]
    _, cov_hi = disk_velocity_moments(R * (1 + h), sigma_solar=sigma, **kw)
    _, cov_lo = disk_velocity_moments(R * (1 - h), sigma_solar=sigma, **kw)
    dlnvar = (np.log(cov_hi[:, 0, 0]) - np.log(cov_lo[:, 0, 0])) / (2 * h)
    vc = (_circular_velocity() + DVCIRC_DR * (R - R_SOLAR)) * np.tanh(
        R / density["R_smooth"]
    )
    # Binney & Tremaine (2008), eq. 4.228, with no tilt term.
    lhs = vc**2 - mean[:, 1] ** 2
    rhs = cov[:, 1, 1] - var_R * (1 + dlnnu + dlnvar)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6)
    np.testing.assert_allclose(mean[:, [0, 2]], 0.0)


def test_disk_rotation_stays_below_circular_and_vanishes_at_the_centre():
    R = np.linspace(0.0, 4.0, 41)
    mean, _ = disk_velocity_moments(R)
    vc = (_circular_velocity() + DVCIRC_DR * (R - R_SOLAR)) * np.tanh(R / 2.0)
    assert mean[0, 1] == 0.0
    assert np.all(mean[:, 1] <= vc * (1 + 1e-12))


def test_disk_does_not_rotate_where_the_jeans_equation_has_no_solution():
    mean, _ = disk_velocity_moments(
        np.array([R_SOLAR]), sigma_solar=(300.0, 10.0, 10.0)
    )
    assert mean[0, 1] == 0.0


def test_halo_ellipsoid_keeps_its_spherical_dispersions():
    rng = np.random.default_rng(3)
    R = np.concatenate([[5.0, 0.0], rng.uniform(0.0, 30.0, 10)])
    Z = np.concatenate([[0.0, 5.0], rng.uniform(-30.0, 30.0, 10)])
    sig = (141.0, 75.0, 85.0)
    mean, cov = halo_velocity_moments(R, Z, sigma_sph=sig, v_phi=20.0)
    # In the plane, v_R is v_r; on the axis, v_z is v_r.
    np.testing.assert_allclose(np.diag(cov[0]), [sig[0] ** 2, sig[2] ** 2, sig[1] ** 2])
    np.testing.assert_allclose(np.diag(cov[1]), [sig[1] ** 2, sig[2] ** 2, sig[0] ** 2])
    for c in cov:
        np.testing.assert_allclose(
            np.linalg.eigvalsh(c), np.sort(np.square(sig)), rtol=1e-10
        )
    np.testing.assert_allclose(mean, np.tile([0.0, 20.0, 0.0], (len(R), 1)))


@pytest.mark.parametrize(
    "call",
    [
        lambda: velocity_to_observables(
            np.array([0.0]), (0, 0), np.zeros((1, 3)), np.zeros((1, 3, 3))
        ),
        lambda: velocity_to_observables(
            np.array([1.0]), (0, 0), np.zeros((2, 3)), np.zeros((1, 3, 3))
        ),
        lambda: disk_velocity_moments(np.array([8.0]), R_sigma=0.0),
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
        lambda: disk_velocity_moments(np.array([-1.0])),
        lambda: halo_velocity_moments(np.array([1.0]), np.array([np.nan])),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
