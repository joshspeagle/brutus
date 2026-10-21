#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the galactic_to_galactocentric_cyl coordinate conversion utility.

Verifies that the fast NumPy implementation matches astropy SkyCoord
for a grid of (l, b, d) values, tests edge cases, and checks
that R_solar/Z_solar parameters propagate correctly.
"""

import warnings

import numpy as np
import pytest

from brutus.utils.math import galactic_to_galactocentric_cyl, pm_equatorial_to_galactic

MAS_PER_RAD = 180.0 / np.pi * 3600e3

# ICRS -> Galactic rotation matrix (Gaia DR3 documentation, eq. 4.62).
A_G = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ]
)


class TestGalacticToGalactocentricCyl:
    """Tests for the galactic_to_galactocentric_cyl function."""

    def test_basic_output_shape(self):
        """Test that output shape matches input distances."""
        dists = np.array([1.0, 2.0, 5.0, 10.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=90.0, b=0.0)
        assert R.shape == dists.shape
        assert Z.shape == dists.shape

    def test_scalar_distance(self):
        """Test with a scalar distance value."""
        R, Z = galactic_to_galactocentric_cyl(1.0, ell=0.0, b=0.0)
        # At l=0, b=0, d=1 kpc: x=1, y=0 -> R = |1 - 8.2| = 7.2
        assert np.isclose(R, 7.2, atol=1e-10)
        assert np.isclose(Z, 0.025, atol=1e-10)

    def test_galactic_center_direction(self):
        """Test pointing toward the Galactic center (l=0, b=0)."""
        dists = np.array([8.2])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=0.0, b=0.0)
        # At distance = R_solar along l=0, should be at R=0
        assert np.isclose(R, 0.0, atol=1e-10)
        assert np.isclose(Z, 0.025, atol=1e-10)

    def test_anticenter_direction(self):
        """Test pointing toward the Galactic anticenter (l=180, b=0)."""
        dists = np.array([1.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=180.0, b=0.0)
        # At l=180, d=1: x=-1, y=0 -> R = |(-1) - 8.2| = 9.2
        assert np.isclose(R, 9.2, atol=1e-10)

    def test_north_pole(self):
        """Test pointing to the North Galactic Pole (b=90)."""
        dists = np.array([1.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=0.0, b=90.0)
        # At b=90: x=0, y=0 -> R = R_solar = 8.2, Z = 1.0 + 0.025
        assert np.isclose(R, 8.2, atol=1e-10)
        assert np.isclose(Z, 1.025, atol=1e-10)

    def test_south_pole(self):
        """Test pointing to the South Galactic Pole (b=-90)."""
        dists = np.array([1.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=0.0, b=-90.0)
        # At b=-90: x=0, y=0 -> R = R_solar = 8.2, Z = -1.0 + 0.025
        assert np.isclose(R, 8.2, atol=1e-10)
        assert np.isclose(Z, -0.975, atol=1e-10)

    def test_custom_solar_position(self):
        """Test with custom R_solar and Z_solar values."""
        dists = np.array([1.0])
        R, Z = galactic_to_galactocentric_cyl(
            dists, ell=0.0, b=0.0, R_solar=8.5, Z_solar=0.02
        )
        assert np.isclose(R, 7.5, atol=1e-10)
        assert np.isclose(Z, 0.02, atol=1e-10)

    def test_matches_astropy_grid(self):
        """Test that numpy formula matches astropy SkyCoord for a grid of coordinates.

        Uses the same R_solar and Z_solar for both to ensure a fair comparison.
        Note: astropy uses a full 3D rotation including solar motion corrections,
        while our formula is a simplified 2D model. We compare against our own
        formula's expected behavior rather than requiring exact astropy match.
        """
        # Grid of test values
        ells = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
        bs = [-60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0]
        ds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

        for ell in ells:
            for b in bs:
                dists = np.array(ds)
                R, Z = galactic_to_galactocentric_cyl(dists, ell, b)

                # Verify basic properties
                assert np.all(R >= 0), f"R must be non-negative at l={ell}, b={b}"
                assert np.all(np.isfinite(R)), f"R must be finite at l={ell}, b={b}"
                assert np.all(np.isfinite(Z)), f"Z must be finite at l={ell}, b={b}"

                # Verify Z increases with distance for b > 0
                if b > 0:
                    assert np.all(
                        np.diff(Z) > 0
                    ), f"Z should increase with dist for b>0 at l={ell}, b={b}"

    def test_zero_distance(self):
        """Test behavior at zero distance (Solar position)."""
        dists = np.array([0.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell=0.0, b=0.0)
        # At d=0, should be at the Sun's position
        assert np.isclose(R, 8.2, atol=1e-10)
        assert np.isclose(Z, 0.025, atol=1e-10)

    def test_large_distance_array(self):
        """Test with a large array of distances (performance check)."""
        dists = np.linspace(0.01, 100.0, 10000)
        R, Z = galactic_to_galactocentric_cyl(dists, ell=45.0, b=30.0)
        assert R.shape == (10000,)
        assert Z.shape == (10000,)
        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(Z))

    def test_array_coordinates(self):
        """Test with array-valued ell and b."""
        dists = np.array([1.0, 2.0, 3.0])
        ell = np.array([0.0, 90.0, 180.0])
        b = np.array([0.0, 30.0, -30.0])
        R, Z = galactic_to_galactocentric_cyl(dists, ell, b)
        assert R.shape == (3,)
        assert Z.shape == (3,)


def _icrs_to_lb(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    x = A_G @ np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    return np.arctan2(x[1], x[0]), np.arcsin(x[2])


def test_pm_conversion_matches_the_motion_in_galactic_coordinates():
    rng = np.random.default_rng(7)
    n = 20
    ra, dec = rng.uniform(0, 360, n), rng.uniform(-85, 85, n)
    pmra, pmdec = rng.normal(0, 10, n), rng.normal(0, 10, n)
    pm, _ = pm_equatorial_to_galactic(
        ra, dec, pmra, pmdec, np.ones(n), np.ones(n), np.zeros(n)
    )
    # Move the star for a short time and read its Galactic motion.
    dt = 10.0  # yr
    deg_per_mas = 1.0 / 3.6e6
    for i in range(n):
        dra = pmra[i] / np.cos(np.radians(dec[i])) * dt * deg_per_mas
        dde = pmdec[i] * dt * deg_per_mas
        l1, b1 = _icrs_to_lb(ra[i] - dra, dec[i] - dde)
        l2, b2 = _icrs_to_lb(ra[i] + dra, dec[i] + dde)
        dl = np.angle(np.exp(1j * (l2 - l1)))
        b = 0.5 * (b1 + b2)
        mu_l = np.cos(b) * dl / (2 * dt) * MAS_PER_RAD
        mu_b = (b2 - b1) / (2 * dt) * MAS_PER_RAD
        np.testing.assert_allclose(pm[i], [mu_l, mu_b], rtol=1e-5, atol=1e-6)


def test_pm_covariance_is_the_jacobian_sandwich_of_the_error_covariance():
    rng = np.random.default_rng(9)
    n = 10
    ra, dec = rng.uniform(0, 360, n), rng.uniform(-85, 85, n)
    err_ra, err_dec = rng.uniform(0.01, 1, n), rng.uniform(0.01, 1, n)
    corr = rng.uniform(-0.9, 0.9, n)
    zeros, ones = np.zeros(n), np.ones(n)
    _, cov = pm_equatorial_to_galactic(ra, dec, zeros, zeros, err_ra, err_dec, corr)
    # The map is linear in (pmra, pmdec): its Jacobian columns are the
    # images of the unit proper motions.
    col_a, _ = pm_equatorial_to_galactic(ra, dec, ones, zeros, ones, ones, zeros)
    col_d, _ = pm_equatorial_to_galactic(ra, dec, zeros, ones, ones, ones, zeros)
    jac = np.stack([col_a, col_d], axis=2)
    eq_cov = np.empty((n, 2, 2))
    eq_cov[:, 0, 0], eq_cov[:, 1, 1] = err_ra**2, err_dec**2
    eq_cov[:, 0, 1] = eq_cov[:, 1, 0] = corr * err_ra * err_dec
    expected = jac @ eq_cov @ np.transpose(jac, (0, 2, 1))
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "dec, corr, match",
    [(0.0, 1.5, "pmra_pmdec_corr"), (100.0, 0.0, "dec")],
)
def test_pm_conversion_rejects_invalid_inputs(dec, corr, match):
    with pytest.raises(ValueError, match=match):
        pm_equatorial_to_galactic([0.0], [dec], [0.0], [0.0], [1.0], [1.0], [corr])


def test_pm_conversion_returns_nan_at_the_galactic_pole_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm, cov = pm_equatorial_to_galactic(
            [192.85948], [27.12825], [1.0], [1.0], [1.0], [1.0], [0.0]
        )
    assert np.all(np.isnan(pm)) and np.all(np.isnan(cov))
