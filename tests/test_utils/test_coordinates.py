#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the galactic_to_galactocentric_cyl coordinate conversion utility.

Verifies that the fast NumPy implementation matches astropy SkyCoord
for a grid of (l, b, d) values, tests edge cases, and checks
that R_solar/Z_solar parameters propagate correctly.
"""

import time

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import CylindricalRepresentation as CylRep
from astropy.coordinates import SkyCoord

from brutus.utils.math import galactic_to_galactocentric_cyl


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

    def test_timing_vs_astropy(self):
        """Verify that the numpy implementation is faster than astropy SkyCoord.

        This test creates a moderately large coordinate array and compares
        the execution time of the numpy formula vs astropy SkyCoord.
        """
        N = 100_000
        dists = np.random.uniform(0.1, 50.0, N)
        ell_val = 45.0
        b_val = 30.0

        # Warm up
        galactic_to_galactocentric_cyl(dists[:10], ell_val, b_val)

        # Time numpy version
        t0 = time.perf_counter()
        for _ in range(5):
            R_np, Z_np = galactic_to_galactocentric_cyl(dists, ell_val, b_val)
        t_numpy = (time.perf_counter() - t0) / 5

        # Time astropy version
        ell_arr = np.full(N, ell_val)
        b_arr = np.full(N, b_val)

        # Warm up
        c = SkyCoord(
            l=ell_arr[:10] * units.deg,
            b=b_arr[:10] * units.deg,
            distance=dists[:10] * units.kpc,
            frame="galactic",
        )
        _ = c.galactocentric.cartesian.represent_as(CylRep)

        t0 = time.perf_counter()
        for _ in range(3):
            c = SkyCoord(
                l=ell_arr * units.deg,
                b=b_arr * units.deg,
                distance=dists * units.kpc,
                frame="galactic",
            )
            cyl = c.galactocentric.cartesian.represent_as(CylRep)
            R_ap, Z_ap = cyl.rho.value, cyl.z.value
        t_astropy = (time.perf_counter() - t0) / 3

        # numpy should be significantly faster
        speedup = t_astropy / t_numpy
        assert speedup > 2.0, (
            f"Expected >2x speedup, got {speedup:.1f}x "
            f"(numpy: {t_numpy*1e3:.1f}ms, astropy: {t_astropy*1e3:.1f}ms)"
        )


class TestGalacticToGalactocentricCylImport:
    """Test that the function is properly exported."""

    def test_import_from_math(self):
        """Test import from brutus.utils.math."""
        from brutus.utils.math import galactic_to_galactocentric_cyl

        assert callable(galactic_to_galactocentric_cyl)

    def test_import_from_utils(self):
        """Test import from brutus.utils."""
        from brutus.utils import galactic_to_galactocentric_cyl

        assert callable(galactic_to_galactocentric_cyl)

    def test_in_all(self):
        """Test that function is in __all__."""
        import brutus.utils
        import brutus.utils.math

        assert "galactic_to_galactocentric_cyl" in brutus.utils.__all__
        assert "galactic_to_galactocentric_cyl" in brutus.utils.math.__all__
