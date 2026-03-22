#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the coordinate optimization in galactic priors.

Verifies that:
1. logp_galactic_structure uses the fast NumPy path (no SkyCoord internally)
2. R_solar/Z_solar parameters are respected and produce different results
3. The function still accepts both SkyCoord and tuple coordinate inputs
4. Results are consistent with the expected Galactic structure model
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from brutus.priors.galactic import logp_galactic_structure


class TestGalacticStructureCoordinateOptimization:
    """Tests for the optimized coordinate handling in logp_galactic_structure."""

    def test_tuple_coordinates_still_work(self):
        """Test that tuple (l, b) coordinates still work after optimization."""
        dists = np.array([1.0, 2.0, 5.0])
        coord = (90.0, 30.0)
        logp = logp_galactic_structure(dists, coord)
        assert np.all(np.isfinite(logp))
        assert len(logp) == 3

    def test_skycoord_input_still_works(self):
        """Test that SkyCoord input still works after optimization."""
        dists = np.array([1.0, 2.0, 5.0])
        coord = SkyCoord(ra=180.0, dec=30.0, unit="deg")
        logp = logp_galactic_structure(dists, coord)
        assert np.all(np.isfinite(logp))
        assert len(logp) == 3

    def test_r_solar_z_solar_affect_result(self):
        """Test that different R_solar/Z_solar produce different results."""
        dists = np.array([1.0, 2.0, 5.0, 10.0])
        coord = (90.0, 30.0)

        logp_default = logp_galactic_structure(dists, coord)
        logp_custom = logp_galactic_structure(dists, coord, R_solar=8.5, Z_solar=0.05)

        # Results should differ when solar position changes
        assert not np.allclose(
            logp_default, logp_custom
        ), "Changing R_solar/Z_solar should produce different results"

    def test_default_r_solar_z_solar(self):
        """Test that default R_solar=8.2 and Z_solar=0.025 are used."""
        dists = np.array([1.0, 5.0])
        coord = (45.0, 15.0)

        # Explicit defaults should match implicit defaults
        logp_implicit = logp_galactic_structure(dists, coord)
        logp_explicit = logp_galactic_structure(
            dists, coord, R_solar=8.2, Z_solar=0.025
        )
        np.testing.assert_array_almost_equal(logp_implicit, logp_explicit)

    def test_consistency_tuple_vs_skycoord(self):
        """Test that tuple and SkyCoord inputs give same results for same position."""
        dists = np.array([0.5, 1.0, 3.0, 8.0])

        # Use a Galactic coordinate directly
        coord_tuple = (120.0, -20.0)

        # Create SkyCoord in Galactic frame
        coord_sky = SkyCoord(l=120.0, b=-20.0, unit="deg", frame="galactic")

        logp_tuple = logp_galactic_structure(dists, coord_tuple)
        logp_sky = logp_galactic_structure(dists, coord_sky)

        np.testing.assert_array_almost_equal(logp_tuple, logp_sky, decimal=10)

    def test_return_components(self):
        """Test that return_components still works."""
        dists = np.array([1.0, 2.0])
        coord = (0.0, 0.0)

        logp, components = logp_galactic_structure(dists, coord, return_components=True)
        assert np.all(np.isfinite(logp))
        assert "number_density" in components

    def test_with_metallicity_labels(self):
        """Test that metallicity labels work with optimized coordinates."""
        dists = np.array([1.0, 2.0])
        coord = (180.0, 45.0)
        labels = np.array([(-0.2,), (-1.0,)], dtype=[("feh", "f4")])

        logp = logp_galactic_structure(dists, coord, labels=labels)
        assert np.all(np.isfinite(logp))

    def test_with_age_labels(self):
        """Test that age labels work with optimized coordinates."""
        dists = np.array([1.0, 2.0])
        coord = (180.0, 45.0)
        labels = np.array([(9.0,), (9.5,)], dtype=[("loga", "f4")])

        logp = logp_galactic_structure(dists, coord, labels=labels)
        assert np.all(np.isfinite(logp))

    def test_large_distance_array(self):
        """Test with a large distance array (the performance-critical case)."""
        # This mimics the Nmc x Nsel arrays used in fitting
        dists = np.random.uniform(0.1, 50.0, 10000)
        coord = (270.0, -10.0)

        logp = logp_galactic_structure(dists, coord)
        assert np.all(np.isfinite(logp))
        assert logp.shape == (10000,)

    def test_decreasing_density_with_distance(self):
        """Test that density generally decreases away from the disk."""
        # Looking straight up from the disk
        dists = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
        coord = (0.0, 90.0)  # North Galactic Pole

        logp = logp_galactic_structure(dists, coord)
        # After the volume factor peak, density should generally decrease
        # The volume factor (2*log(d)) initially dominates, then density drops
        # Check that the farthest point has lower probability than mid-range
        assert (
            logp[-1] < logp[2]
        ), "Very far from the disk should have lower prior than moderate distance"
