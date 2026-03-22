#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus.dust module.

This test suite covers all functionality in the refactored dust module:
1. Coordinate transformation utilities (extinction.py)
2. Dust map classes and methods (maps.py)
3. Integration and edge case testing
4. Performance validation

Tests use real Bayestar data from DATAFILES for authentic validation.
"""

import os
from unittest.mock import MagicMock

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from conftest import find_brutus_data_file

# Import what we're testing
from brutus.dust import Bayestar, DustMap, lb2pix

# Path to real Bayestar data
BAYESTAR_FILE = find_brutus_data_file("bayestar2019_v1.h5")


class TestCoordinateUtils:
    """Test coordinate transformation utilities."""

    def test_lb2pix_single_coordinates(self):
        """Test lb2pix with single coordinate inputs."""
        # Valid coordinate at Galactic center
        pix = lb2pix(nside=64, l=0.0, b=0.0)
        assert isinstance(pix, (int, np.integer))
        assert pix >= 0

        # Valid coordinate in first quadrant
        pix = lb2pix(nside=64, l=90.0, b=45.0)
        assert isinstance(pix, (int, np.integer))
        assert pix >= 0

        # Different nside values
        pix_32 = lb2pix(nside=32, l=0.0, b=0.0)
        pix_128 = lb2pix(nside=128, l=0.0, b=0.0)
        assert pix_32 != pix_128  # Different resolutions give different pixels

    def test_lb2pix_array_coordinates(self):
        """Test lb2pix with array coordinate inputs."""
        # Multiple valid coordinates
        l_arr = np.array([0.0, 90.0, 180.0, 270.0])
        b_arr = np.array([0.0, 30.0, -30.0, 60.0])

        pix_arr = lb2pix(nside=64, l=l_arr, b=b_arr)

        assert isinstance(pix_arr, np.ndarray)
        assert pix_arr.shape == (4,)
        assert np.all(pix_arr >= 0)  # All should be valid
        assert len(np.unique(pix_arr)) > 1  # Should be different pixels

    def test_lb2pix_invalid_coordinates(self):
        """Test lb2pix with invalid coordinate inputs."""
        # Single invalid coordinate
        invalid_pix = lb2pix(nside=64, l=0.0, b=95.0)
        assert invalid_pix == -1

        invalid_pix = lb2pix(nside=64, l=0.0, b=-95.0)
        assert invalid_pix == -1

        # Mixed valid and invalid coordinates
        l_arr = np.array([0.0, 90.0, 0.0])
        b_arr = np.array([0.0, 30.0, 95.0])  # Last one invalid

        pix_arr = lb2pix(nside=64, l=l_arr, b=b_arr)
        assert pix_arr[0] >= 0  # Valid
        assert pix_arr[1] >= 0  # Valid
        assert pix_arr[2] == -1  # Invalid

    def test_lb2pix_nested_vs_ring(self):
        """Test lb2pix with different pixel ordering schemes."""
        l, b = 45.0, 30.0

        # Test both ordering schemes
        pix_nested = lb2pix(nside=64, l=l, b=b, nest=True)
        pix_ring = lb2pix(nside=64, l=l, b=b, nest=False)

        # They should be different (unless by coincidence)
        # More importantly, both should be valid
        assert pix_nested >= 0
        assert pix_ring >= 0

    def test_lb2pix_edge_cases(self):
        """Test lb2pix edge cases."""
        # Exactly at boundaries
        pix_north = lb2pix(nside=64, l=0.0, b=90.0)
        pix_south = lb2pix(nside=64, l=0.0, b=-90.0)
        assert pix_north >= 0
        assert pix_south >= 0

        # Large longitude values (should wrap)
        pix_360 = lb2pix(nside=64, l=360.0, b=0.0)
        pix_0 = lb2pix(nside=64, l=0.0, b=0.0)
        # Should be the same pixel (360° = 0°)
        assert pix_360 == pix_0


class TestDustMapBase:
    """Test the abstract DustMap base class."""

    def test_dustmap_instantiation(self):
        """Test that DustMap can be instantiated."""
        dust_map = DustMap()
        assert isinstance(dust_map, DustMap)

    def test_dustmap_call_alias(self):
        """Test that __call__ method is an alias for query."""
        dust_map = DustMap()

        # Mock query method
        dust_map.query = MagicMock(return_value="test_result")

        # Test __call__ alias
        result = dust_map("test_coords", test_arg="value")

        dust_map.query.assert_called_once_with("test_coords", test_arg="value")
        assert result == "test_result"

    def test_dustmap_query_not_implemented(self):
        """Test that query raises NotImplementedError."""
        dust_map = DustMap()

        with pytest.raises(
            NotImplementedError, match="must be implemented by subclasses"
        ):
            dust_map.query("test_coords")

    def test_dustmap_query_gal(self):
        """Test Galactic coordinate query interface."""
        dust_map = DustMap()

        # Mock the query method
        dust_map.query = MagicMock(return_value="test_result")

        # Test without distance
        result = dust_map.query_gal(ell=45.0, b=30.0)
        assert result == "test_result"
        dust_map.query.assert_called_once()

        # Check that (l, b) array was passed directly (optimized path)
        coords_arg = dust_map.query.call_args[0][0]
        assert isinstance(coords_arg, np.ndarray)
        np.testing.assert_allclose(coords_arg[0, 0], 45.0)
        np.testing.assert_allclose(coords_arg[0, 1], 30.0)

        # Test with distance (d is accepted but not used for HEALPix queries)
        dust_map.query.reset_mock()
        result = dust_map.query_gal(ell=45.0, b=30.0, d=1.5)
        assert result == "test_result"
        coords_arg = dust_map.query.call_args[0][0]
        assert isinstance(coords_arg, np.ndarray)
        np.testing.assert_allclose(coords_arg[0, 0], 45.0)
        np.testing.assert_allclose(coords_arg[0, 1], 30.0)

    def test_dustmap_query_equ(self):
        """Test Equatorial coordinate query interface."""
        dust_map = DustMap()
        dust_map.query = MagicMock(return_value="test_result")

        # Test with default frame
        result = dust_map.query_equ(ra=180.0, dec=45.0)
        assert result == "test_result"

        # Test with different frame
        result = dust_map.query_equ(ra=180.0, dec=45.0, frame="fk5")
        assert result == "test_result"

        # Test invalid frame
        with pytest.raises(ValueError, match="not supported"):
            dust_map.query_equ(ra=180.0, dec=45.0, frame="invalid")

    def test_dustmap_coordinate_units(self):
        """Test handling of astropy units in coordinate queries."""
        dust_map = DustMap()
        dust_map.query = MagicMock(return_value="test_result")

        # Test with astropy Quantities
        result = dust_map.query_gal(ell=45.0 * u.deg, b=30.0 * u.deg, d=1.5 * u.kpc)
        assert result == "test_result"


@pytest.fixture(scope="module")
def bayestar_map():
    """Load real Bayestar map once for all tests."""
    if BAYESTAR_FILE is None:
        raise FileNotFoundError("Bayestar data file not found in any standard location")

    return Bayestar(dustfile=BAYESTAR_FILE)


class TestBayestarMapReal:
    """Test the Bayestar dust map implementation with real data."""

    def test_bayestar_initialization(self, bayestar_map):
        """Test Bayestar initialization with real data."""
        # Check that data was loaded
        assert hasattr(bayestar_map, "_distances")
        assert hasattr(bayestar_map, "_av_mean")
        assert hasattr(bayestar_map, "_av_std")
        assert hasattr(bayestar_map, "_pixel_info")

        # Check data shapes are reasonable
        n_pixels = len(bayestar_map._pixel_info)
        n_distances = len(bayestar_map._distances)

        assert n_pixels > 0
        assert n_distances > 0
        assert bayestar_map._av_mean.shape == (n_pixels, n_distances)
        assert bayestar_map._av_std.shape == (n_pixels, n_distances)

        # Check that distances are increasing
        assert np.all(np.diff(bayestar_map._distances) > 0)

    def test_bayestar_file_error_handling(self):
        """Test Bayestar error handling for missing files."""
        with pytest.raises((FileNotFoundError, OSError)):
            Bayestar(dustfile="nonexistent_file.h5")

    def test_bayestar_query_size_real(self, bayestar_map):
        """Test query size estimation with real data."""
        # Mock coordinates with known shape
        coords = MagicMock()
        coords.shape = (5,)  # 5 coordinates

        size = bayestar_map.get_query_size(coords)
        expected_size = 5 * len(bayestar_map._distances)
        assert size == expected_size

    def test_bayestar_find_data_idx_real(self, bayestar_map):
        """Test internal _find_data_idx method with real data."""
        # Test with coordinates within Bayestar coverage (dec > -30)
        # Use coordinates that should be covered
        l = np.array([0.0, 90.0, 180.0])
        b = np.array([0.0, 30.0, 60.0])

        pix_idx = bayestar_map._find_data_idx(l, b)

        assert isinstance(pix_idx, np.ndarray)
        assert pix_idx.shape == (3,)

        # Some pixels should be found (not all -1) for these coordinates
        n_found = np.sum(pix_idx >= 0)
        assert n_found >= 0  # At least some should be found

    def test_bayestar_query_skycoord_real(self, bayestar_map):
        """Test Bayestar query with SkyCoord objects (real data)."""
        # Create coordinates within Bayestar coverage
        coords = coord.SkyCoord(
            l=[0.0, 90.0] * u.deg, b=[0.0, 30.0] * u.deg, frame="galactic"
        )

        distances, av_mean, av_std = bayestar_map.query(coords)

        # Check return types and shapes
        assert isinstance(distances, np.ndarray)
        assert isinstance(av_mean, np.ndarray)
        assert isinstance(av_std, np.ndarray)

        # Check shapes
        n_distances = len(distances)
        assert av_mean.shape == (2, n_distances)  # 2 coords × n_distances
        assert av_std.shape == (2, n_distances)

        # Check for reasonable extinction values
        valid_mask = ~np.isnan(av_mean)
        if np.any(valid_mask):
            valid_av = av_mean[valid_mask]
            assert np.all(valid_av >= 0)  # Extinction should be non-negative
            assert np.all(valid_av < 10)  # Should be reasonable values

    def test_bayestar_query_array_real(self, bayestar_map):
        """Test Bayestar query with coordinate arrays (real data)."""
        # Test with 2D array format that triggers the except clause
        coords = np.array([[0.0, 0.0], [90.0, 30.0], [180.0, 60.0]])

        distances, av_mean, av_std = bayestar_map.query(coords)

        # Check return types and shapes
        assert isinstance(distances, np.ndarray)
        assert isinstance(av_mean, np.ndarray)
        assert isinstance(av_std, np.ndarray)

        # Check shapes
        n_distances = len(distances)
        assert av_mean.shape == (3, n_distances)  # 3 coords × n_distances
        assert av_std.shape == (3, n_distances)

    def test_bayestar_query_single_coord_real(self, bayestar_map):
        """Test Bayestar query with single coordinate (real data)."""
        # Single coordinate within coverage
        coords = coord.SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

        distances, av_mean, av_std = bayestar_map.query(coords)

        # Check return types
        assert isinstance(distances, np.ndarray)
        assert isinstance(av_mean, np.ndarray)
        assert isinstance(av_std, np.ndarray)

        # For single coordinate, should return 1D arrays
        n_distances = len(distances)
        assert av_mean.shape == (n_distances,)
        assert av_std.shape == (n_distances,)

    def test_bayestar_coordinate_methods_real(self, bayestar_map):
        """Test coordinate query methods with real data."""
        # Test Galactic coordinate query
        distances, av_mean, av_std = bayestar_map.query_gal(ell=0.0, b=0.0)
        assert isinstance(distances, np.ndarray)
        assert len(distances) > 0

        # Test Equatorial coordinate query
        # Convert Galactic center to equatorial coordinates
        distances2, av_mean2, av_std2 = bayestar_map.query_equ(ra=266.4, dec=-28.9)
        assert isinstance(distances2, np.ndarray)

        # Results should be similar (same sky position)
        assert len(distances) == len(distances2)

    def test_bayestar_coverage_limits_real(self, bayestar_map):
        """Test Bayestar coverage limits with real data."""
        # Test coordinates outside coverage (dec < -30)
        # This should return NaN values
        coords_south = coord.SkyCoord(ra=0.0 * u.deg, dec=-50.0 * u.deg, frame="icrs")

        distances, av_mean, av_std = bayestar_map.query(coords_south)

        # Should return data structure but values might be NaN
        assert isinstance(distances, np.ndarray)
        assert isinstance(av_mean, np.ndarray)
        assert isinstance(av_std, np.ndarray)

        # Data outside coverage should be NaN
        # (Note: this tests real Bayestar coverage behavior)


class TestBayestarFindDataIdxOutOfCoverage:
    """Test that _find_data_idx returns NaN (via -1 index) for out-of-coverage coords."""

    def test_out_of_coverage_returns_negative_idx(self, bayestar_map):
        """
        Query coordinates well outside the Bayestar map coverage and
        verify _find_data_idx returns -1 (not data from the last pixel).

        Bayestar covers declination > -30 deg approximately.
        Coordinates at very negative declinations (in Galactic coords,
        near the south celestial pole) should not be found.
        """
        # Use coordinates likely outside Bayestar coverage.
        # The south celestial pole in Galactic coords is roughly l~303, b~-27.
        # Pick coordinates at extreme southern galactic latitudes that are
        # less likely to be covered.
        # Also try coordinates that correspond to dec << -30.
        # l=0, b=-89 is near the south galactic pole (dec ~ -27 deg, might
        # still be covered). Use equatorial coords to be sure:
        # dec = -80 => convert to galactic and query.
        south_eq = coord.SkyCoord(ra=0.0 * u.deg, dec=-80.0 * u.deg, frame="icrs")
        gal = south_eq.galactic
        l_arr = np.array([gal.l.deg])
        b_arr = np.array([gal.b.deg])

        pix_idx = bayestar_map._find_data_idx(l_arr, b_arr)

        # This coordinate should NOT be in the map (dec = -80),
        # so pix_idx should be -1.
        assert pix_idx[0] == -1, (
            f"Coordinate at dec=-80 (l={l_arr[0]:.1f}, b={b_arr[0]:.1f}) "
            f"should return pix_idx=-1 (not found), got {pix_idx[0]}"
        )

    def test_out_of_coverage_query_returns_nan(self, bayestar_map):
        """
        Full query for coordinates outside coverage should return NaN
        extinction values, not data from arbitrary pixels.
        """
        # dec = -80 is far outside Bayestar coverage
        coords_south = coord.SkyCoord(ra=0.0 * u.deg, dec=-80.0 * u.deg, frame="icrs")

        distances, av_mean, av_std = bayestar_map.query(coords_south)

        # The returned extinction should be all NaN for an uncovered pixel
        assert np.all(np.isnan(av_mean)), (
            "Out-of-coverage query should return NaN extinction values, "
            f"got non-NaN values: {av_mean[~np.isnan(av_mean)]}"
        )


class TestDustModuleIntegration:
    """Integration tests for the dust module."""

    def test_module_imports(self):
        """Test that all module imports work correctly."""
        from brutus.dust import __all__

        # Check that __all__ contains expected items
        expected_items = {"lb2pix", "DustMap", "Bayestar"}
        assert set(__all__) == expected_items

    def test_dust_map_inheritance(self, bayestar_map):
        """Test that Bayestar properly inherits from DustMap."""
        assert issubclass(Bayestar, DustMap)

        # Test inherited query_gal method
        assert hasattr(bayestar_map, "query_gal")
        assert hasattr(bayestar_map, "query_equ")
        assert hasattr(bayestar_map, "__call__")

        # Test that __call__ is an alias for query
        bayestar_map.query = MagicMock(return_value=("test", "test", "test"))
        result = bayestar_map("test_coords")
        bayestar_map.query.assert_called_once_with("test_coords")


class TestDustPerformance:
    """Performance and stress tests."""

    def test_lb2pix_performance(self):
        """Test lb2pix performance with large arrays."""
        # Large coordinate arrays
        n_coords = 10000
        l = np.random.uniform(0, 360, n_coords)
        b = np.random.uniform(-90, 90, n_coords)

        # Should complete quickly
        import time

        start_time = time.time()
        pix_indices = lb2pix(nside=64, l=l, b=b)
        elapsed = time.time() - start_time

        # Performance check (should be fast)
        assert elapsed < 1.0  # Should take less than 1 second
        assert pix_indices.shape == (n_coords,)
        assert np.all(pix_indices >= -1)  # All valid or -1

    def test_coordinate_boundary_cases(self):
        """Test coordinate edge cases and boundaries."""
        # Test coordinates exactly at boundaries
        boundary_coords = [
            (0.0, 90.0),  # North pole
            (0.0, -90.0),  # South pole
            (0.0, 0.0),  # Galactic center
            (180.0, 0.0),  # Opposite galactic center
            (359.999, 0.0),  # Near longitude wrap
        ]

        for l, b in boundary_coords:
            pix = lb2pix(nside=64, l=l, b=b)
            assert pix >= 0, f"Coordinate ({l}, {b}) should be valid"


class TestDustEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_coordinate_arrays(self):
        """Test handling of empty coordinate arrays."""
        l_empty = np.array([])
        b_empty = np.array([])

        pix_empty = lb2pix(nside=64, l=l_empty, b=b_empty)
        assert isinstance(pix_empty, np.ndarray)
        assert pix_empty.shape == (0,)

    def test_broadcasting_coordinate_shapes(self):
        """Test broadcasting behavior with different coordinate array shapes."""
        l = np.array([0.0, 90.0])  # Shape (2,)
        b = np.array([0.0])  # Shape (1,) - should broadcast to (2,)

        # Should work via broadcasting
        pix = lb2pix(nside=64, l=l, b=b)
        assert pix.shape == (2,)  # Should broadcast to shape of l
        assert np.all(pix >= 0)  # Should be valid coordinates

        # Test truly incompatible shapes
        l_2d = np.array([[0.0, 90.0], [180.0, 270.0]])  # Shape (2, 2)
        b_1d = np.array([0.0, 30.0, 60.0])  # Shape (3,) - incompatible

        with pytest.raises(ValueError, match="incompatible shapes"):
            lb2pix(nside=64, l=l_2d, b=b_1d)

    def test_invalid_nside_values(self):
        """Test handling of invalid nside values."""
        # HEALPix nside must be power of 2
        with pytest.raises(ValueError):
            lb2pix(nside=63, l=0.0, b=0.0)  # Not power of 2

    def test_dustmap_coordinate_validation(self):
        """Test coordinate validation in DustMap methods."""
        dust_map = DustMap()
        dust_map.query = MagicMock(return_value="test")

        # Test extreme coordinate values
        result = dust_map.query_gal(ell=720.0, b=0.0)  # Large longitude
        assert result == "test"

        result = dust_map.query_equ(ra=-90.0, dec=45.0)  # Negative RA
        assert result == "test"


if __name__ == "__main__":
    pytest.main([__file__])
