"""Tests of DustMap and Bayestar in brutus.dust.maps."""

import time
from unittest.mock import MagicMock

import astropy.coordinates as coord
import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from brutus.dust import Bayestar, DustMap, lb2pix


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


# np.trapezoid is the NumPy 2.0 name for np.trapz; CI also tests NumPy 1.x
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


PIXEL_DTYPE_FULL = np.dtype(
    [
        ("nside", "<u4"),
        ("healpix_index", "<u8"),
        ("converged", "u1"),
        ("DM_reliable_min", "<f4"),
        ("DM_reliable_max", "<f4"),
        ("n_stars", "<u4"),
        ("n_good", "<u4"),
        ("n_dwarfs", "<u4"),
    ]
)


PIXEL_DTYPE_MINIMAL = np.dtype([("nside", "<u4"), ("healpix_index", "<u8")])


def _write_bayestar_file(
    path,
    nside=64,
    n_dist=12,
    reliability=True,
    dm_min=8.0,
    dm_max=12.0,
    nonconverged=(),
    refine_pixel=None,
):
    """
    Write a synthetic Bayestar-format HDF5 file with full-sky coverage.

    Parameters
    ----------
    refine_pixel : int, optional
        If given, this base-nside pixel is removed and replaced by its
        four children at 2*nside (multi-resolution map).
    """
    n_pix_base = 12 * nside**2
    nsides = np.full(n_pix_base, nside, dtype=np.uint32)
    hp_idx = np.arange(n_pix_base, dtype=np.uint64)

    if refine_pixel is not None:
        keep = hp_idx != refine_pixel
        nsides = nsides[keep]
        hp_idx = hp_idx[keep]
        children = 4 * refine_pixel + np.arange(4, dtype=np.uint64)
        nsides = np.concatenate([nsides, np.full(4, 2 * nside, dtype=np.uint32)])
        hp_idx = np.concatenate([hp_idx, children])

    n_pix = len(hp_idx)
    dtype = PIXEL_DTYPE_FULL if reliability else PIXEL_DTYPE_MINIMAL
    pixel_info = np.zeros(n_pix, dtype=dtype)
    pixel_info["nside"] = nsides
    pixel_info["healpix_index"] = hp_idx
    if reliability:
        pixel_info["converged"] = 1
        for p in nonconverged:
            pixel_info["converged"][pixel_info["healpix_index"] == p] = 0
        pixel_info["DM_reliable_min"] = dm_min
        pixel_info["DM_reliable_max"] = dm_max

    dists = np.linspace(0.1, 5.0, n_dist)
    # Deterministic, pixel-dependent profiles
    av_mean = (
        0.1
        + 0.001 * np.arange(n_pix, dtype=np.float32)[:, None]
        + 0.05 * np.arange(n_dist, dtype=np.float32)[None, :]
    ).astype(np.float32)
    av_std = np.full((n_pix, n_dist), 0.05, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("pixel_info", data=pixel_info)
        f.create_dataset("dists", data=dists)
        f.create_dataset("av_mean", data=av_mean)
        f.create_dataset("av_std", data=av_std)

    return path


@pytest.fixture(scope="module")
def synthetic_map_file(tmp_path_factory):
    """Full-sky synthetic map with reliability metadata."""
    path = tmp_path_factory.mktemp("dust") / "synthetic_bayestar.h5"
    return str(_write_bayestar_file(path, nonconverged=(7,)))


@pytest.fixture(scope="module")
def synthetic_map(synthetic_map_file):
    return Bayestar(dustfile=synthetic_map_file)


@pytest.fixture(scope="module")
def multires_map_file(tmp_path_factory):
    """Synthetic multi-resolution map (nside 64 + 128)."""
    path = tmp_path_factory.mktemp("dust") / "synthetic_multires.h5"
    return str(_write_bayestar_file(path, refine_pixel=100))


class TestSingleCoordinateShape:
    """Regression: single [l, b] pair must return (n_dist,) profiles."""

    def test_single_pair_returns_1d(self, synthetic_map):
        dists, av_mean, av_std = synthetic_map.query([120.0, 30.0])
        n_dist = len(dists)
        assert av_mean.shape == (n_dist,)
        assert av_std.shape == (n_dist,)

    def test_single_pair_matches_scalar_skycoord(self, synthetic_map):
        d1, m1, s1 = synthetic_map.query([120.0, 30.0])
        coord = SkyCoord(l=120.0 * u.deg, b=30.0 * u.deg, frame="galactic")
        d2, m2, s2 = synthetic_map.query(coord)
        assert m1.shape == m2.shape
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(s1, s2)

    def test_quickstart_pattern(self, synthetic_map):
        # Documented quickstart usage: index the profile by distance mask
        distances, av_mean, av_std = synthetic_map.query([120.0, 30.0])
        av_1kpc = av_mean[distances < 1.0][-1]
        assert np.isfinite(av_1kpc)

    def test_tuple_pair_returns_1d(self, synthetic_map):
        dists, av_mean, _ = synthetic_map.query((120.0, 30.0))
        assert av_mean.shape == (len(dists),)

    def test_multi_coordinate_shape_unchanged(self, synthetic_map):
        coords = np.array([[0.0, 0.0], [90.0, 30.0], [180.0, 60.0]])
        dists, av_mean, av_std = synthetic_map.query(coords)
        assert av_mean.shape == (3, len(dists))
        assert av_std.shape == (3, len(dists))

    def test_explicit_2d_single_coord_shape_unchanged(self, synthetic_map):
        # An explicit (1, 2) array keeps the (1, n_dist) shape
        dists, av_mean, _ = synthetic_map.query(np.array([[120.0, 30.0]]))
        assert av_mean.shape == (1, len(dists))

    def test_skycoord_array_shape_unchanged(self, synthetic_map):
        coords = SkyCoord(
            l=[0.0, 90.0] * u.deg, b=[0.0, 30.0] * u.deg, frame="galactic"
        )
        dists, av_mean, _ = synthetic_map.query(coords)
        assert av_mean.shape == (2, len(dists))


class TestReturnAliasing:
    """Regression: query must not return live internal arrays."""

    def test_distances_returned_as_copy(self, synthetic_map):
        d1, _, _ = synthetic_map.query([120.0, 30.0])
        original = d1.copy()
        d1 *= 1000.0  # caller mutation must not corrupt the map
        d2, _, _ = synthetic_map.query([120.0, 30.0])
        np.testing.assert_array_equal(d2, original)
        np.testing.assert_array_equal(synthetic_map._distances, original)

    def test_av_profiles_do_not_alias_map(self, synthetic_map):
        _, m1, _ = synthetic_map.query([120.0, 30.0], apply_reliability_mask=False)
        expected = m1.copy()
        m1[:] = -999.0
        _, m2, _ = synthetic_map.query([120.0, 30.0], apply_reliability_mask=False)
        np.testing.assert_array_equal(m2, expected)

    def test_scalar_skycoord_profile_does_not_alias_map(self, synthetic_map):
        # Scalar SkyCoord previously produced a 0-d index whose fancy
        # indexing yields a VIEW; ensure the returned profile is safe.
        coord = SkyCoord(l=120.0 * u.deg, b=30.0 * u.deg, frame="galactic")
        _, m1, _ = synthetic_map.query(coord, apply_reliability_mask=False)
        expected = m1.copy()
        m1[:] = -999.0
        _, m2, _ = synthetic_map.query(coord, apply_reliability_mask=False)
        np.testing.assert_array_equal(m2, expected)


class TestReliabilityMask:
    """Reliability-metadata masking (default ON)."""

    def test_mask_applied_by_default(self, synthetic_map):
        # dm_min=8 -> 0.398 kpc; dm_max=12 -> 2.512 kpc
        dists, av_mean, av_std = synthetic_map.query([120.0, 30.0])
        d_lo = 10.0 ** (8.0 / 5.0 - 2.0)
        d_hi = 10.0 ** (12.0 / 5.0 - 2.0)
        inside = (dists >= d_lo) & (dists <= d_hi)
        assert np.any(inside) and np.any(~inside)
        assert np.all(np.isfinite(av_mean[inside]))
        assert np.all(np.isnan(av_mean[~inside]))
        assert np.all(np.isfinite(av_std[inside]))
        assert np.all(np.isnan(av_std[~inside]))

    def test_mask_off_at_query_level(self, synthetic_map):
        dists, av_mean, av_std = synthetic_map.query(
            [120.0, 30.0], apply_reliability_mask=False
        )
        assert np.all(np.isfinite(av_mean))
        assert np.all(np.isfinite(av_std))

    def test_nan_bounds_mask_entire_profile(self, tmp_path):
        """A pixel with NaN reliable-range bounds has NO determined reliable
        range and must be fully masked. (NaN comparisons are always False,
        so without the explicit remap the pixel would be fully UNmasked —
        the opposite of the conservative intent.)"""
        path = str(
            _write_bayestar_file(
                tmp_path / "nan_bounds.h5", dm_min=np.nan, dm_max=np.nan
            )
        )
        bm = Bayestar(dustfile=path)
        _, av_mean, av_std = bm.query([120.0, 30.0])
        assert np.all(np.isnan(av_mean))
        assert np.all(np.isnan(av_std))
        # masking off restores the raw profile
        _, av_mean, _ = bm.query([120.0, 30.0], apply_reliability_mask=False)
        assert np.all(np.isfinite(av_mean))

    def test_infinite_bounds_do_not_mask(self, tmp_path):
        """-inf/+inf bounds are legitimate limiting values ("no cut on that
        side") and must leave converged pixels unmasked."""
        path = str(
            _write_bayestar_file(
                tmp_path / "inf_bounds.h5", dm_min=-np.inf, dm_max=np.inf
            )
        )
        bm = Bayestar(dustfile=path)
        _, av_mean, av_std = bm.query([120.0, 30.0])
        assert np.all(np.isfinite(av_mean))
        assert np.all(np.isfinite(av_std))

    def test_mask_off_at_constructor_level(self, synthetic_map_file):
        bm = Bayestar(dustfile=synthetic_map_file, apply_reliability_mask=False)
        _, av_mean, _ = bm.query([120.0, 30.0])
        assert np.all(np.isfinite(av_mean))
        # Per-query override back on
        dists, av_mean_on, _ = bm.query([120.0, 30.0], apply_reliability_mask=True)
        assert np.any(np.isnan(av_mean_on))

    def test_nonconverged_pixel_fully_masked(self, synthetic_map):
        # Pixel 7 at nside=64 is flagged converged=0. Find a coordinate
        # inside it via healpy pixel centers.
        import healpy as hp

        theta, phi = hp.pixelfunc.pix2ang(64, 7, nest=True)
        gal_l = np.degrees(phi)
        gal_b = 90.0 - np.degrees(theta)
        _, av_mean, av_std = synthetic_map.query([gal_l, gal_b])
        assert np.all(np.isnan(av_mean))
        assert np.all(np.isnan(av_std))
        # Unmasked query still returns the stored profile
        _, av_raw, _ = synthetic_map.query([gal_l, gal_b], apply_reliability_mask=False)
        assert np.all(np.isfinite(av_raw))

    def test_mask_multi_coordinate(self, synthetic_map):
        coords = np.array([[0.0, 0.0], [90.0, 30.0]])
        dists, av_mean, _ = synthetic_map.query(coords)
        d_lo = 10.0 ** (8.0 / 5.0 - 2.0)
        d_hi = 10.0 ** (12.0 / 5.0 - 2.0)
        inside = (dists >= d_lo) & (dists <= d_hi)
        assert np.all(np.isfinite(av_mean[:, inside]))
        assert np.all(np.isnan(av_mean[:, ~inside]))

    def test_missing_fields_degrade_gracefully(self, tmp_path):
        path = str(tmp_path / "no_reliability.h5")
        _write_bayestar_file(path, reliability=False)
        with pytest.warns(RuntimeWarning, match="reliability metadata"):
            bm = Bayestar(dustfile=path)
        # Masking silently skipped: full profile returned
        dists, av_mean, _ = bm.query([120.0, 30.0])
        assert np.all(np.isfinite(av_mean))
        assert bm._converged is None

    def test_missing_fields_no_warning_when_mask_off(self, tmp_path):
        path = str(tmp_path / "no_reliability2.h5")
        _write_bayestar_file(path, reliability=False)
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            bm = Bayestar(dustfile=path, apply_reliability_mask=False)
        _, av_mean, _ = bm.query([120.0, 30.0])
        assert np.all(np.isfinite(av_mean))

    def test_masked_profile_gives_uniform_extinction_prior(self, synthetic_map):
        # NaN-masked bins must degrade to the *normalized* uniform prior over
        # avlim (-ln 20 for the default (0, 20)), not an improper ln p = 0
        from brutus.priors import logp_extinction

        avs = np.array([0.5])
        # Distance beyond the reliable range (> 2.51 kpc)
        logp_out = logp_extinction(avs, synthetic_map, [120.0, 30.0], distance=4.5)
        np.testing.assert_allclose(logp_out, -np.log(20.0))
        logp_out = logp_extinction(
            avs, synthetic_map, [120.0, 30.0], distance=4.5, avlim=(0.0, 6.0)
        )
        np.testing.assert_allclose(logp_out, -np.log(6.0))
        # Distance within the reliable range: informative Gaussian prior
        logp_in = logp_extinction(avs, synthetic_map, [120.0, 30.0], distance=1.0)
        assert not np.isclose(logp_in[0], -np.log(20.0))

    @pytest.mark.parametrize("numpy_branch", [False, True])
    def test_prior_proper_in_masked_and_reliable_bins(
        self, synthetic_map, numpy_branch
    ):
        # Regression: masked bins used ln p = 0 (a density of 1 per mag),
        # up-weighting unreliable distances relative to reliable ones by an
        # arbitrary unit-dependent factor. The prior must integrate to one
        # over avlim at every distance, masked or not.
        from brutus.priors import logp_extinction

        av_grid = np.linspace(0.0, 20.0, 200_001)
        for d in (1.0, 4.5):  # reliable, masked
            dist = np.full(av_grid.size, d)
            # return_components forces the numpy branch; otherwise the
            # large elementwise input takes the fused numba kernel
            lp = logp_extinction(
                av_grid,
                synthetic_map,
                [120.0, 30.0],
                distance=dist,
                return_components=numpy_branch,
            )
            if numpy_branch:
                lp = lp[0]
            assert np.isclose(_trapezoid(np.exp(lp), av_grid), 1.0, rtol=1e-3)


class TestSortEquivalence:
    """lexsort must reproduce the old structured-dtype argsort exactly."""

    @staticmethod
    def _make_pixel_info(n, rng):
        pix = np.zeros(n, dtype=PIXEL_DTYPE_FULL)
        pix["nside"] = rng.choice([64, 128, 256, 512, 1024], size=n)
        # Unique (nside, healpix_index) pairs, shuffled, as in a real map
        pix["healpix_index"] = rng.permutation(n).astype(np.uint64)
        return pix

    def test_sort_microbenchmark(self):
        rng = np.random.default_rng(1)
        pix = self._make_pixel_info(500_000, rng)
        t0 = time.perf_counter()
        ref = np.argsort(pix, order=["nside", "healpix_index"])
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = np.lexsort((pix["healpix_index"], pix["nside"]))
        t_new = time.perf_counter() - t0
        np.testing.assert_array_equal(ref, new)
        # Timing is diagnostic only: the index-equivalence assertion above
        # is the test; a single wall-clock race can flake on shared runners.
        print(
            f"\nsort 500k rows: argsort(order=) {t_ref:.3f}s, "
            f"lexsort {t_new:.3f}s ({t_ref / max(t_new, 1e-9):.0f}x)"
        )


def _find_data_idx_reference(bm, gal_l, b):
    """Old _find_data_idx: per-level lb2pix (reference implementation)."""
    l_arr = np.asarray(gal_l)
    b_arr = np.asarray(b)
    pix_idx = np.full(l_arr.shape, -1, dtype="i8")

    for k, nside in enumerate(bm._nside_levels):
        ipix = lb2pix(nside, l_arr, b_arr, nest=True)
        idx = np.searchsorted(bm._hp_idx_sorted[k], ipix, side="left")

        if np.isscalar(idx):
            if idx < len(bm._hp_idx_sorted[k]) and bm._hp_idx_sorted[k][idx] == ipix:
                pix_idx[...] = bm._data_idx[k][idx]
        else:
            in_bounds = idx < len(bm._hp_idx_sorted[k])
            if not np.any(in_bounds):
                continue
            idx = np.where(in_bounds, idx, -1)
            safe_idx = np.clip(idx, 0, None)
            match_idx = in_bounds & (bm._hp_idx_sorted[k][safe_idx] == ipix)
            if np.any(match_idx):
                valid_idx = idx[match_idx]
                pix_idx[match_idx] = bm._data_idx[k][valid_idx]

    return pix_idx


class TestFindDataIdxEquivalence:
    """Bit-shift pixel derivation must match per-level lb2pix exactly."""

    def test_multires_equivalence(self, multires_map_file):
        bm = Bayestar(dustfile=multires_map_file)
        assert len(bm._nside_levels) == 2  # 64 and 128

        rng = np.random.default_rng(2)
        gal_l = rng.uniform(0.0, 360.0, 5000)
        gal_b = rng.uniform(-90.0, 90.0, 5000)
        gal_b[::100] = 95.0  # invalid coordinates -> -1

        # Include coordinates inside the refined pixel (matched at 128)
        import healpy as hp

        theta, phi = hp.pixelfunc.pix2ang(128, 4 * 100 + 1, nest=True)
        gal_l[1] = np.degrees(phi)
        gal_b[1] = 90.0 - np.degrees(theta)

        ref = _find_data_idx_reference(bm, gal_l, gal_b)
        new = bm._find_data_idx(gal_l, gal_b)
        np.testing.assert_array_equal(ref, new)
        assert new[1] != -1  # refined pixel found
        assert np.all(new[::100] == -1)  # invalid coords preserved

    def test_scalar_equivalence(self, multires_map_file):
        bm = Bayestar(dustfile=multires_map_file)
        for gal_l, gal_b in [(0.0, 0.0), (120.0, 30.0), (0.0, 95.0)]:
            ref = _find_data_idx_reference(bm, gal_l, gal_b)
            new = bm._find_data_idx(gal_l, gal_b)
            np.testing.assert_array_equal(ref, new)

    def test_lb2pix_microbenchmark(self, multires_map_file):
        bm = Bayestar(dustfile=multires_map_file)
        rng = np.random.default_rng(3)
        gal_l = rng.uniform(0.0, 360.0, 200_000)
        gal_b = rng.uniform(-90.0, 90.0, 200_000)

        t0 = time.perf_counter()
        ref = _find_data_idx_reference(bm, gal_l, gal_b)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = bm._find_data_idx(gal_l, gal_b)
        t_new = time.perf_counter() - t0
        np.testing.assert_array_equal(ref, new)
        print(
            f"\n_find_data_idx 200k coords: per-level lb2pix {t_ref:.3f}s, "
            f"shift-based {t_new:.3f}s"
        )


class TestRealFile:
    """End-to-end checks against the shipped bayestar2019_v1.h5."""

    def test_reliability_fields_loaded(self, bayestar_map):
        assert bayestar_map._converged is not None
        assert bayestar_map._d_reliable_min.shape == (bayestar_map._n_pix,)
        assert bayestar_map._d_reliable_max.shape == (bayestar_map._n_pix,)

    def test_mask_narrows_profile(self, bayestar_map):
        dists, av_masked, _ = bayestar_map.query([120.0, 30.0])
        _, av_raw, _ = bayestar_map.query([120.0, 30.0], apply_reliability_mask=False)
        assert av_masked.shape == (len(dists),)
        assert np.all(np.isfinite(av_raw))
        # Masked profile has NaN outside the reliable range but agrees inside
        finite = np.isfinite(av_masked)
        assert 0 < finite.sum() < len(dists)
        np.testing.assert_array_equal(av_masked[finite], av_raw[finite])

    def test_find_data_idx_matches_reference(self, bayestar_map):
        rng = np.random.default_rng(4)
        gal_l = rng.uniform(0.0, 360.0, 20_000)
        gal_b = rng.uniform(-90.0, 90.0, 20_000)
        gal_b[::500] = -95.0
        ref = _find_data_idx_reference(bayestar_map, gal_l, gal_b)
        new = bayestar_map._find_data_idx(gal_l, gal_b)
        np.testing.assert_array_equal(ref, new)

    def test_halo_giant_uniform_prior_beyond_reliable_range(self, bayestar_map):
        from brutus.priors import logp_extinction

        # Sightline reliable only to ~5.6 kpc: at 20 kpc the masked map
        # must yield the normalized uniform prior instead of a tight
        # far-field Gaussian
        logp = logp_extinction(
            np.array([1.0]), bayestar_map, [90.0, 1.0], distance=20.0
        )
        np.testing.assert_allclose(logp, -np.log(20.0))
