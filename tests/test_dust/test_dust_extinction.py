"""Tests of lb2pix in brutus.dust.extinction."""

import numpy as np

from brutus.dust import lb2pix


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

    def test_lb2pix_scalar_nan_returns_minus_one(self):
        """Regression: scalar NaN latitude must return -1 like the array path.

        The old scalar check `(b < -90.) or (b > 90.)` is False for NaN, so
        NaN fell through to healpy's ang2pix and raised
        `ValueError: THETA is out of range [0,pi]`.
        """
        assert lb2pix(nside=64, l=0.0, b=float("nan")) == -1
        # Array path (already correct) must agree.
        arr = lb2pix(nside=64, l=np.array([0.0]), b=np.array([np.nan]))
        assert arr[0] == -1

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


def test_non_finite_longitude_gives_invalid_pixel():
    assert lb2pix(1024, np.nan, 10.0) == -1
    assert lb2pix(1024, np.inf, 10.0) == -1
    pix = lb2pix(1024, np.array([np.nan, 90.0, np.inf]), np.array([10.0, 40.0, 10.0]))
    assert pix[0] == -1 and pix[2] == -1
    assert pix[1] >= 0
