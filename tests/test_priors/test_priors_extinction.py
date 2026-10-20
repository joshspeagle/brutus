"""Tests of brutus.priors.extinction."""

from unittest.mock import Mock

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from brutus.priors import logp_extinction


class TestExtinctionPriors:
    """Test dust extinction priors."""

    # --- Tests for simple 2-tuple dust maps ---

    def test_logp_extinction_valid_dustmap(self):
        """Test extinction prior with valid 2-tuple dust map."""
        avs = np.array([0.1, 0.2, 0.3])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map returning (mean, std) 2-tuple
        dustmap = Mock()
        dustmap.query.return_value = (0.2, 0.05)

        logp = logp_extinction(avs, dustmap, coord)

        # Should be Gaussian
        assert np.all(np.isfinite(logp))
        assert logp[1] > logp[0]  # Closer to mean has higher prob
        assert logp[1] > logp[2]

    def test_logp_extinction_no_coverage(self):
        """Test extinction prior without dust map coverage."""
        avs = np.array([0.1, 0.2, 0.3])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map with no coverage
        dustmap = Mock()
        dustmap.query.return_value = (np.nan, np.nan)

        logp = logp_extinction(avs, dustmap, coord)

        # Should be the normalized uniform over the default avlim (0, 20)
        assert np.allclose(logp, -np.log(20.0))

    def test_logp_extinction_with_components(self):
        """Test extinction prior returning components."""
        avs = np.array([0.15])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        # Mock dust map
        dustmap = Mock()
        av_mean, av_err = 0.2, 0.03
        dustmap.query.return_value = (av_mean, av_err)

        logp, (mean, err) = logp_extinction(avs, dustmap, coord, return_components=True)

        assert mean == av_mean
        # Default 0.2 mag systematic scatter is added in quadrature
        assert np.isclose(err, np.sqrt(av_err**2 + 0.2**2))

        logp, (mean, err) = logp_extinction(
            avs, dustmap, coord, return_components=True, scatter=0.0
        )
        assert err == av_err

    def test_logp_extinction_dustmap_error(self):
        """Test extinction prior with dust map query error."""
        avs = np.array([0.1, 0.2])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Mock dust map that raises exception
        dustmap = Mock()
        dustmap.query.side_effect = TypeError("Query failed")

        # Should return uniform prior on error, but warn about it
        with pytest.warns(RuntimeWarning, match="query failed"):
            logp = logp_extinction(avs, dustmap, coord)
        assert np.allclose(logp, -np.log(20.0))

    # --- Tests for 3D dust maps (Bayestar-like 3-tuple) ---

    def test_logp_extinction_3d_dustmap_with_distance(self):
        """Test extinction prior with 3D dust map and distance."""
        avs = np.array([0.1, 0.3, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        # Mock 3D dust map returning (distances, av_profile, std_profile)
        map_distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        av_profile = np.array([0.05, 0.15, 0.3, 0.5, 0.8])
        std_profile = np.array([0.02, 0.05, 0.1, 0.15, 0.2])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        # Evaluate at d=1.0 kpc where map gives av_mean=0.3, av_std=0.1
        logp = logp_extinction(avs, dustmap, coord, distance=1.0)

        assert np.all(np.isfinite(logp))
        # av=0.3 is closest to the map mean at d=1 kpc
        assert logp[1] > logp[0]
        assert logp[1] > logp[2]

    def test_3d_dustmap_without_distance_raises(self):
        dustmap = Mock()
        dustmap.query.return_value = (
            np.array([0.1, 1.0, 5.0]),
            np.array([0.05, 0.3, 0.8]),
            np.array([0.02, 0.1, 0.2]),
        )
        with pytest.raises(ValueError, match="distance"):
            logp_extinction(np.array([0.1, 0.5]), dustmap, (90.0, 0.0))

    @pytest.mark.parametrize("wrap", [list, np.array])
    def test_list_or_array_query_result_matches_tuple(self, wrap):
        profile = (
            np.array([0.1, 1.0, 5.0]),
            np.array([0.05, 0.3, 0.8]),
            np.array([0.02, 0.1, 0.2]),
        )
        avs, dist = np.array([0.0, 1.0, 2.0]), np.full(3, 3.0)
        tuple_map, wrapped_map = Mock(), Mock()
        tuple_map.query.return_value = profile
        wrapped_map.query.return_value = wrap(profile)
        np.testing.assert_array_equal(
            logp_extinction(avs, wrapped_map, None, distance=dist),
            logp_extinction(avs, tuple_map, None, distance=dist),
        )

    @pytest.mark.parametrize("result", [None, 0.3, (1.0,), (1, 2, 3, 4)])
    def test_unreadable_query_result_raises(self, result):
        dustmap = Mock()
        dustmap.query.return_value = result
        with pytest.raises(TypeError, match="dustmap.query"):
            logp_extinction(np.array([0.1]), dustmap, None, distance=1.0)

    def test_logp_extinction_3d_dustmap_array_distances(self):
        """Test 3D dust map with array of distances (BruteForce MC case)."""
        # Simulate the BruteForce case: matching arrays of (av, distance)
        # Use uniform sigma so normalization doesn't affect comparisons
        # avs[0]=0.1 at d=0.5 (map gives 0.15, mismatch)
        # avs[1]=0.3 at d=1.0 (map gives 0.3, exact match)
        # avs[2]=1.5 at d=2.0 (map gives 0.5, large mismatch)
        # avs[3]=0.8 at d=5.0 (map gives 0.8, exact match)
        avs = np.array([0.1, 0.3, 1.5, 0.8])
        distances = np.array([0.5, 1.0, 2.0, 5.0])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        av_profile = np.array([0.05, 0.15, 0.3, 0.5, 0.8])
        std_profile = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # uniform sigma

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        # A wide avlim and scatter=0: a plain Gaussian (the truncated mass
        # is 1 to rounding) with the raw map spread, for the check below
        logp = logp_extinction(
            avs, dustmap, coord, distance=distances, avlim=(-50.0, 50.0), scatter=0.0
        )

        assert logp.shape == avs.shape
        assert np.all(np.isfinite(logp))

        # With uniform sigma, exact matches should have highest logp
        assert logp[1] > logp[0]  # exact match vs small mismatch
        assert logp[3] > logp[2]  # exact match vs large mismatch

        # Verify against manual Gaussian computation
        for i in range(len(avs)):
            expected_mean = np.interp(distances[i], map_distances, av_profile)
            sigma = 0.1
            expected = -0.5 * (
                (avs[i] - expected_mean) ** 2 / sigma**2 + np.log(2 * np.pi * sigma**2)
            )
            assert np.isclose(logp[i], expected, atol=1e-10)

    def test_logp_extinction_3d_dustmap_nan_coverage(self):
        """Test 3D dust map with NaN values (out of coverage)."""
        avs = np.array([0.1, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 1.0, 5.0])
        av_profile = np.array([np.nan, np.nan, np.nan])
        std_profile = np.array([np.nan, np.nan, np.nan])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        logp = logp_extinction(avs, dustmap, coord, distance=1.0)
        assert np.allclose(logp, -np.log(20.0))

    def test_logp_extinction_3d_components(self):
        """Test 3D dust map with return_components."""
        avs = np.array([0.3])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        map_distances = np.array([0.1, 1.0, 5.0])
        av_profile = np.array([0.05, 0.3, 0.8])
        std_profile = np.array([0.02, 0.1, 0.2])

        dustmap = Mock()
        dustmap.query.return_value = (map_distances, av_profile, std_profile)

        logp, (mean, err) = logp_extinction(
            avs, dustmap, coord, distance=1.0, return_components=True
        )

        assert np.isclose(mean, 0.3)
        assert np.isclose(err, np.sqrt(0.1**2 + 0.2**2))

    def test_logp_extinction_string_dustfile(self):
        """Test that passing a string (file path) raises TypeError.

        Regression: an unloaded path (str or pathlib.Path) used to be
        silently treated as 'no coverage', disabling the dust prior for
        the whole fit with no warning.
        """
        avs = np.array([0.1, 0.5])
        coord = SkyCoord(l=90.0, b=0.0, unit="deg", frame="galactic")

        # String has no .query() method — must fail loudly
        with pytest.raises(TypeError, match="query"):
            logp_extinction(avs, "bayestar2019_v1.h5", coord, distance=1.0)


class TestExtinctionAvlimValidation:
    """Degenerate avlim bounds must raise instead of silently corrupting
    the truncation normalization."""

    def _mock_map(self):
        m = Mock()
        m.query.return_value = (1.0, 0.3)  # (mean, std) simple-map form
        return m

    def test_reversed_bounds_raise(self):
        with pytest.raises(ValueError, match="avlim"):
            logp_extinction(
                np.array([0.5]), self._mock_map(), (90.0, 30.0), avlim=(6.0, 0.0)
            )

    def test_nonfinite_bounds_raise(self):
        with pytest.raises(ValueError, match="avlim"):
            logp_extinction(
                np.array([0.5]),
                self._mock_map(),
                (90.0, 30.0),
                avlim=(0.0, np.inf),
            )


class TestExtinctionDustmapValidation:
    """Objects without query() must raise; query failures must warn."""

    def test_pathlib_path_raises(self):
        # Old code: silently returned a uniform prior, disabling the dust
        # prior for the entire fit.
        from pathlib import Path

        with pytest.raises(TypeError, match="query"):
            logp_extinction(
                np.array([0.1, 0.5]),
                Path("map.h5"),
                np.array([10.0, 20.0]),
                distance=np.array([1.0, 1.0]),
            )

    def test_string_raises(self):
        with pytest.raises(TypeError, match="query"):
            logp_extinction(np.array([0.1]), "bayestar2019_v1.h5", None, distance=1.0)

    def test_query_failure_warns_and_returns_uniform(self):
        dustmap = Mock()
        dustmap.query.side_effect = TypeError("Query failed")
        with pytest.warns(RuntimeWarning, match="query failed"):
            logp = logp_extinction(np.array([0.1, 0.2]), dustmap, None)
        assert np.allclose(logp, -np.log(20.0))


class Fake3DMap:
    """Minimal Bayestar-like 3D dust map."""

    def __init__(self, xp, mu, sig):
        self.xp, self.mu, self.sig = xp, mu, sig

    def query(self, coord):
        return (self.xp, self.mu, self.sig)


class TestExtinctionTruncatedNormalization:
    """The Gaussian dust prior must be normalized over the A_V support."""

    def setup_method(self):
        xp = np.linspace(0.05, 5.0, 60)
        mu = 0.005 + 0.075 * (xp / xp[-1])
        sig = 0.01 + 0.02 * (xp / xp[-1])
        self.map = Fake3DMap(xp, mu, sig)

    def test_prior_integrates_to_one_on_support(self):
        # Old code: at d=0.1 kpc the integral over [0, 20] was ~0.69
        # (full-real-line normalization with mu ~ 0.5 sigma above zero).
        av_grid = np.linspace(0.0, 20.0, 200001)
        for d in (0.1, 2.0):
            lp = logp_extinction(
                av_grid, self.map, None, distance=np.full_like(av_grid, d)
            )
            integral = np.trapezoid(np.exp(lp), av_grid)
            assert np.isclose(integral, 1.0, rtol=1e-3)

    def test_avlim_none_raises(self):
        with pytest.raises(ValueError, match="avlim"):
            logp_extinction(np.array([0.1]), self.map, None, distance=1.0, avlim=None)

    def test_wide_avlim_gives_plain_gaussian(self):
        avs = np.array([0.1, 0.2])
        dist = np.array([1.0, 1.0])
        lp = logp_extinction(
            avs, self.map, None, distance=dist, avlim=(-50.0, 50.0), scatter=0.0
        )
        mu = np.interp(1.0, self.map.xp, self.map.mu)
        sig = np.interp(1.0, self.map.xp, self.map.sig)
        expected = -0.5 * ((avs - mu) ** 2 / sig**2 + np.log(2 * np.pi * sig**2))
        assert np.allclose(lp, expected)

    def test_truncation_is_distance_dependent(self):
        # The truncation correction -log(Z(d)) must differ between a
        # near-boundary profile point and a well-separated one.
        avs = np.array([0.0, 0.0])
        dist = np.array([0.05, 5.0])
        lp_trunc = logp_extinction(avs, self.map, None, distance=dist)
        lp_plain = logp_extinction(
            avs, self.map, None, distance=dist, avlim=(-50.0, 50.0)
        )
        corr = lp_trunc - lp_plain
        assert corr[0] > corr[1] > 0.0

    def test_simple_2tuple_map_also_truncated(self):
        dustmap = Mock()
        dustmap.query.return_value = (0.05, 0.1)  # mu within 0.5 sigma of 0
        av_grid = np.linspace(0.0, 20.0, 200001)
        lp = logp_extinction(av_grid, dustmap, None)
        integral = np.trapezoid(np.exp(lp), av_grid)
        assert np.isclose(integral, 1.0, rtol=1e-3)


class TestExtinctionScatter:
    """The 0.2 mag systematic scatter (brutus <= 0.8.3) must be applied."""

    def setup_method(self):
        xp = np.linspace(0.05, 5.0, 60)
        self.mu = 0.005 + 0.075 * (xp / xp[-1])
        self.sig = 0.01 + 0.02 * (xp / xp[-1])
        self.map = Fake3DMap(xp, self.mu, self.sig)

    def test_default_scatter_added_in_quadrature(self):
        # Regression: the refactor dropped the scatter, leaving the raw
        # ~0.01-0.03 mag map spreads as the prior width
        _, (mean, err) = logp_extinction(
            np.array([0.1]), self.map, None, distance=1.0, return_components=True
        )
        raw = np.interp(1.0, self.map.xp, self.sig)
        assert np.isclose(err, np.sqrt(raw**2 + 0.2**2))

    @pytest.mark.parametrize("scatter", [0.0, 0.05, 0.5])
    def test_scatter_matches_plain_gaussian(self, scatter):
        avs = np.array([0.0, 0.1, 0.4])
        lp = logp_extinction(
            avs,
            self.map,
            None,
            distance=np.ones(3),
            avlim=(-50.0, 50.0),
            scatter=scatter,
        )
        mu = np.interp(1.0, self.map.xp, self.mu)
        sig = np.hypot(np.interp(1.0, self.map.xp, self.sig), scatter)
        expected = -0.5 * ((avs - mu) ** 2 / sig**2 + np.log(2 * np.pi * sig**2))
        assert np.allclose(lp, expected)

    def test_zero_spread_bins_become_informative(self):
        # Bayestar has bins with av_std == 0 exactly; with scatter they get
        # a proper N(mu, 0.2) prior, without it they stay uninformative
        sig = self.sig.copy()
        sig[:] = 0.0
        m = Fake3DMap(self.map.xp, self.mu, sig)
        avs = np.array([0.05, 1.0])
        lp = logp_extinction(avs, m, None, distance=np.ones(2))
        assert lp[0] > lp[1]
        lp0 = logp_extinction(avs, m, None, distance=np.ones(2), scatter=0.0)
        np.testing.assert_allclose(lp0, -np.log(20.0))

    def test_negative_spread_stays_invalid(self):
        sig = np.full_like(self.sig, -0.5)
        m = Fake3DMap(self.map.xp, self.mu, sig)
        lp = logp_extinction(np.array([0.1]), m, None, distance=1.0)
        np.testing.assert_allclose(lp, -np.log(20.0))

    @pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
    def test_invalid_scatter_raises(self, bad):
        with pytest.raises(ValueError, match="scatter"):
            logp_extinction(np.array([0.1]), self.map, None, distance=1.0, scatter=bad)
