#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression and equivalence tests for audit fixes in brutus.dust.maps.

Covers:
1. Reliability-metadata masking (converged, DM_reliable_min/max) in
   Bayestar.query, default ON, with graceful degradation for files
   lacking the fields.
2. query() returning (n_dist,) profiles for a single [l, b] pair
   (previously (1, n_dist), breaking the documented quickstart usage).
3. query() returning a copy of the internal distance grid (previously
   returned by reference, so caller mutation corrupted later queries).
4. lexsort replacing structured-dtype argsort in
   _prepare_index_structures (equivalence + micro-benchmark).
5. Single finest-level lb2pix call + nested bit-shifts replacing
   per-level lb2pix in _find_data_idx (equivalence + micro-benchmark).
"""

import os
import time

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from conftest import find_brutus_data_file

from brutus.dust import Bayestar
from brutus.dust.extinction import lb2pix

BAYESTAR_FILE = find_brutus_data_file("bayestar2019_v1.h5")

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
        # NaN-masked bins must degrade to a uniform prior (logp = 0)
        from brutus.priors import logp_extinction

        avs = np.array([0.5])
        # Distance beyond the reliable range (> 2.51 kpc)
        logp_out = logp_extinction(avs, synthetic_map, [120.0, 30.0], distance=4.5)
        np.testing.assert_array_equal(logp_out, 0.0)
        # Distance within the reliable range: informative Gaussian prior
        logp_in = logp_extinction(avs, synthetic_map, [120.0, 30.0], distance=1.0)
        assert logp_in[0] != 0.0


class TestSortEquivalence:
    """lexsort must reproduce the old structured-dtype argsort exactly."""

    @staticmethod
    def _make_pixel_info(n, rng):
        pix = np.zeros(n, dtype=PIXEL_DTYPE_FULL)
        pix["nside"] = rng.choice([64, 128, 256, 512, 1024], size=n)
        # Unique (nside, healpix_index) pairs, shuffled, as in a real map
        pix["healpix_index"] = rng.permutation(n).astype(np.uint64)
        return pix

    def test_lexsort_matches_argsort_reference(self):
        rng = np.random.default_rng(0)
        pix = self._make_pixel_info(100_000, rng)
        # Reference: the old implementation
        ref = np.argsort(pix, order=["nside", "healpix_index"])
        new = np.lexsort((pix["healpix_index"], pix["nside"]))
        np.testing.assert_array_equal(ref, new)

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


@pytest.mark.skipif(
    BAYESTAR_FILE is None or not os.path.exists(str(BAYESTAR_FILE)),
    reason="Real Bayestar data file not available",
)
class TestRealFile:
    """End-to-end checks against the shipped bayestar2019_v1.h5."""

    @pytest.fixture(scope="class")
    def real_map(self):
        return Bayestar(dustfile=BAYESTAR_FILE)

    def test_reliability_fields_loaded(self, real_map):
        assert real_map._converged is not None
        assert real_map._d_reliable_min.shape == (real_map._n_pix,)
        assert real_map._d_reliable_max.shape == (real_map._n_pix,)

    def test_mask_narrows_profile(self, real_map):
        dists, av_masked, _ = real_map.query([120.0, 30.0])
        _, av_raw, _ = real_map.query([120.0, 30.0], apply_reliability_mask=False)
        assert av_masked.shape == (len(dists),)
        assert np.all(np.isfinite(av_raw))
        # Masked profile has NaN outside the reliable range but agrees inside
        finite = np.isfinite(av_masked)
        assert 0 < finite.sum() < len(dists)
        np.testing.assert_array_equal(av_masked[finite], av_raw[finite])

    def test_find_data_idx_matches_reference(self, real_map):
        rng = np.random.default_rng(4)
        gal_l = rng.uniform(0.0, 360.0, 20_000)
        gal_b = rng.uniform(-90.0, 90.0, 20_000)
        gal_b[::500] = -95.0
        ref = _find_data_idx_reference(real_map, gal_l, gal_b)
        new = real_map._find_data_idx(gal_l, gal_b)
        np.testing.assert_array_equal(ref, new)

    def test_halo_giant_uniform_prior_beyond_reliable_range(self, real_map):
        from brutus.priors import logp_extinction

        # Sightline reliable only to ~5.6 kpc: at 20 kpc the masked map
        # must yield a uniform prior instead of a tight far-field Gaussian
        logp = logp_extinction(np.array([1.0]), real_map, [90.0, 1.0], distance=20.0)
        np.testing.assert_array_equal(logp, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
