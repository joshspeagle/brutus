#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression and equivalence tests for the EEPTracks / StarEvolTrack audit fixes.

Covers:

1. Pickle-cache format versioning: the cache filename embeds a format
   version, stale caches written under the old (unversioned) name are
   ignored, and cached loads reproduce fresh-build predictions exactly.
2. Construction-only intermediates (libparams/output/X) are freed after
   the interpolator is built and are not written to the cache.
3. `_make_lib` single-pass HDF5 read is equivalent to the previous
   per-column re-read implementation.
4. Age weights are computed as d(age)/d(EEP) against the EEP coordinate,
   not against the array index (regression: non-unit EEP spacing).
5. Secondary EEP age-matching uses the primary's [a/Fe] (clamped to the
   library range) instead of hard-coded 0.0.
6. The vectorized 1-D age-curve inversion for the secondary EEP agrees
   with the previous Nelder-Mead solve on the real MIST tracks and is
   faster.
"""

import pickle
import time
import warnings

import h5py
import numpy as np
import pytest

from brutus.core.individual import (
    _EEPTRACKS_CACHE_ATTRS,
    _EEPTRACKS_CACHE_VERSION,
    EEPTracks,
    StarEvolTrack,
    rename,
)

# Predictions used for the synthetic-file tests (loga/logl/logt/logg are
# required by EEPTracks' internal indexing)
PREDICTIONS = ["loga", "logl", "logt", "logg"]


def _make_synthetic_mist_file(path, eep_step=1.0, n_eep=40):
    """
    Write a tiny MIST-shaped HDF5 track library.

    Structure matches what EEPTracks._make_lib expects: an 'index' dataset
    of track names, and one compound dataset per track with the renamed
    label and prediction columns. Ages are linear in EEP
    (age = AGE_SLOPE * eep years) so the exact d(age)/d(EEP) Jacobian is
    known analytically.
    """
    minis = np.array([0.5, 1.0])
    fehs = np.array([0.0])
    afes = np.array([0.0])
    eeps = 200.0 + eep_step * np.arange(n_eep)

    dtype = np.dtype(
        [
            (rename["mini"], "f8"),
            (rename["eep"], "f8"),
            (rename["feh"], "f8"),
            (rename["afe"], "f8"),
            (rename["loga"], "f8"),
            (rename["logl"], "f8"),
            (rename["logt"], "f8"),
            (rename["logg"], "f8"),
        ]
    )

    with h5py.File(path, "w") as f:
        names = []
        for m in minis:
            for z in fehs:
                for a in afes:
                    name = f"trk_m{m}_z{z}_a{a}"
                    arr = np.zeros(len(eeps), dtype=dtype)
                    arr[rename["mini"]] = m
                    arr[rename["eep"]] = eeps
                    arr[rename["feh"]] = z
                    arr[rename["afe"]] = a
                    # age linear in EEP: age = AGE_SLOPE*eep -> known Jacobian
                    arr[rename["loga"]] = np.log10(AGE_SLOPE * eeps)
                    arr[rename["logl"]] = 0.01 * (eeps - 200.0) + m
                    arr[rename["logt"]] = 3.7 - 0.001 * (eeps - 200.0)
                    arr[rename["logg"]] = 4.5 - 0.002 * (eeps - 200.0)
                    f.create_dataset(name, data=arr)
                    names.append(name)
        f.create_dataset("index", data=np.array(names, dtype="S"))
    return path


AGE_SLOPE = 1.0e7  # yr per EEP in the synthetic library


class TestCacheVersioning:
    """Cache key must be versioned; stale caches must be ignored."""

    def test_cache_filename_contains_version(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=True,
        )
        caches = list(tmp_path.glob("*.pkl"))
        assert len(caches) == 1
        assert f"_cachev{_EEPTRACKS_CACHE_VERSION}" in caches[0].name

    def test_stale_unversioned_cache_ignored(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        # Plant a poisoned cache at the OLD (unversioned) cache path; a
        # version-unaware loader would restore this garbage as attributes.
        old_name = f"tracks_ageweightTrue_pred{''.join(PREDICTIONS)}.pkl"
        with open(tmp_path / old_name, "wb") as f:
            pickle.dump({"interpolator": "garbage", "gridpoints": None}, f)

        tracks = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=True,
        )
        # Must have rebuilt from the HDF5 file, not loaded the stale pickle
        assert not isinstance(tracks.interpolator, str)
        preds = tracks.get_predictions([0.75, 210.0, 0.0, 0.0])
        assert np.all(np.isfinite(preds))

    def test_cached_load_matches_fresh_build(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        fresh = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=True,  # writes the cache
        )
        cached = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=True,  # loads the cache written above
        )
        queries = np.array(
            [
                [0.5, 205.0, 0.0, 0.0],
                [0.75, 220.0, 0.0, 0.0],
                [1.0, 230.5, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(
            fresh.get_predictions(queries), cached.get_predictions(queries)
        )
        assert cached.predictions == fresh.predictions

    def test_intermediates_freed_and_not_cached(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        tracks = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=True,
        )
        for attr in ("libparams", "output", "X"):
            assert not hasattr(tracks, attr)
        # The cache must contain only the runtime whitelist
        cache_file = next(iter(tmp_path.glob("*.pkl")))
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        assert set(cached_data) <= set(_EEPTRACKS_CACHE_ATTRS)
        for attr in ("libparams", "output", "X"):
            assert attr not in cached_data


class TestMakeLibSinglePass:
    """Single-pass HDF5 read must be equivalent to the old per-column read."""

    @staticmethod
    def _reference_make_lib(mistfile, labels, predictions):
        """Previous implementation: full read for labels + per-column reads."""
        with h5py.File(mistfile, "r") as misth5:
            cols = [rename[p] for p in labels]
            libparams = np.concatenate(
                [np.array(misth5[z])[cols] for z in misth5["index"]]
            )
            libparams.dtype.names = tuple(labels)

            cols_to_read = [rename[p] for p in predictions]
            output_data = [
                np.concatenate([misth5[z][p] for z in misth5["index"]])
                for p in cols_to_read
            ]
            output = np.empty((len(output_data[0]), len(predictions)), dtype="f8")
            for i, col in enumerate(output_data):
                output[:, i] = col
        return libparams, output

    def test_equivalence_and_benchmark(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5", n_eep=200)

        # New implementation, captured before __init__ frees the arrays
        tracks = EEPTracks.__new__(EEPTracks)
        tracks.labels = ["mini", "eep", "feh", "afe"]
        tracks.predictions = list(PREDICTIONS)
        t0 = time.perf_counter()
        with h5py.File(mistfile, "r") as misth5:
            tracks._make_lib(misth5, verbose=False)
        t_new = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref_libparams, ref_output = self._reference_make_lib(
            mistfile, tracks.labels, tracks.predictions
        )
        t_ref = time.perf_counter() - t0

        assert tracks.libparams.dtype.names == ref_libparams.dtype.names
        for name in ref_libparams.dtype.names:
            np.testing.assert_array_equal(tracks.libparams[name], ref_libparams[name])
        np.testing.assert_array_equal(tracks.output, ref_output)

        print(
            f"\n_make_lib: single-pass {1e3 * t_new:.1f} ms vs per-column "
            f"reference {1e3 * t_ref:.1f} ms ({t_ref / t_new:.1f}x)"
        )


class TestAgeWeightJacobian:
    """agewt must be d(age)/d(EEP), not Delta(age) per array index."""

    def test_unit_spacing_matches_analytic_jacobian(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "unit.h5", eep_step=1.0)
        tracks = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=False,
        )
        agewt = tracks.get_predictions([0.75, 215.0, 0.0, 0.0], apply_corr=False)[
            tracks.predictions.index("agewt")
        ]
        assert agewt == pytest.approx(AGE_SLOPE, rel=1e-8)

    def test_non_unit_spacing_regression(self, tmp_path):
        # Regression: with EEP steps of 6, the old np.gradient(linear_ages)
        # (index spacing) returned 6x the true Jacobian d(age)/d(EEP).
        step = 6.0
        mistfile = _make_synthetic_mist_file(tmp_path / "coarse.h5", eep_step=step)
        tracks = EEPTracks(
            mistfile=mistfile,
            predictions=PREDICTIONS,
            verbose=False,
            use_cache=False,
        )
        agewt = tracks.get_predictions([0.75, 215.0, 0.0, 0.0], apply_corr=False)[
            tracks.predictions.index("agewt")
        ]
        # Old code returned ~step * AGE_SLOPE here
        assert agewt == pytest.approx(AGE_SLOPE, rel=1e-8)
        assert not np.isclose(agewt, step * AGE_SLOPE, rtol=0.5)


class _AfeDependentTracks:
    """Duck-typed tracks whose age depends on [a/Fe] (multi-afe library)."""

    labels = ["mini", "eep", "feh", "afe"]
    predictions = ["loga"]
    gridpoints = {
        "mini": np.linspace(0.1, 2.0, 20),
        "eep": np.linspace(200.0, 500.0, 301),
        "feh": np.array([0.0]),
        "afe": np.array([-0.2, 0.0, 0.4]),
    }

    @staticmethod
    def _loga(eep, afe):
        # Age increases with EEP; alpha enhancement shifts ages: matching an
        # age on afe=0 tracks lands at the wrong EEP for afe != 0
        return 0.01 * np.asarray(eep, dtype=float) + 0.1 * np.asarray(afe, dtype=float)

    def get_predictions(self, labels, apply_corr=True, corr_params=None):
        labels = np.asarray(labels, dtype=float)
        if labels.ndim == 1:
            return np.array([self._loga(labels[1], labels[3])])
        return self._loga(labels[:, 1], labels[:, 3])[:, None]


class TestSecondaryAfeConsistency:
    """Secondary age-matching must use the primary's [a/Fe], not 0.0."""

    def _star_track(self):
        st = StarEvolTrack.__new__(StarEvolTrack)
        st.tracks = _AfeDependentTracks()
        return st

    def test_secondary_matched_at_primary_afe(self):
        st = self._star_track()
        tracks = st.tracks
        afe = 0.4
        mini, smf, feh = 1.0, 0.8, 0.0
        eep_p = 350.0
        loga_target = float(tracks._loga(eep_p, afe))

        eep2 = st._get_eep_for_secondary(loga_target, mini, eep_p, feh, afe, smf, 1e-2)
        assert np.isfinite(eep2)

        # Age of the secondary evaluated at the SAME afe used downstream in
        # get_seds must match the primary's age. The old code solved on
        # afe=0 tracks, leaving a 0.1*afe = 0.04 dex mismatch (> tol).
        loga_2 = float(
            tracks.get_predictions([mini * smf, eep2, feh, afe], apply_corr=False)[0]
        )
        assert abs(loga_2 - loga_target) < 1e-3

    def test_afe_clamped_to_library_range(self):
        st = self._star_track()
        tracks = st.tracks
        # Request afe beyond the library maximum (0.4): must clamp, not
        # extrapolate
        afe_req, afe_max = 1.0, 0.4
        eep_p = 350.0
        loga_target = float(tracks._loga(eep_p, afe_max))
        eep2 = st._get_eep_for_secondary(
            loga_target, 1.0, eep_p, 0.0, afe_req, 0.8, 1e-2
        )
        assert np.isfinite(eep2)
        loga_2 = float(
            tracks.get_predictions([0.8, eep2, 0.0, afe_max], apply_corr=False)[0]
        )
        assert abs(loga_2 - loga_target) < 1e-3


def _find_data_file(name):
    from pathlib import Path

    for base in ("data/DATAFILES", "./data/DATAFILES"):
        p = Path(base) / name
        if p.exists():
            return p
    try:
        import pooch

        p = Path(pooch.os_cache("astro-brutus")) / name
        if p.exists():
            return p
    except ImportError:
        pass
    return None


_track_file = _find_data_file("MIST_1.2_EEPtrk.h5")


@pytest.mark.requires_data
@pytest.mark.skipif(_track_file is None, reason="MIST track data not available")
class TestSecondaryEEPInversionRealTracks:
    """1-D inversion must match the old Nelder-Mead solve on real MIST."""

    @pytest.fixture(scope="class")
    def star_track(self):
        tracks = EEPTracks(mistfile=_track_file, verbose=False)
        st = StarEvolTrack.__new__(StarEvolTrack)
        st.tracks = tracks
        return st

    @staticmethod
    def _nelder_mead_reference(st, loga, mini, eep, feh, afe, smf, tol):
        """Previous implementation (scipy Nelder-Mead on afe=0 tracks)."""
        from scipy.optimize import minimize

        aidx = st.tracks.predictions.index("loga")

        def loss(x):
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x[0]
            try:
                loga_pred = st.tracks.get_predictions([mini * smf, x, feh, 0.0])[aidx]
                return (loga_pred - loga) ** 2
            except Exception:
                return 1e6

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(loss, eep, method="Nelder-Mead")
        if res.fun < tol**2:
            return res.x[0]
        return np.nan

    def test_equivalence_and_speed(self, star_track):
        st = star_track
        tracks = st.tracks
        aidx = tracks.predictions.index("loga")
        tol = 1e-2

        cases = [
            # (mini, eep, feh, smf) spanning MS + varied masses/metallicities
            (1.0, 350.0, 0.0, 0.8),
            (1.0, 400.0, 0.0, 0.5),
            (1.2, 450.0, -0.5, 0.7),
            (0.8, 300.0, 0.25, 0.9),
            (1.5, 420.0, 0.0, 0.6),
            (2.0, 400.0, -1.0, 0.5),
            (0.6, 350.0, 0.0, 0.95),
            (1.1, 460.0, -0.25, 0.75),
        ]

        n_agree = 0
        t_new = t_ref = 0.0
        for mini, eep, feh, smf in cases:
            loga = tracks.get_predictions([mini, eep, feh, 0.0], apply_corr=False)[aidx]
            if not np.isfinite(loga):
                continue

            t0 = time.perf_counter()
            eep2_new = st._get_eep_for_secondary(loga, mini, eep, feh, 0.0, smf, tol)
            t_new += time.perf_counter() - t0

            t0 = time.perf_counter()
            eep2_ref = self._nelder_mead_reference(
                st, loga, mini, eep, feh, 0.0, smf, tol
            )
            t_ref += time.perf_counter() - t0

            if np.isfinite(eep2_ref):
                assert np.isfinite(eep2_new), (mini, eep, feh, smf)
                # Both must reproduce the target age within tolerance
                loga_new = tracks.get_predictions(
                    [mini * smf, eep2_new, feh, 0.0], apply_corr=False
                )[aidx]
                assert abs(loga_new - loga) < tol, (mini, eep, feh, smf)
                # The two solutions must agree closely in EEP (Nelder-Mead
                # terminates at xatol~1e-4; the age curve inversion is exact)
                assert abs(eep2_new - eep2_ref) < 1.0, (mini, eep, feh, smf)
                n_agree += 1

        assert n_agree >= 6  # the vast majority of cases must be comparable
        # Timing is reported as a diagnostic only: the agreement assertions
        # above are the test; a wall-clock race is flaky on shared runners.
        print(
            f"\nsecondary EEP solve ({n_agree} cases): vectorized inversion "
            f"{1e3 * t_new / len(cases):.2f} ms/solve vs Nelder-Mead "
            f"{1e3 * t_ref / len(cases):.2f} ms/solve "
            f"({t_ref / t_new:.1f}x faster)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
