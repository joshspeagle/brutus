"""Tests of EEPTracks and StarEvolTrack in brutus.core.individual."""

import os
import pickle
import warnings
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from conftest import mock_nn_predictor, requires_data

from brutus.core.individual import (
    _EEPTRACKS_CACHE_ATTRS,
    _EEPTRACKS_CACHE_VERSION,
    EEPTracks,
    StarEvolTrack,
    rename,
)


class TestEEPTracksInitialization:
    """Test EEPTracks class initialization."""

    def test_eep_tracks_with_real_cached_data(self, eep_tracks):
        """The shipped tracks give finite predictions for a solar-mass star."""
        tracks = eep_tracks
        preds = tracks.get_predictions([[1.0, 300, 0.0, 0.0]])  # mini, eep, feh, afe
        assert preds.shape == (1, len(tracks.predictions))
        assert np.all(np.isfinite(preds))

    @pytest.mark.parametrize(
        "predictions",
        [
            ["loga", "logl", "logg"],
            ["logl", "logt", "logg"],
            ["loga", "logl", "logt", "logg", "bogus"],
        ],
    )
    def test_invalid_predictions_raise_before_reading_the_file(self, predictions):
        with pytest.raises(ValueError, match="predictions"):
            EEPTracks(mistfile="/non/existent.h5", predictions=predictions)


def test_get_corrections_rejects_malformed_corr_params():
    tracks = MagicMock()
    tracks.mini_idx, tracks.eep_idx, tracks.feh_idx = 0, 1, 2
    with pytest.raises(ValueError, match="corr_params"):
        EEPTracks.get_corrections(tracks, [0.8, 350.0, 0.0, 0.0], (0.09, -0.09))


class TestStarEvolTrackInitialization:
    """Test StarEvolTrack class initialization."""

    @pytest.fixture
    def mock_eep_tracks(self):
        """Create a mock EEPTracks for StarEvolTrack testing."""
        rng = np.random.RandomState(0)
        tracks = MagicMock()
        tracks.get_predictions.return_value = rng.random(6)  # 6 parameters
        tracks.pred_labels = ["age", "logl", "logt", "logr", "logg", "phase"]
        tracks.predictions = ["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]
        return tracks

    def test_star_evol_track_basic_initialization(self, mock_eep_tracks):
        """Test StarEvolTrack initialization with basic parameters."""
        with mock_nn_predictor("brutus.core.individual") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=mock_eep_tracks, verbose=False)

            assert star_track.tracks is mock_eep_tracks
            assert hasattr(star_track, "filters")
            assert hasattr(star_track, "predictor")

    def test_star_evol_track_custom_filters(self, mock_eep_tracks):
        """Test StarEvolTrack initialization with custom filters."""
        custom_filters = ["U", "B", "V", "R", "I"]

        with mock_nn_predictor("brutus.core.individual") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(
                tracks=mock_eep_tracks, filters=custom_filters, verbose=False
            )

            assert star_track.filters == custom_filters


class TestStarEvolTrackMethods:
    """Test StarEvolTrack class methods."""

    @pytest.fixture
    def mock_star_evol_track(self):
        """Create a mock StarEvolTrack for testing."""
        rng = np.random.RandomState(0)
        tracks = MagicMock()
        tracks.get_predictions.return_value = rng.random(7)
        tracks.pred_labels = [
            "loga",
            "logl",
            "logt",
            "logr",
            "logg",
            "feh_surf",
            "afe_surf",
        ]
        tracks.predictions = [
            "loga",
            "logl",
            "logt",
            "logr",
            "logg",
            "feh_surf",
            "afe_surf",
        ]
        tracks.labels = ["mini", "eep", "feh", "afe"]  # Input labels
        tracks.mini_bound = 0.08  # Minimum mass bound
        tracks.mini_bounds = np.array([0.08, 10.0])  # Mass bounds array

        with mock_nn_predictor("brutus.core.individual") as mock_nn:
            # Mock neural network predictor
            mock_predictor = MagicMock()
            # Ensure the predictor returns finite values for SED generation
            mock_predictor.sed.return_value = np.array(
                [15.0, 14.5, 14.0, 13.8, 13.5]
            )  # 5 filters (magnitudes)
            mock_predictor.NFILT = 5
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=tracks, verbose=False)
            star_track.filters = ["u", "g", "r", "i", "z"]  # Set filters for testing
            star_track.predictor = mock_predictor  # Explicitly set predictor

            return star_track

    def test_get_seds_basic_functionality(self, mock_star_evol_track):
        """Test basic get_seds functionality."""
        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0
        )

        # Check output types and shapes
        assert isinstance(sed, np.ndarray)
        assert isinstance(params, dict)  # Default return_dict=True
        assert isinstance(params2, dict)
        assert sed.ndim == 1  # Single star
        assert len(sed) == len(mock_star_evol_track.filters)
        assert np.all(np.isfinite(sed))

    def test_get_seds_with_extinction(self, mock_star_evol_track):
        """Test get_seds with extinction parameters."""
        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0, av=0.5, rv=3.1
        )

        assert isinstance(sed, np.ndarray)
        assert np.all(np.isfinite(sed))

    def test_get_seds_rejects_unknown_keywords(self, mock_star_evol_track):
        with pytest.raises(TypeError, match="Av"):
            mock_star_evol_track.get_seds(mini=1.0, eep=350, feh=0.0, Av=5.0)

    def test_get_seds_with_distance(self, mock_star_evol_track):
        """Test get_seds with distance modulus."""
        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0, dist=1000.0
        )

        assert isinstance(sed, np.ndarray)
        assert np.all(np.isfinite(sed))

    def test_get_seds_binary_star(self, mock_star_evol_track):
        """Test get_seds with binary star companion."""
        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0, smf=0.7  # Secondary mass fraction
        )

        assert isinstance(sed, np.ndarray)
        assert isinstance(params2, dict)
        assert np.all(np.isfinite(sed))
        # params2 should have secondary star information
        assert len(params2) > 0

    def test_get_seds_binary_primary_survives_failed_companion(self):
        """A valid primary SED must survive a non-converging companion.

        Regression: ``_get_eep_for_secondary`` used an unrealistically tight
        tolerance, so a companion whose age could not be matched returned NaN.
        That NaN propagated through ``add_mag`` and turned the ENTIRE combined
        SED into NaN, silently discarding the otherwise-valid primary during
        grid generation. The fix returns a primary-only SED in that case.

        Uses faithful mocks: ``get_predictions`` returns NaN for non-finite
        labels (and an unmatchable constant companion age), and the predictor
        returns a NaN SED for NaN stellar parameters -- unlike the shared
        fixture whose fixed return value masks the bug.
        """
        import warnings

        preds = ["loga", "logl", "logt", "logr", "logg", "feh_surf", "afe_surf"]

        def get_predictions(labels, **kwargs):
            labels = np.asarray(labels, dtype=float)
            if not np.all(np.isfinite(labels)):
                return np.full(7, np.nan)
            mini_l, feh_l, afe_l = labels[0], labels[2], labels[3]
            # Primary (mini~1.0) has age 9.0 dex; the companion's age is a
            # constant 9.05 dex regardless of EEP, so it can never be matched
            # -> eep2 = NaN. The primary itself is a perfectly valid track point.
            loga = 9.0 if np.isclose(mini_l, 1.0) else 9.05
            return np.array([loga, 0.0, 3.7, 0.0, 4.5, feh_l, afe_l])

        def sed(logl, logt, logg, feh_surf, afe, av, rv, dist):
            # Faithful NN: NaN stellar params -> NaN SED.
            if not np.all(np.isfinite([logl, logt, logg, feh_surf, afe])):
                return np.full(5, np.nan)
            return np.array([15.0, 14.5, 14.0, 13.8, 13.5])

        tracks = MagicMock()
        tracks.get_predictions.side_effect = get_predictions
        tracks.predictions = preds
        tracks.labels = ["mini", "eep", "feh", "afe"]
        tracks.mini_bound = 0.08

        with mock_nn_predictor("brutus.core.individual"):
            predictor = MagicMock()
            predictor.sed.side_effect = sed
            predictor.NFILT = 5
            st = StarEvolTrack(tracks=tracks, verbose=False)
            st.filters = ["u", "g", "r", "i", "z"]
            st.predictor = predictor

        # combine_seds=True: the combined SED must NOT be all-NaN (the bug).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_c, _, _ = st.get_seds(
                mini=1.0, eep=350, feh=0.0, afe=0.0, smf=0.7, combine_seds=True
            )
        sed_c = np.asarray(sed_c)
        assert np.all(np.isfinite(sed_c)), "valid primary destroyed by failed companion"

        # combine_seds=False: returns [primary, secondary]; primary stays finite.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_s, _, _ = st.get_seds(
                mini=1.0, eep=350, feh=0.0, afe=0.0, smf=0.7, combine_seds=False
            )
        assert np.all(np.isfinite(np.asarray(sed_s[0])))

    def test_get_seds_multiple_stars(self, mock_star_evol_track):
        """Test get_seds with different stellar parameters."""
        # Test with single star but different parameters to exercise more code paths
        sed1, _, _ = mock_star_evol_track.get_seds(mini=1.0, eep=350, feh=0.0, afe=0.0)
        sed2, _, _ = mock_star_evol_track.get_seds(mini=1.5, eep=400, feh=-0.5, afe=0.2)
        sed3, _, _ = mock_star_evol_track.get_seds(mini=2.0, eep=450, feh=0.5, afe=0.0)

        # Each should be valid SED
        for sed in [sed1, sed2, sed3]:
            assert isinstance(sed, np.ndarray)
            assert sed.shape == (5,)  # 5 filters
            assert np.all(np.isfinite(sed))

    def test_get_seds_array_output(self, mock_star_evol_track):
        """Test get_seds with return_dict=False."""
        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0, return_dict=False
        )

        assert isinstance(sed, np.ndarray)
        assert isinstance(params, np.ndarray)  # Should be array when return_dict=False
        assert isinstance(params2, np.ndarray)

    def test_get_seds_error_handling(self):
        """Test get_seds error handling paths."""
        # Test with tracks that fail prediction
        tracks = MagicMock()
        tracks.get_predictions.side_effect = Exception("Prediction failed")
        tracks.labels = ["mini", "eep", "feh", "afe"]
        tracks.predictions = ["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]

        with mock_nn_predictor("brutus.core.individual") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=tracks, verbose=False)
            star_track.predictor = mock_predictor

            # Should raise RuntimeError when prediction fails (line 842)
            with pytest.raises(
                RuntimeError, match="Failed to generate stellar parameters"
            ):
                star_track.get_seds(mini=1.0, eep=350, feh=0.0, afe=0.0)

    def test_tracks_without_sed_labels_raise(self):
        tracks = MagicMock()
        tracks.predictions = ["loga", "logl", "logt"]
        with pytest.raises(ValueError, match=r"lacks \['logg', 'feh_surf'"):
            StarEvolTrack(tracks=tracks, verbose=False)

    def test_unknown_filter_raises_at_construction(self):
        tracks = MagicMock()
        tracks.predictions = ["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]
        nnfile = requires_data("nn_c3k.h5")
        with pytest.raises(ValueError, match=r"filters \['BOGUS_V'\]"):
            StarEvolTrack(tracks, filters=["BOGUS_V"], nnfile=nnfile, verbose=False)

    def test_get_seds_old_star_age_cutoff(self, mock_star_evol_track):
        """Test get_seds with old stars exceeding loga_max cutoff."""
        # Mock tracks to return very old age
        mock_star_evol_track.tracks.get_predictions.return_value = np.array(
            [11.0, 1.5, 3.8, 0.2, 4.2, 0.0, 0.0]  # loga > 10.15 (very old star)
        )

        sed, params, params2 = mock_star_evol_track.get_seds(
            mini=1.0, eep=350, feh=0.0, afe=0.0, loga_max=10.15
        )

        # Should return NaN SED for very old stars
        assert isinstance(sed, np.ndarray)
        # Some values might be NaN due to age cutoff
        assert len(sed) == len(mock_star_evol_track.filters)


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


def test_single_afe_grid_is_padded_by_a_view_and_cached_unpadded(tmp_path):
    mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
    tracks = EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=False)
    assert tracks.ygrid.shape[-2] == 1
    assert tracks.interpolator.values.shape[-2] == 2
    assert np.shares_memory(tracks.interpolator.values, tracks.ygrid)

    with open(next(iter(tmp_path.glob("*.pkl"))), "rb") as f:
        cached = pickle.load(f)
    assert cached["ygrid"].shape == tracks.ygrid.shape
    assert "interpolator" not in cached


class TestBuildFromFile:
    """EEPTracks built from a small HDF5 file reproduces the stored tracks."""

    def test_predictions_at_a_grid_node_equal_the_stored_values(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        tracks = EEPTracks(
            mistfile=mistfile, predictions=PREDICTIONS, verbose=False, use_cache=False
        )
        eep = 210.0
        preds = tracks.get_predictions([1.0, eep, 0.0, 0.0], apply_corr=False)
        expected = [
            np.log10(AGE_SLOPE * eep),
            0.01 * (eep - 200.0) + 1.0,
            3.7 - 0.001 * (eep - 200.0),
            4.5 - 0.002 * (eep - 200.0),
            AGE_SLOPE,  # agewt = d(age)/d(EEP)
        ]
        assert tracks.predictions == PREDICTIONS + ["agewt"]
        np.testing.assert_allclose(preds, expected, rtol=1e-12)

    def test_corrupt_cache_falls_back_to_the_hdf5_file(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=False)
        cache_file = next(iter(tmp_path.glob("*.pkl")))
        cache_file.write_bytes(b"not a pickle")

        tracks = EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=False)
        preds = tracks.get_predictions([1.0, 210.0, 0.0, 0.0], apply_corr=False)
        np.testing.assert_allclose(preds[0], np.log10(AGE_SLOPE * 210.0), rtol=1e-12)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_tracks"):
            EEPTracks(mistfile=tmp_path / "absent.h5", verbose=False)

    def test_verbose_messages_report_build_cache_and_rebuild(self, tmp_path, capsys):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5")
        EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=True)
        err = capsys.readouterr().err
        assert "Loading evolutionary tracks" in err
        assert "EEPTracks cached to" in err

        EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=True)
        assert "Cached EEPTracks loaded successfully" in capsys.readouterr().err

        cache_file = next(iter(tmp_path.glob("*.pkl")))
        newer = cache_file.stat().st_mtime + 10.0
        os.utime(mistfile, (newer, newer))
        EEPTracks(mistfile=mistfile, predictions=PREDICTIONS, verbose=True)
        assert "Cache is older than data file" in capsys.readouterr().err

    def test_default_path_is_found_in_the_data_directory(self, tmp_path, monkeypatch):
        from brutus.data.download import strato

        cache_file = _make_synthetic_mist_file(tmp_path / "MIST_1.2_EEPtrk.h5")
        monkeypatch.setattr(strato, "path", tmp_path)
        tracks = EEPTracks(predictions=PREDICTIONS, verbose=False, use_cache=False)
        assert tracks.mistfile == cache_file


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

    def test_matches_per_column_reference(self, tmp_path):
        mistfile = _make_synthetic_mist_file(tmp_path / "tracks.h5", n_eep=200)

        # New implementation, captured before __init__ frees the arrays
        tracks = EEPTracks.__new__(EEPTracks)
        tracks.labels = ["mini", "eep", "feh", "afe"]
        tracks.predictions = list(PREDICTIONS)
        with h5py.File(mistfile, "r") as misth5:
            tracks._make_lib(misth5, verbose=False)

        ref_libparams, ref_output = self._reference_make_lib(
            mistfile, tracks.labels, tracks.predictions
        )

        assert tracks.libparams.dtype.names == ref_libparams.dtype.names
        for name in ref_libparams.dtype.names:
            np.testing.assert_array_equal(tracks.libparams[name], ref_libparams[name])
        np.testing.assert_array_equal(tracks.output, ref_output)


class TestAgeWeightJacobian:
    """agewt must be d(age)/d(EEP), not Delta(age) per array index."""

    def test_matches_a_per_track_gradient_on_unsorted_tracks(self):
        rng = np.random.default_rng(3)
        rows = []
        for mini, feh, n in [(0.5, 0.0, 7), (1.0, 0.0, 5), (1.0, -1.0, 1)]:
            eeps = 200.0 + np.cumsum(rng.uniform(0.5, 3.0, n))  # non-uniform
            for eep in eeps:
                rows.append((mini, eep, feh, 0.0, np.log10(1e7 * eep**1.5)))
        rows = [rows[i] for i in rng.permutation(len(rows))]

        tracks = EEPTracks.__new__(EEPTracks)
        tracks.libparams = np.array(
            [r[:4] for r in rows],
            dtype=[(n, "f8") for n in ("mini", "eep", "feh", "afe")],
        )
        tracks.output = np.array([[r[4]] for r in rows])
        tracks.predictions, tracks._ageidx = ["loga"], 0
        tracks._add_age_weights(verbose=False)

        lib = tracks.libparams
        expected = np.zeros(len(rows))
        for mini, feh in [(0.5, 0.0), (1.0, 0.0)]:
            idx = np.flatnonzero((lib["mini"] == mini) & (lib["feh"] == feh))
            idx = idx[np.argsort(lib["eep"][idx])]
            expected[idx] = np.gradient(10 ** tracks.output[idx, 0], lib["eep"][idx])
        np.testing.assert_array_equal(tracks.output[:, 1], expected)
        assert tracks.predictions == ["loga", "agewt"]

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

    def test_tracks_without_gridpoints_use_primary_afe(self):
        # Without an EEP grid axis, the Nelder-Mead fallback also solves at
        # the primary's [a/Fe]; it used 0.0 (#139).
        st = self._star_track()
        st.tracks.gridpoints = None
        afe, eep_p = 0.4, 350.0
        loga_target = float(st.tracks._loga(eep_p, afe))
        eep2 = st._get_eep_for_secondary(loga_target, 1.0, eep_p, 0.0, afe, 0.8, 1e-2)
        loga_2 = float(
            st.tracks.get_predictions([0.8, eep2, 0.0, afe], apply_corr=False)[0]
        )
        assert abs(loga_2 - loga_target) < 1e-3


class TestSecondaryEEPInversionRealTracks:
    """1-D inversion must match the old Nelder-Mead solve on real MIST."""

    @pytest.fixture(scope="class")
    def star_track(self, eep_tracks):
        st = StarEvolTrack.__new__(StarEvolTrack)
        st.tracks = eep_tracks
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

    def test_matches_nelder_mead_reference(self, star_track):
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
        for mini, eep, feh, smf in cases:
            loga = tracks.get_predictions([mini, eep, feh, 0.0], apply_corr=False)[aidx]
            if not np.isfinite(loga):
                continue

            eep2_new = st._get_eep_for_secondary(loga, mini, eep, feh, 0.0, smf, tol)
            eep2_ref = self._nelder_mead_reference(
                st, loga, mini, eep, feh, 0.0, smf, tol
            )

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
