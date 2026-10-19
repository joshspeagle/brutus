#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus individual stellar modeling module.

This test suite covers:
1. EEPTracks class initialization and methods
2. StarEvolTrack class initialization and methods
3. Error handling and edge cases
4. Cache management and performance
5. Integration between EEPTracks and StarEvolTrack
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import find_brutus_data_file

from brutus.core.individual import EEPTracks, StarEvolTrack


class TestEEPTracksInitialization:
    """Test EEPTracks class initialization."""

    def test_eep_tracks_with_real_cached_data(self):
        """The shipped tracks give finite predictions for a solar-mass star."""
        v12_file = find_brutus_data_file("MIST_1.2_EEPtrk.h5")
        if v12_file is None:
            pytest.skip("MIST_1.2_EEPtrk.h5 not available")

        tracks = EEPTracks(mistfile=v12_file, verbose=False)
        preds = tracks.get_predictions([[1.0, 300, 0.0, 0.0]])  # mini, eep, feh, afe
        assert preds.shape == (1, len(tracks.predictions))
        assert np.all(np.isfinite(preds))


class TestStarEvolTrackInitialization:
    """Test StarEvolTrack class initialization."""

    @pytest.fixture
    def mock_eep_tracks(self):
        """Create a mock EEPTracks for StarEvolTrack testing."""
        tracks = MagicMock()
        tracks.get_predictions.return_value = np.random.random(6)  # 6 parameters
        tracks.pred_labels = ["age", "logl", "logt", "logr", "logg", "phase"]
        return tracks

    def test_star_evol_track_basic_initialization(self, mock_eep_tracks):
        """Test StarEvolTrack initialization with basic parameters."""
        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=mock_eep_tracks, verbose=False)

            assert star_track.tracks is mock_eep_tracks
            assert hasattr(star_track, "filters")
            assert hasattr(star_track, "predictor")

    def test_star_evol_track_custom_filters(self, mock_eep_tracks):
        """Test StarEvolTrack initialization with custom filters."""
        custom_filters = ["U", "B", "V", "R", "I"]

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(
                tracks=mock_eep_tracks, filters=custom_filters, verbose=False
            )

            assert star_track.filters == custom_filters

    def test_star_evol_track_verbose_output(self, mock_eep_tracks, capsys):
        """Test that verbose output is produced when requested."""
        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=mock_eep_tracks, verbose=True)

            captured = capsys.readouterr()
            # Should contain some verbose output about initialization
            assert len(captured.err) > 0


class TestStarEvolTrackMethods:
    """Test StarEvolTrack class methods."""

    @pytest.fixture
    def mock_star_evol_track(self):
        """Create a mock StarEvolTrack for testing."""
        tracks = MagicMock()
        tracks.get_predictions.return_value = np.random.random(7)
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

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
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

        with patch("brutus.core.neural_nets.FastNNPredictor"):
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

        with patch("brutus.core.neural_nets.FastNNPredictor") as mock_nn:
            mock_predictor = MagicMock()
            mock_nn.return_value = mock_predictor

            star_track = StarEvolTrack(tracks=tracks, verbose=False)
            star_track.predictor = mock_predictor

            # Should raise RuntimeError when prediction fails (line 842)
            with pytest.raises(
                RuntimeError, match="Failed to generate stellar parameters"
            ):
                star_track.get_seds(mini=1.0, eep=350, feh=0.0, afe=0.0)

    def test_get_seds_no_predictor_error(self):
        """Test get_seds when no neural network predictor available (line 830)."""
        tracks = MagicMock()

        # Create StarEvolTrack with no predictor
        star_track = StarEvolTrack.__new__(StarEvolTrack)
        star_track.tracks = tracks
        star_track.predictor = None  # No predictor

        # Should raise RuntimeError (line 830)
        with pytest.raises(
            RuntimeError, match="Neural network predictor not available"
        ):
            star_track.get_seds(mini=1.0, eep=350, feh=0.0, afe=0.0)

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


class TestIndividualStarsEdgeCases:
    """Test edge cases and error conditions."""

    def test_star_evol_track_invalid_inputs(self):
        """Test StarEvolTrack raises when predictor is unavailable."""
        tracks = MagicMock()
        tracks.get_predictions.return_value = np.random.random(6)
        tracks.pred_labels = ["age", "logl", "logt", "logr", "logg", "phase"]

        with patch("brutus.core.individual.FastNNPredictor") as mock_nn:
            mock_nn.return_value = MagicMock()

            star_track = StarEvolTrack(tracks=tracks, verbose=False)
            # Force predictor to None to simulate initialization failure
            star_track.predictor = None

            # Test that RuntimeError is raised when predictor is unavailable
            with pytest.raises(RuntimeError):
                star_track.get_seds(mini=1.0, eep=350, feh=0.0, afe=0.0, smf=2.0)

            with pytest.raises(RuntimeError):
                star_track.get_seds(mini=-1.0, eep=350, feh=0.0, afe=0.0)
