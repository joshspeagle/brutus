#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slow integration tests for brutus core functionality.

These tests involve loading actual data files and should only be run
when explicitly requested. They test the full pipeline including
EEPTracks loading which takes significant time.
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Every test here is slow (loads multi-GB MIST data); deselect with
# `-m "not slow"`.
pytestmark = pytest.mark.slow


# Global session-scoped fixtures - loaded once for all slow tests
@pytest.fixture(scope="session")
def shared_eep_tracks():
    """Load EEPTracks once for all slow tests in this session."""
    from conftest import find_brutus_data_file

    if find_brutus_data_file("MIST_1.2_EEPtrk.h5") is None:
        pytest.skip(
            "MIST track data not available; run fetch_tracks() or set "
            "BRUTUS_DATA_DIR to enable these slow integration tests"
        )
    try:
        from brutus.core.individual import EEPTracks

        print("\n🔥 Loading EEPTracks (this may take ~1 minute)...")
        tracks = EEPTracks()
        print("✅ EEPTracks loaded successfully!")
        return tracks

    except (ImportError, FileNotFoundError) as e:
        raise  # Should not fail with all data available


@pytest.fixture(scope="session")
def shared_star_track(shared_eep_tracks):
    """Create StarEvolTrack instance using shared EEPTracks."""
    try:
        from brutus.core.individual import StarEvolTrack

        print("🔧 Creating StarEvolTrack from shared EEPTracks...")
        star_track = StarEvolTrack(tracks=shared_eep_tracks)
        print("✅ StarEvolTrack created successfully!")
        return star_track

    except (ImportError, FileNotFoundError) as e:
        raise  # Should not fail with all data available


class TestEEPTracksIntegration:
    """Integration tests for EEPTracks that require data loading."""

    def test_eep_tracks_basic_functionality(self, shared_eep_tracks):
        """Test basic EEPTracks functionality."""

        # Test parameter prediction for solar-type star
        mini, eep, feh, afe = 1.0, 350, 0.0, 0.0

        try:
            params = shared_eep_tracks.get_predictions([mini, eep, feh, afe])

            # Should return stellar parameters
            assert isinstance(params, np.ndarray)
            assert len(params) > 0

            # Parameters should be physically reasonable for solar-type star
            # (This is a basic sanity check)
            assert np.all(np.isfinite(params))

        except Exception as e:
            raise  # Should not fail with all data available

    def test_eep_tracks_mass_range(self, shared_eep_tracks):
        """Test EEPTracks over a range of stellar masses."""

        masses = [0.5, 1.0, 2.0]  # Low, solar, high mass
        eep, feh, afe = 350, 0.0, 0.0

        results = []
        for mini in masses:
            try:
                params = shared_eep_tracks.get_predictions([mini, eep, feh, afe])
                results.append(params)
            except Exception as e:
                raise  # Should not fail with all data available

        # Should have results for all masses
        assert len(results) == len(masses)

        # All results should be finite
        for params in results:
            assert np.all(np.isfinite(params))


class TestStarEvolTrackIntegration:
    """Integration tests for StarEvolTrack that require EEPTracks."""

    def test_star_track_sed_generation(self, shared_star_track):
        """Test SED generation for individual star."""

        # Solar-type star parameters
        mini, eep, feh, afe = 1.0, 350, 0.0, 0.0
        av, dist = 0.1, 1000.0  # Small extinction, 1 kpc

        try:
            sed, params, params2 = shared_star_track.get_seds(
                mini=mini, eep=eep, feh=feh, afe=afe, av=av, dist=dist
            )

            # Should return SEDs and parameters
            assert isinstance(sed, np.ndarray)
            assert isinstance(
                params, dict
            )  # params is a dictionary with stellar parameters
            assert len(sed) > 0

            # SEDs should be finite
            assert np.all(np.isfinite(sed))

            # Parameters should be finite (check dictionary values)
            param_values = list(params.values())
            assert all(
                np.isfinite(v)
                for v in param_values
                if isinstance(v, (int, float, np.number))
            )

            print(f"✅ Generated SED with {len(sed)} photometric bands")

        except Exception as e:
            raise  # Should not fail with all data available

    def test_star_track_multiple_seds(self, shared_star_track):
        """Test multiple SED generations using same StarEvolTrack."""

        # Test multiple stellar configurations representing typical stellar types
        test_cases = [
            {
                "mini": 0.8,
                "eep": 300,
                "feh": -0.5,
                "afe": 0.0,
                "av": 0.0,
                "dist": 500,
            },  # Low-mass, metal-poor
            {
                "mini": 1.0,
                "eep": 350,
                "feh": 0.0,
                "afe": 0.0,
                "av": 0.1,
                "dist": 1000,
            },  # Solar-type
            {
                "mini": 1.5,
                "eep": 400,
                "feh": 0.2,
                "afe": 0.0,
                "av": 0.05,
                "dist": 1500,
            },  # Higher-mass, metal-rich
        ]

        results = []
        for i, params in enumerate(test_cases):
            try:
                sed, stellar_params, params2 = shared_star_track.get_seds(**params)
                results.append((sed, stellar_params))

                # Basic validity checks
                assert isinstance(sed, np.ndarray)
                assert isinstance(
                    stellar_params, dict
                )  # stellar_params is a dictionary
                assert len(sed) > 0
                assert np.all(np.isfinite(sed))

                # Check stellar parameters are finite
                param_values = list(stellar_params.values())
                assert all(
                    np.isfinite(v)
                    for v in param_values
                    if isinstance(v, (int, float, np.number))
                )

            except Exception as e:
                raise  # Should not fail with all data available

        # Should have processed all test cases
        assert len(results) == len(test_cases)

        # Check that different parameters give different results
        seds = [result[0] for result in results]
        assert not np.array_equal(
            seds[0], seds[1]
        ), "Different stellar parameters should yield different SEDs"
        assert not np.array_equal(
            seds[1], seds[2]
        ), "Different stellar parameters should yield different SEDs"

        print(
            f"✅ Processed {len(results)} stellar configurations with varying results"
        )


class TestNeuralNetworkIntegration:
    """Integration tests for neural network functionality."""

    def test_neural_network_imports(self):
        """Test that neural network classes can be imported."""
        try:
            from brutus.core.neural_nets import FastNN, FastNNPredictor

            # Should be callable classes
            assert callable(FastNN)
            assert callable(FastNNPredictor)

        except ImportError as e:
            raise  # Should not fail with all data available

    def test_neural_network_predictor_creation(self):
        """Test creating FastNNPredictor with data file."""
        # Check for neural network file (both possible names)
        nn_file = None
        if os.path.exists("data/DATAFILES/nnMIST_BC.h5"):
            nn_file = "data/DATAFILES/nnMIST_BC.h5"
        elif os.path.exists("data/DATAFILES/nn_c3k.h5"):
            nn_file = "data/DATAFILES/nn_c3k.h5"
        else:
            # Try to find it using the helper
            from conftest import find_brutus_data_file

            nn_file = find_brutus_data_file("nn_c3k.h5")
            if nn_file is None:
                nn_file = find_brutus_data_file("nnMIST_BC.h5")

        assert nn_file is not None, "Neural network data file not available"

        try:
            from brutus.core.neural_nets import FastNNPredictor

            # Use actual filter names that exist in the NN file
            test_filters = ["SDSS_g", "SDSS_r", "2MASS_J", "2MASS_H", "Gaia_G_MAW"]

            predictor = FastNNPredictor(filters=test_filters, verbose=False)

            # Should have been initialized successfully
            assert hasattr(predictor, "filters")
            assert len(predictor.filters) == len(test_filters)

            print(
                f"✅ Neural network loaded successfully with {len(test_filters)} filters"
            )

        except Exception as e:
            raise  # Should not fail with all data available


class TestFullPipeline:
    """Integration tests for complete brutus workflows."""

    def test_end_to_end_individual_star(self, shared_star_track):
        """Test complete individual star modeling pipeline using shared fixtures."""

        try:
            # Import required modules
            from brutus.utils.photometry import magnitude

            # Solar-type star - simple, reliable test case
            mini, eep, feh, afe = (
                1.0,
                350,
                0.0,
                0.0,
            )  # afe=0.0 always (no alpha variation in models)
            av, dist = 0.0, 1000.0  # No extinction, 1 kpc

            # Generate SED using shared StarEvolTrack
            sed, params, params2 = shared_star_track.get_seds(
                mini=mini, eep=eep, feh=feh, afe=afe, av=av, dist=dist
            )

            # Convert to magnitudes (magnitude function needs flux and error)
            tiny_errors = np.full_like(
                sed, 0.001
            )  # Small errors for magnitude conversion
            mags, mag_errs = magnitude(sed, tiny_errors)
            print(f"✅ Magnitude conversion: {len(mags)} magnitudes")

            # Basic functionality checks
            assert len(mags) > 0
            print(f"✅ Length check passed")

            assert np.all(np.isfinite(mags))
            print(f"✅ Finite check passed")

            assert isinstance(params, dict)
            print(f"✅ Parameter type check passed")

            # Reasonable magnitude range (magnitudes can be negative for very bright sources)
            assert np.all(mags > -10)  # Not impossibly bright
            print(f"✅ Brightness range check passed: {mags.min():.2f} > -10")

            assert np.all(mags < 30)  # Not impossibly faint
            print(f"✅ Faintness range check passed: {mags.max():.2f} < 30")

            # Should have basic stellar parameters
            assert "logt" in params
            assert "logg" in params
            assert "loga" in params
            print(f"✅ Parameter key checks passed")

            print(
                f"🎉 End-to-end pipeline complete success: {len(mags)} magnitudes, T_eff={10**params['logt']:.0f}K"
            )

        except Exception as e:
            raise  # Should not fail with all data available


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
