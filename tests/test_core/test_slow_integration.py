#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests of EEPTracks, StarEvolTrack and FastNNPredictor on the
shipped data files."""

import numpy as np
import pytest
from conftest import requires_data


@pytest.fixture(scope="session")
def shared_star_track(eep_tracks):
    """Create StarEvolTrack instance using shared EEPTracks."""
    from brutus.core.individual import StarEvolTrack

    return StarEvolTrack(
        tracks=eep_tracks, nnfile=requires_data("nn_c3k.h5"), verbose=False
    )


class TestEEPTracksIntegration:
    """Integration tests for EEPTracks that require data loading."""

    def test_eep_tracks_basic_functionality(self, eep_tracks):
        """Test basic EEPTracks functionality."""

        # Test parameter prediction for solar-type star
        mini, eep, feh, afe = 1.0, 350, 0.0, 0.0

        params = eep_tracks.get_predictions([mini, eep, feh, afe])

        # Should return stellar parameters
        assert isinstance(params, np.ndarray)
        assert len(params) > 0

        # Parameters should be physically reasonable for solar-type star
        # (This is a basic sanity check)
        assert np.all(np.isfinite(params))

    def test_eep_tracks_mass_range(self, eep_tracks):
        """Test EEPTracks over a range of stellar masses."""

        masses = [0.5, 1.0, 2.0]  # Low, solar, high mass
        eep, feh, afe = 350, 0.0, 0.0

        results = []
        for mini in masses:
            params = eep_tracks.get_predictions([mini, eep, feh, afe])
            results.append(params)

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
            sed, stellar_params, params2 = shared_star_track.get_seds(**params)
            results.append((sed, stellar_params))

            # Basic validity checks
            assert isinstance(sed, np.ndarray)
            assert isinstance(stellar_params, dict)  # stellar_params is a dictionary
            assert len(sed) > 0
            assert np.all(np.isfinite(sed))

            # Check stellar parameters are finite
            param_values = list(stellar_params.values())
            assert all(
                np.isfinite(v)
                for v in param_values
                if isinstance(v, (int, float, np.number))
            )

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

    def test_neural_network_predictor_creation(self):
        """Test creating FastNNPredictor with data file."""
        nn_file = requires_data("nn_c3k.h5")

        from brutus.core.neural_nets import FastNNPredictor

        # Use actual filter names that exist in the NN file
        test_filters = ["SDSS_g", "SDSS_r", "2MASS_J", "2MASS_H", "Gaia_G_MAW"]

        predictor = FastNNPredictor(filters=test_filters, nnfile=nn_file, verbose=False)

        # Should have been initialized successfully
        assert hasattr(predictor, "filters")
        assert len(predictor.filters) == len(test_filters)

        print(f"✅ Neural network loaded successfully with {len(test_filters)} filters")


class TestFullPipeline:
    """Integration tests for complete brutus workflows."""

    def test_end_to_end_individual_star(self, shared_star_track):
        """Test complete individual star modeling pipeline using shared fixtures."""

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
        tiny_errors = np.full_like(sed, 0.001)  # Small errors for magnitude conversion
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
