#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for grid generation module.

This module tests the GridGenerator class that creates pre-computed
stellar model grids with reddening coefficients.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from brutus.core.grid_generation import GridGenerator
from brutus.core.individual import EEPTracks, StarEvolTrack, StarGrid
from brutus.data.loader import load_models


class TestGridGenerator:
    """Test suite for GridGenerator class."""

    def test_init(self):
        """Test GridGenerator initialization."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        assert gen.tracks is tracks
        assert len(gen.filters) == 3
        assert all(f in ["SDSS_g", "SDSS_r", "SDSS_i"] for f in gen.filters)
        assert isinstance(gen.star_track, StarEvolTrack)

    def test_init_default_filters(self):
        """Test GridGenerator with default filters."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, verbose=False)

        # Should have many default filters
        assert len(gen.filters) > 3

    def test_make_grid_minimal(self):
        """Test grid generation with minimal parameters."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g", "SDSS_r"], verbose=False)

        # Create very small grid for testing
        mini_grid = np.array([1.0])
        eep_grid = np.array([350.0])
        feh_grid = np.array([0.0])
        afe_grid = np.array([0.0])
        smf_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=afe_grid,
            smf_grid=smf_grid,
            verbose=False,
        )

        # Check outputs
        assert hasattr(gen, "grid_labels")
        assert hasattr(gen, "grid_seds")
        assert hasattr(gen, "grid_params")
        assert hasattr(gen, "grid_sel")

        # Should have 1 model
        assert len(gen.grid_labels) == 1
        assert len(gen.grid_seds) == 1
        assert len(gen.grid_params) == 1

    def test_grid_structure(self):
        """Test structure of generated grid."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        # Create small grid
        mini_grid = np.array([0.8, 1.0, 1.2])
        eep_grid = np.array([300.0, 400.0])
        feh_grid = np.array([0.0])
        afe_grid = np.array([0.0])
        smf_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=afe_grid,
            smf_grid=smf_grid,
            verbose=False,
        )

        # Should have 3*2*1*1*1 = 6 models
        assert len(gen.grid_labels) == 6

        # Check labels structure
        assert "mini" in gen.grid_labels.dtype.names
        assert "eep" in gen.grid_labels.dtype.names
        assert "feh" in gen.grid_labels.dtype.names

        # Check SEDs structure (structured array with filter names)
        assert "SDSS_g" in gen.grid_seds.dtype.names
        assert "SDSS_r" in gen.grid_seds.dtype.names
        assert "SDSS_i" in gen.grid_seds.dtype.names

        # Each filter should have 3 coefficients
        assert gen.grid_seds["SDSS_g"].shape == (6, 3)

    def test_reference_distance(self):
        """Test that grid is generated at 1 kpc reference distance."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            dist=1000.0,  # Explicitly set to 1 kpc
            verbose=False,
        )

        # The base magnitude (first coefficient) should correspond to 1 kpc
        # We can verify this by comparing to direct StarEvolTrack call
        star_track = StarEvolTrack(tracks, filters=["SDSS_g"], verbose=False)
        sed_direct, _, _ = star_track.get_seds(
            mini=1.0,
            eep=350.0,
            feh=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
            return_dict=False,
        )

        # Base coefficient should match direct evaluation
        base_mag = gen.grid_seds["SDSS_g"][0, 0]
        np.testing.assert_allclose(base_mag, sed_direct[0], rtol=1e-3)

    def test_reddening_coefficients(self):
        """Test that reddening coefficients are plausible."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        # Check that we have 3 coefficients per filter
        for filt in ["SDSS_g", "SDSS_r", "SDSS_i"]:
            coeffs = gen.grid_seds[filt][0]
            assert len(coeffs) == 3

            # Base magnitude should be reasonable (roughly 0-10 at 1 kpc)
            assert -5 < coeffs[0] < 15

            # Av coefficient should be positive (extinction makes stars fainter)
            assert coeffs[1] > 0

            # Rv coefficient typically small
            assert abs(coeffs[2]) < 2.0

    def test_invalid_models_flagged(self):
        """Test that invalid models are properly flagged."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Include some parameters that should produce invalid models
        mini_grid = np.array([0.1, 1.0])  # 0.1 Msun too low
        eep_grid = np.array([350.0])
        feh_grid = np.array([0.0])

        gen.make_grid(
            mini_grid=mini_grid,
            eep_grid=eep_grid,
            feh_grid=feh_grid,
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            mini_bound=0.5,  # Should exclude 0.1 Msun
            verbose=False,
        )

        # Should have flagged some models as invalid
        assert not all(gen.grid_sel)

    def test_save_and_load(self):
        """Test saving grid to HDF5 and loading it back."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(
            tracks, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
        )

        # Generate small test grid
        mini_grid = np.array([0.9, 1.0, 1.1])
        eep_grid = np.array([350.0, 400.0])
        feh_grid = np.array([0.0])

        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate and save
            gen.make_grid(
                mini_grid=mini_grid,
                eep_grid=eep_grid,
                feh_grid=feh_grid,
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                output_file=tmp_path,
                verbose=False,
            )

            # Load with h5py to check structure
            with h5py.File(tmp_path, "r") as f:
                assert "mag_coeffs" in f
                assert "labels" in f
                assert "parameters" in f

                # Check attributes
                assert "reference_distance_pc" in f.attrs
                assert f.attrs["reference_distance_pc"] == 1000.0

                # Check dimensions
                assert f["mag_coeffs"].shape[0] == 6  # 3*2*1*1*1
                assert f["labels"].shape[0] == 6

            # Load with load_models
            models, labels, label_mask = load_models(
                tmp_path, filters=["SDSS_g", "SDSS_r", "SDSS_i"], verbose=False
            )

            assert len(models) > 0
            assert models.shape[1] == 3  # 3 filters
            assert models.shape[2] == 3  # 3 coefficients

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_grid_compatible_with_stargrid(self):
        """Test that generated grid can be loaded by StarGrid."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g", "SDSS_r"], verbose=False)

        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate small grid
            gen.make_grid(
                mini_grid=np.array([0.9, 1.0, 1.1]),
                eep_grid=np.array([350.0, 400.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                output_file=tmp_path,
                verbose=False,
            )

            # Load with load_models
            models, labels, label_mask = load_models(
                tmp_path, filters=["SDSS_g", "SDSS_r"], verbose=False
            )

            # Create StarGrid instance
            grid = StarGrid(models, labels, filters=["SDSS_g", "SDSS_r"], verbose=False)

            # Test that we can get predictions
            preds = grid.get_predictions(mini=1.0, eep=375.0, feh=0.0)

            # Should return valid predictions (dict or structured array)
            assert preds is not None
            # Grid was successfully created and loaded - functional test passed

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_different_dist_warning(self):
        """Test that using non-standard distance is preserved."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate with non-standard distance
            gen.make_grid(
                mini_grid=np.array([1.0]),
                eep_grid=np.array([350.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                dist=500.0,  # Non-standard!
                output_file=tmp_path,
                verbose=False,
            )

            # Check that distance is recorded
            with h5py.File(tmp_path, "r") as f:
                assert f.attrs["reference_distance_pc"] == 500.0

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_grid_params_structure(self):
        """Test that grid parameters match track predictions."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        # Check parameter names match tracks
        param_names = gen.grid_params.dtype.names
        track_predictions = tracks.predictions

        for pred in track_predictions:
            assert pred in param_names

    def test_binary_handling(self):
        """Test grid generation with binary stars."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Generate grid with and without binaries
        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0, 0.5]),  # Single + binary
            verbose=False,
        )

        # Should have 2 models
        assert len(gen.grid_labels) == 2

        # Binary should be brighter (smaller magnitude)
        mag_single = gen.grid_seds["SDSS_g"][0, 0]
        mag_binary = gen.grid_seds["SDSS_g"][1, 0]

        # Binary should be brighter (if both valid)
        if np.isfinite(mag_single) and np.isfinite(mag_binary):
            assert mag_binary < mag_single


class TestGridGeneratorValidity:
    """Tests for invalid-model handling and the save/load round trip."""

    def test_saved_grid_excludes_invalid_models_roundtrip(self):
        """Invalid (NaN) models must not be written to file, and a loaded
        user-generated grid must be immediately fit-ready.

        Old behavior wrote the full arrays including all-NaN rows and
        load_models passed them through, crashing BruteForce.loglike_grid
        with 'zero-size array to reduction operation maximum' for every star.
        """
        filters = ["SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"]
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=filters, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # eep=800 is unreachable within loga_max for the low-mass models,
            # producing a mix of valid and invalid grid points.
            gen.make_grid(
                mini_grid=np.array([0.5, 0.9, 1.0, 1.1]),
                eep_grid=np.array([800.0, 350.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                output_file=tmp_path,
                verbose=False,
            )
            n_valid = int(gen.grid_sel.sum())
            assert 0 < n_valid < len(gen.grid_sel), "need a mixed grid"

            # File must contain only the valid models, with finite data.
            with h5py.File(tmp_path, "r") as f:
                assert f["mag_coeffs"].shape[0] == n_valid
                assert f["labels"].shape[0] == n_valid
                assert f["parameters"].shape[0] == n_valid
                for filt in filters:
                    assert np.all(np.isfinite(f["mag_coeffs"][filt]))
                assert f.attrs["n_models_valid"] == n_valid

            # Round trip: load and check fit-readiness end to end.
            models, labels, label_mask = load_models(
                tmp_path, filters=filters, verbose=False
            )
            assert len(models) == n_valid
            assert np.all(np.isfinite(models))
            for name in ("mini", "eep", "feh", "loga"):
                assert name in labels.dtype.names

            from brutus.analysis import BruteForce

            grid = StarGrid(models, labels, filters=filters, verbose=False)
            fitter = BruteForce(grid, verbose=False)

            # Clean synthetic photometry from the first model at 1 kpc.
            flux = 10.0 ** (-0.4 * models[0, :, 0].astype(np.float64))
            err = 0.05 * flux
            mask = np.ones(len(filters), dtype=bool)
            results = fitter.loglike_grid(flux, err, mask)
            lnl = results[0]
            assert np.isfinite(lnl).any()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mini_bound_masks_subthreshold_primaries(self):
        """Primaries below mini_bound must be masked per the documented
        contract (old code only gated binary secondaries with it)."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([0.3, 1.0]),
            eep_grid=np.array([250.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            mini_bound=0.5,
            verbose=False,
        )

        assert not gen.grid_sel[0], "0.3 Msun primary must be masked"
        assert gen.grid_sel[1], "1.0 Msun primary must stay valid"
        assert np.all(np.isnan(np.array(gen.grid_seds[0].tolist())))

    def test_out_of_bounds_reddening_grid_flagged_invalid(self):
        """(av, rv) lattice points outside the NN training bounds yield NaN
        photometry; such models must be flagged invalid, not stored as valid
        NaN coefficients (old code checked validity only at the base SED)."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        av_beyond = gen.predictor.xmax[4] + 1.0  # above the NN A_V bound
        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            av_grid=np.array([0.0, 1.0, av_beyond]),
            verbose=False,
        )

        assert not gen.grid_sel[0]
        # n_models_valid must reflect the flagging (no overcounting).
        assert gen.grid_sel.sum() == 0

    def test_rejects_incompatible_tracks(self):
        """Objects lacking the EEPTracks interface (e.g. Isochrone) must be
        rejected with a clear TypeError at construction."""

        class NotTracks:
            pass

        with pytest.raises(TypeError, match="EEPTracks interface"):
            GridGenerator(NotTracks(), filters=["SDSS_g"], verbose=False)


class TestReddeningFitEquivalence:
    """The batched reddening fit must reproduce the old scalar-loop fit."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_matches_scalar_reference_and_is_faster(self):
        import time

        filters = ["SDSS_g", "SDSS_r", "SDSS_i"]
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=filters, verbose=False)

        dist = 1000.0
        av_grid = np.arange(0.0, 1.5 + 1e-5, 0.3)
        av_grid[-1] -= 1e-5
        av_wt = (1e-5 + av_grid) ** -1.0
        rv_grid = np.arange(2.4, 4.2 + 1e-5, 0.3)
        rv_wt = np.exp(-np.abs(rv_grid - 3.3) / 0.5)

        def reference_fit(mini, eep, feh, afe, smf, eep2, sed_base):
            """Old implementation: Nav x Nrv scalar get_seds calls."""
            seds = np.array(
                [
                    [
                        gen.star_track.get_seds(
                            mini=mini,
                            eep=eep,
                            feh=feh,
                            afe=afe,
                            smf=smf,
                            eep2=eep2,
                            av=av,
                            rv=rv,
                            dist=dist,
                            loga_max=10.14,
                            eep_binary_max=480.0,
                            mini_bound=0.5,
                            apply_corr=True,
                            corr_params=None,
                            return_dict=False,
                        )[0]
                        for av in av_grid
                    ]
                    for rv in rv_grid
                ]
            )
            sfits = np.array([np.polyfit(av_grid, s, 1, w=av_wt).T for s in seds])
            sedr, seda = np.polyfit(rv_grid, sfits[:, :, 0], 1, w=rv_wt)
            return np.c_[sed_base, seda, sedr]

        cases = [
            dict(mini=1.0, eep=350.0, feh=0.0, afe=0.0, smf=0.0),  # single
            dict(mini=1.2, eep=400.0, feh=-0.2, afe=0.0, smf=0.6),  # binary
            dict(mini=1.1, eep=454.0, feh=0.0, afe=0.0, smf=0.0),  # turnoff
        ]

        t_old = t_new = 0.0
        for c in cases:
            sed, params, params2, eep2 = gen.star_track.get_seds(
                av=0.0,
                rv=3.3,
                dist=dist,
                loga_max=10.14,
                eep_binary_max=480.0,
                mini_bound=0.5,
                apply_corr=True,
                corr_params=None,
                return_dict=False,
                return_eep2=True,
                **c,
            )
            assert np.all(np.isfinite(sed)), c

            t0 = time.perf_counter()
            ref = reference_fit(
                c["mini"], c["eep"], c["feh"], c["afe"], c["smf"], eep2, sed
            )
            t_old += time.perf_counter() - t0

            t0 = time.perf_counter()
            new = gen._fit_reddening_coefficients(
                params=params,
                params2=params2,
                sed_base=sed,
                av_grid=av_grid,
                av_wt=av_wt,
                rv_grid=rv_grid,
                rv_wt=rv_wt,
                dist=dist,
            )
            t_new += time.perf_counter() - t0

            np.testing.assert_allclose(new, ref, rtol=1e-8, atol=1e-8)

        # Measured ~17x on the dev machine; require only a loose margin so
        # the check is robust on loaded CI runners.
        assert t_new < t_old, f"batched fit slower than scalar ({t_new} vs {t_old})"


class TestGridGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_default_grids(self):
        """Test that default grid parameters are used when not specified.

        This test starts grid generation with all defaults (~300k models)
        but terminates early after verifying it's working correctly.
        """
        import signal
        import time

        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Patch make_grid to terminate after a few successful models
        original_get_seds = gen.star_track.get_seds
        call_count = [0]

        def limited_get_seds(*args, **kwargs):
            call_count[0] += 1
            result = original_get_seds(*args, **kwargs)
            # After 5 successful calls, we've verified defaults work
            if call_count[0] >= 5:
                raise KeyboardInterrupt("Test verified - defaults working")
            return result

        gen.star_track.get_seds = limited_get_seds

        try:
            # This will use all default grids (~300k models)
            # but will be interrupted after 5 successful model generations
            gen.make_grid(
                mini_grid=None,  # Use default
                eep_grid=None,  # Use default
                feh_grid=None,  # Use default
                afe_grid=None,  # Use default
                smf_grid=None,  # Use default
                verbose=False,
            )
        except KeyboardInterrupt:
            # Expected - we interrupted after verifying defaults work
            pass

        # Verify that defaults were applied and models were generated
        assert call_count[0] == 5, "Should have generated 5 models before interrupting"

        # Restore original method
        gen.star_track.get_seds = original_get_seds

    def test_empty_filter_list(self):
        """Test behavior with empty filter list."""
        tracks = EEPTracks(verbose=False)

        # Should use default filters
        gen = GridGenerator(tracks, filters=None, verbose=False)
        assert len(gen.filters) > 0

    def test_single_point_grid(self):
        """Test grid with single point in each dimension."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            verbose=False,
        )

        assert len(gen.grid_labels) == 1

    def test_grid_without_save(self):
        """Test generating grid without saving to file."""
        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=False)

        # Generate without output_file
        gen.make_grid(
            mini_grid=np.array([1.0]),
            eep_grid=np.array([350.0]),
            feh_grid=np.array([0.0]),
            afe_grid=np.array([0.0]),
            smf_grid=np.array([0.0]),
            output_file=None,  # Don't save
            verbose=False,
        )

        # Should still have results in memory
        assert hasattr(gen, "grid_seds")
        assert len(gen.grid_seds) == 1

    def test_verbose_output(self):
        """Test that verbose mode produces output."""
        import io
        import sys

        tracks = EEPTracks(verbose=False)
        gen = GridGenerator(tracks, filters=["SDSS_g"], verbose=True)

        # Capture stderr
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            # Generate small grid with verbose=True
            gen.make_grid(
                mini_grid=np.array([0.9, 1.0]),
                eep_grid=np.array([350.0, 400.0]),
                feh_grid=np.array([0.0]),
                afe_grid=np.array([0.0]),
                smf_grid=np.array([0.0]),
                verbose=True,
            )

            # Check that some output was produced
            output = sys.stderr.getvalue()
            assert "Generating grid" in output or "Grid generation" in output
        finally:
            sys.stderr = old_stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
