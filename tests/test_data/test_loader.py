"""Tests of load_models and load_offsets in brutus.data.loader."""

import numpy as np
import pytest
from conftest import requires_data

from brutus.data.loader import find_nn_file, load_models, load_offsets


class TestDataLoaderFunctions:
    """Unit tests for data loading functions."""

    def test_load_models_error_handling(self):
        """Test load_models error handling."""
        from brutus.data.loader import load_models

        # Test with non-existent file
        with pytest.raises((FileNotFoundError, OSError)):
            load_models("/non/existent/file.h5")

    @pytest.mark.slow
    def test_load_models_comprehensive_real_data(self):
        """Comprehensive test of load_models with real MIST v9 data (single load for efficiency)."""
        from brutus.data.loader import load_models

        data_file = requires_data("grid_mist_v9.h5")

        # Single load with reasonable filter set - this covers most functionality
        test_filters = [
            "SDSS_g",
            "SDSS_r",
            "SDSS_i",
            "SDSS_u",
            "SDSS_z",
        ]  # 5 filters for good coverage
        models, labels, label_mask = load_models(
            data_file, filters=test_filters, verbose=False
        )

        # Verify output shapes and types
        assert isinstance(models, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(label_mask, np.ndarray)

        assert models.ndim == 3  # (n_models, n_filters, n_coeffs)
        assert models.shape[0] > 0  # Should have models
        assert models.shape[1] == len(test_filters)  # Should match filter count
        assert models.shape[2] == 3  # Should have 3 coefficients per filter

        assert len(labels) == len(models)  # Labels should match models
        assert len(label_mask) == 1  # Should have one mask row

        # Test that all models have finite values (no NaNs/Infs)
        assert np.all(np.isfinite(models))

        # Test that labels contain expected stellar parameters
        expected_params = ["mini", "feh", "eep"]  # Most basic ones
        available_params = labels.dtype.names
        for param in expected_params:
            if param in available_params:
                assert np.all(np.isfinite(labels[param]))

    @pytest.mark.slow
    def test_load_models_filtering_options_fast(self):
        """Test load_models filtering options with minimal I/O."""

        from brutus.data.loader import load_models

        data_file = requires_data("grid_mist_v9.h5")

        # Get baseline with fast filter set
        fast_filters = ["SDSS_g", "SDSS_r"]  # Just 2 filters for speed
        models_full, _, _ = load_models(data_file, filters=fast_filters, verbose=False)

        # Test main sequence only
        models_ms, _, _ = load_models(
            data_file, filters=fast_filters, include_postms=False, verbose=False
        )
        assert len(models_ms) > 0
        assert len(models_ms) < len(models_full)  # Should be subset

        # The cut applies when the labels omit 'eep'.
        models_ms_labels, _, _ = load_models(
            data_file,
            filters=fast_filters,
            labels=["mini", "feh"],
            include_postms=False,
            verbose=False,
        )
        assert len(models_ms_labels) == len(models_ms)

        # Test post-main sequence only
        models_postms, _, _ = load_models(
            data_file, filters=fast_filters, include_ms=False, verbose=False
        )
        assert len(models_postms) > 0
        assert len(models_postms) < len(models_full)  # Should be subset

        # Test excluding binaries
        models_singles, _, _ = load_models(
            data_file, filters=fast_filters, include_binaries=False, verbose=False
        )
        assert len(models_singles) > 0
        assert len(models_singles) <= len(models_full)  # Should be <= total

    @pytest.mark.slow
    def test_load_models_performance_no_scaling(self):
        """Test that load_models has minimal performance scaling with filter count."""

        from brutus.data.loader import load_models

        data_file = requires_data("grid_mist_v9.h5")

        # Test with small and large filter sets (just validate shapes, not timing)
        models_small, _, _ = load_models(
            data_file, filters=["SDSS_g", "SDSS_r"], verbose=False
        )
        models_large, _, _ = load_models(
            data_file,
            filters=["SDSS_g", "SDSS_r", "SDSS_i", "SDSS_u", "SDSS_z", "2MASS_J"],
            verbose=False,
        )

        # Results should be valid
        assert len(models_small) > 0
        assert len(models_large) > 0
        assert models_small.shape[1] == 2  # 2 filters
        assert models_large.shape[1] == 6  # 6 filters
        assert models_small.shape[0] == models_large.shape[0]  # Same number of models

    @pytest.mark.slow
    def test_load_models_with_custom_labels(self):
        """Test load_models with custom label selection (fast version)."""

        from brutus.data.loader import load_models

        data_file = requires_data("grid_mist_v9.h5")

        # Test with specific labels (minimal filters for speed)
        custom_labels = ["mini", "feh", "eep", "loga"]
        test_filters = ["SDSS_g"]  # Just 1 filter for speed
        models, labels, label_mask = load_models(
            data_file, filters=test_filters, labels=custom_labels, verbose=False
        )

        # Basic validation
        assert len(models) > 0
        assert models.shape[1] == 1  # 1 filter

        # Verify that requested labels are accessible
        available_labels = [
            name for name in labels.dtype.names if not np.all(np.isnan(labels[name]))
        ]
        assert len(available_labels) > 0  # Should have some valid labels

    def test_both_phases_excluded_raises_before_reading_the_file(self):
        with pytest.raises(ValueError, match="include_ms and include_postms"):
            load_models("/non/existent/file.h5", include_ms=False, include_postms=False)

    def test_load_offsets_with_real_mist_v9_data(self):
        """Test load_offsets with real MIST v9 offset data."""

        from brutus.data.loader import load_offsets

        offset_file = requires_data("offsets_mist_v9.txt")

        # Test loading all offsets
        offsets = load_offsets(offset_file, verbose=False)

        # Verify output
        assert isinstance(offsets, np.ndarray)
        assert len(offsets) > 0
        assert np.all(np.isfinite(offsets))

        # Offsets should be close to 1.0 (small corrections)
        assert np.all(np.abs(offsets - 1.0) < 0.5)  # Reasonable range

        # Test with specific filters
        test_filters = ["PS_g", "PS_r", "PS_i"]
        offsets_gri = load_offsets(offset_file, filters=test_filters, verbose=False)
        assert len(offsets_gri) == len(test_filters)
        assert np.all(np.isfinite(offsets_gri))

    def test_load_offsets_reads_file_order_and_defaults_to_one(self, tmp_path):
        p = tmp_path / "offsets.txt"
        p.write_text("PS_r 0.98\nPS_g 1.02\n")
        offsets = load_offsets(str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=False)
        np.testing.assert_array_equal(offsets, [1.02, 0.98, 1.0])

    def test_load_offsets_duplicate_filter_raises(self, tmp_path):
        p = tmp_path / "offsets.txt"
        p.write_text("PS_g 1.02\nPS_g 1.03\nPS_r 0.98\n")
        with pytest.raises(ValueError, match="Something went wrong when extracting"):
            load_offsets(str(p), filters=["PS_g", "PS_r"], verbose=False)

    def test_load_offsets_warns_only_for_unknown_filter_names(self, tmp_path):
        import warnings

        p = tmp_path / "offsets.txt"
        p.write_text("PS_g 1.02\nMyBand 1.05\n")
        with pytest.warns(UserWarning, match="PS_gg"):
            offsets = load_offsets(str(p), filters=["PS_g", "PS_gg"], verbose=False)
        np.testing.assert_array_equal(offsets, [1.02, 1.0])
        # A known name absent from the file, or a custom name in the file,
        # does not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            offsets = load_offsets(str(p), filters=["PS_r", "MyBand"], verbose=False)
        np.testing.assert_array_equal(offsets, [1.0, 1.05])

    def test_load_offsets_verbose_prints_signed_percent(self, tmp_path, capsys):
        p = tmp_path / "offsets.txt"
        p.write_text("PS_g 1.1234\nPS_r 0.95\n")
        load_offsets(str(p), filters=["PS_g", "PS_r"], verbose=True)
        err = capsys.readouterr().err
        assert "PS_g (+12.34%)" in err
        assert "PS_r (-5.00%)" in err


def _write_grid(
    path,
    n_models=6,
    filters=("PS_g", "PS_r", "PS_i"),
    nan_coeff_rows=(),
    nan_param_rows=(),
    label_fields=("mini", "feh", "eep", "afe", "smf"),
    param_fields=("loga", "logl", "logt", "logg", "agewt"),
    afe_values=(0.0,),
    eep_range=(300.0, 420.0),
    smf_values=(0.0,),
):
    """Write a synthetic HDF5 grid in GridGenerator._save_grid format."""
    import h5py

    rng = np.random.default_rng(42)

    stype = np.dtype([(f, "f4", 3) for f in filters])
    mag_coeffs = np.zeros(n_models, dtype=stype)
    for f in filters:
        mag_coeffs[f] = rng.uniform(5.0, 15.0, size=(n_models, 3)).astype("f4")
    for i in nan_coeff_rows:
        for f in filters:
            mag_coeffs[f][i] = np.nan

    ltype = np.dtype([(n, "f8") for n in label_fields])
    labels = np.zeros(n_models, dtype=ltype)
    if "mini" in label_fields:
        labels["mini"] = np.linspace(0.6, 1.4, n_models)
    if "eep" in label_fields:
        labels["eep"] = np.linspace(*eep_range, n_models)
    if "afe" in label_fields:
        labels["afe"] = np.resize(afe_values, n_models)
    if "smf" in label_fields:
        labels["smf"] = np.resize(smf_values, n_models)

    ptype = np.dtype([(n, "f8") for n in param_fields])
    params = np.zeros(n_models, dtype=ptype)
    for n in param_fields:
        params[n] = rng.uniform(0.0, 10.0, n_models)
    for i in nan_param_rows:
        for n in param_fields:
            params[n][i] = np.nan

    with h5py.File(path, "w") as f:
        f.create_dataset("mag_coeffs", data=mag_coeffs)
        f.create_dataset("labels", data=labels)
        f.create_dataset("parameters", data=params)
    return path


class TestLoadModelsRegressions:
    """Regression tests for load_models robustness fixes."""

    def test_drops_nonfinite_model_rows(self, tmp_path, capsys):
        """NaN coefficient rows (invalid grid points) must not survive loading.

        Old behavior passed them through, crashing BruteForce.loglike_grid
        with 'zero-size array to reduction operation maximum' for EVERY star.
        """
        p = tmp_path / "grid_nan.h5"
        _write_grid(p, n_models=8, nan_coeff_rows=(0, 3, 5))

        models, labels, _ = load_models(
            str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=True
        )
        assert len(models) == 5
        assert np.all(np.isfinite(models))
        assert len(labels) == len(models)
        err = capsys.readouterr().err
        assert "Extracting 3 requested filters" in err
        assert "Dropping 3 models with non-finite" in err

    def test_label_columns_survive_invalid_first_row(self, tmp_path):
        """Label availability must be decided per-column, not from row 0.

        Old behavior inspected only combined_labels[0]: a NaN-parameter first
        row silently discarded loga/logl/... for the ENTIRE grid.
        """
        p = tmp_path / "grid_row0.h5"
        _write_grid(p, n_models=5, nan_param_rows=(0,))

        _, labels, _ = load_models(
            str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=False
        )
        for name in ("loga", "logl", "logt", "logg", "agewt"):
            assert name in labels.dtype.names, f"{name} column was dropped"

    def test_phase_cut_without_eep_column_raises(self, tmp_path):
        """A phase cut on a file without 'eep' raises instead of being skipped."""
        p = tmp_path / "grid_noeep.h5"
        _write_grid(p, n_models=5, label_fields=("mini", "feh"))

        with pytest.raises(ValueError, match="include_postms"):
            load_models(str(p), filters=["PS_g"], include_postms=False, verbose=False)
        models, _, _ = load_models(str(p), filters=["PS_g"], verbose=False)
        assert len(models) == 5

    @pytest.mark.parametrize("include_ms", [True, False])
    def test_phase_cut_applies_when_eep_is_not_requested(self, tmp_path, include_ms):
        p = tmp_path / "grid_phase.h5"
        _write_grid(p, n_models=10, eep_range=(400.0, 500.0))

        kwargs = dict(
            filters=["PS_g"],
            include_ms=include_ms,
            include_postms=not include_ms,
            verbose=False,
        )
        _, default_labels, _ = load_models(str(p), **kwargs)
        _, labels, _ = load_models(str(p), labels=["mini", "feh"], **kwargs)
        assert 0 < len(default_labels) < 10
        np.testing.assert_array_equal(labels["mini"], default_labels["mini"])
        assert labels.dtype.names == ("mini", "feh")

    def test_binary_cut_applies_when_smf_is_not_requested(self, tmp_path):
        p = tmp_path / "grid_smf.h5"
        _write_grid(p, n_models=6, smf_values=(0.0, 0.5))

        models, labels, _ = load_models(
            str(p), filters=["PS_g"], labels=["mini", "feh"], verbose=False
        )
        assert len(models) == 3
        models, _, _ = load_models(
            str(p),
            filters=["PS_g"],
            labels=["mini", "feh"],
            include_binaries=True,
            verbose=False,
        )
        assert len(models) == 6

    def test_warns_on_missing_requested_labels(self, tmp_path):
        import warnings

        p = tmp_path / "grid_labels.h5"
        _write_grid(p, n_models=4)

        with pytest.warns(UserWarning, match="bogus"):
            _, labels, _ = load_models(
                str(p), filters=["PS_g"], labels=["mini", "bogus"], verbose=False
            )
        assert labels.dtype.names == ("mini",)
        # The default labels include 'Mr', which this file lacks.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_models(str(p), filters=["PS_g"], verbose=False)

    def test_warns_on_missing_filters(self, tmp_path):
        """Requested filters absent from the file must be reported, and the
        surviving column order must be recoverable from the warning."""
        p = tmp_path / "grid_filt.h5"
        _write_grid(p, n_models=4)

        with pytest.warns(UserWarning, match="TYPO_x"):
            models, _, _ = load_models(
                str(p), filters=["PS_g", "TYPO_x", "PS_r"], verbose=False
            )
        assert models.shape[1] == 2  # only the two real filters

        # No warning when every requested filter exists.
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            load_models(str(p), filters=["PS_g", "PS_r"], verbose=False)

    def test_raises_when_no_filters_survive(self, tmp_path):
        """If none of the requested filters exist, fail fast with a clear
        error instead of returning a (Nmodel, 0, 3) array that breaks
        StarGrid/BruteForce with an opaque shape error later."""
        p = tmp_path / "grid_nofilt.h5"
        _write_grid(p, n_models=4)

        with pytest.raises(ValueError, match="None of the requested filters"):
            load_models(str(p), filters=["TYPO_x", "TYPO_y"], verbose=False)

    def test_default_labels_include_afe(self, tmp_path):
        """Multi-afe grids must keep 'afe' so label tuples stay unique."""
        p = tmp_path / "grid_afe.h5"
        _write_grid(p, n_models=6, afe_values=(-0.2, 0.0, 0.4))

        _, labels, _ = load_models(
            str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=False
        )
        assert "afe" in labels.dtype.names
        assert len(np.unique(labels["afe"])) == 3

    def test_constant_afe_dropped_from_default_labels(self, tmp_path):
        """Single-afe grids drop the constant 'afe' column under default
        labels (pre-existing schema preserved; mirrors the constant-'smf'
        convention)."""
        p = tmp_path / "grid_afe_const.h5"
        _write_grid(p, n_models=6, afe_values=(0.0,))

        _, labels, _ = load_models(
            str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=False
        )
        assert "afe" not in labels.dtype.names

    def test_explicit_afe_kept_even_when_constant(self, tmp_path):
        """An explicitly requested 'afe' label is honored even if constant."""
        p = tmp_path / "grid_afe_expl.h5"
        _write_grid(p, n_models=6, afe_values=(0.0,))

        _, labels, _ = load_models(
            str(p),
            filters=["PS_g", "PS_r", "PS_i"],
            labels=["mini", "feh", "eep", "afe"],
            verbose=False,
        )
        assert "afe" in labels.dtype.names
        assert np.all(labels["afe"] == 0.0)


def test_find_nn_file_searches_the_data_directory_first(tmp_path, monkeypatch):
    from brutus.data.download import strato

    nnfile = tmp_path / "nn_c3k.h5"
    nnfile.touch()
    monkeypatch.setattr(strato, "path", tmp_path)
    assert find_nn_file() == nnfile


def test_find_nn_file_names_the_data_directory_when_absent(tmp_path, monkeypatch):
    from brutus.data.download import strato

    monkeypatch.setattr(strato, "path", tmp_path)
    with pytest.raises(FileNotFoundError, match="ASTRO_BRUTUS_DATA_DIR"):
        find_nn_file(possible_names=("absent.h5",))


def test_find_nn_file_skips_directories_and_rejects_a_string(tmp_path, monkeypatch):
    from brutus.data.download import strato

    (tmp_path / "nn_c3k.h5").mkdir()
    monkeypatch.setattr(strato, "path", tmp_path)
    with pytest.raises(FileNotFoundError):
        find_nn_file(possible_names=("nn_c3k.h5",))
    with pytest.raises(TypeError, match="possible_names"):
        find_nn_file("nn_c3k.h5")


class TestLoadOffsetsSingleRow:
    """Regression: a single-filter offsets file must not crash."""

    def test_single_row_file(self, tmp_path):
        # np.loadtxt collapses a one-row file to shape (2,); the old arr.T
        # unpack then yielded 0-d scalars and a "nonzero on 0d arrays"
        # ValueError downstream.
        p = tmp_path / "offsets_single.txt"
        p.write_text("PS_g 1.02\n")
        out = load_offsets(str(p), filters=["PS_g"], verbose=False)
        assert out.shape == (1,)
        assert np.isclose(out[0], 1.02)

    def test_single_row_missing_filter(self, tmp_path):
        p = tmp_path / "offsets_single.txt"
        p.write_text("PS_g 1.02\n")
        out = load_offsets(str(p), filters=["PS_g", "PS_r"], verbose=False)
        assert np.isclose(out[0], 1.02)
        assert out[1] == 1.0  # not in file -> no-offset default
