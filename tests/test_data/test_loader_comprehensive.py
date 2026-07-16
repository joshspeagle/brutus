#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus data loader utilities.

Note: This file previously contained extensive mock-based tests that have been
removed in favor of superior real data tests in test_data_comprehensive.py.

The real data tests provide better coverage by testing actual functionality
with real MIST data files rather than artificial mock scenarios.

All mock-based test classes removed:
- TestLoadModels: Redundant with real data tests
- TestLoadModelsErrorHandling: Artificial error scenarios
- TestLoadOffsets: Redundant with real data tests
- TestLoadOffsetsErrorHandling: Artificial error scenarios
- TestDataLoaderIntegration: Mock integration tests
- TestDataLoaderPerformance: Mock performance tests

For actual functionality testing, see:
- tests/test_data/test_data_comprehensive.py (real MIST data tests)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brutus.data.loader import load_models, load_offsets

# All test classes removed - see test_data_comprehensive.py for real data tests
# that provide superior functional coverage with actual MIST data files.


class TestPlaceholder:
    """Placeholder class to keep file structure valid."""

    def test_imports(self):
        """Test that imports work correctly."""
        # Basic smoke test to ensure imports don't fail
        assert callable(load_models)
        assert callable(load_offsets)

        # This minimal test ensures the file doesn't break pytest discovery
        # while we rely on the superior real data tests in test_data_comprehensive.py
        pass


def _write_grid(
    path,
    n_models=6,
    filters=("PS_g", "PS_r", "PS_i"),
    nan_coeff_rows=(),
    nan_param_rows=(),
    label_fields=("mini", "feh", "eep", "afe", "smf"),
    param_fields=("loga", "logl", "logt", "logg", "agewt"),
    afe_values=(0.0,),
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
        labels["eep"] = np.linspace(300.0, 420.0, n_models)
    if "afe" in label_fields:
        labels["afe"] = np.resize(afe_values, n_models)

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

    def test_drops_nonfinite_model_rows(self, tmp_path):
        """NaN coefficient rows (invalid grid points) must not survive loading.

        Old behavior passed them through, crashing BruteForce.loglike_grid
        with 'zero-size array to reduction operation maximum' for EVERY star.
        """
        p = tmp_path / "grid_nan.h5"
        _write_grid(p, n_models=8, nan_coeff_rows=(0, 3, 5))

        models, labels, _ = load_models(
            str(p), filters=["PS_g", "PS_r", "PS_i"], verbose=False
        )
        assert len(models) == 5
        assert np.all(np.isfinite(models))
        assert len(labels) == len(models)

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

    def test_ms_cut_without_eep_column_keeps_models(self, tmp_path):
        """A file lacking 'eep' must skip the MS/post-MS cut, not return 0 models.

        Old behavior compared the all-NaN eep column (NaN <= 454 is False
        elementwise) and silently returned an empty grid.
        """
        p = tmp_path / "grid_noeep.h5"
        _write_grid(p, n_models=5, label_fields=("mini", "feh"))

        models, _, _ = load_models(
            str(p),
            filters=["PS_g", "PS_r", "PS_i"],
            include_postms=False,
            verbose=False,
        )
        assert len(models) == 5

        # And the labels argument omitting 'eep' must not raise either.
        models, _, _ = load_models(
            str(p),
            filters=["PS_g", "PS_r", "PS_i"],
            labels=["mini", "feh"],
            include_postms=False,
            verbose=False,
        )
        assert len(models) == 5

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
