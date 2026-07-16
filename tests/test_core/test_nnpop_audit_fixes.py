#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for the 2026-07 audit fixes to core neural nets and
population synthesis (group: nnpop).

Covers:
- FastNN._load_NN bounds-consistency check (elementwise, not value counting)
- FastNNPredictor.sed_batch with a single filter (squeeze shape bug)
- Isochrone `predictions` argument actually subsetting/reordering columns
- Isochrone rejecting unknown prediction names
- Isochrone default-path fallback to the pooch cache
- Isochrone singleton-afe padding via a zero-copy broadcast view
- StellarPop.get_seds_smf_grid matching per-SMF get_seds calls exactly
"""

import os

import h5py
import numpy as np
import pytest

from brutus.core.neural_nets import FastNN, FastNNPredictor
from brutus.core.populations import Isochrone, StellarPop


def _iso_file():
    from conftest import find_brutus_data_file

    return find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")


def _nn_file():
    from conftest import find_brutus_data_file

    nn = find_brutus_data_file("nnMIST_BC.h5")
    if nn is None:
        nn = find_brutus_data_file("nn_c3k.h5")
    return nn


@pytest.fixture(scope="module")
def real_isochrone():
    iso_file = _iso_file()
    if iso_file is None:
        pytest.skip("MIST isochrone data file not available")
    return Isochrone(mistfile=iso_file, verbose=False)


@pytest.fixture(scope="module")
def real_stellarpop(real_isochrone):
    nn_file = _nn_file()
    if nn_file is None:
        pytest.skip("Neural network data file not available")
    return StellarPop(
        isochrone=real_isochrone,
        filters=["SDSS_g", "SDSS_r", "SDSS_i"],
        nnfile=nn_file,
        verbose=False,
    )


def _write_nn_file(path, per_filter_bounds, seed=42):
    """Write a synthetic NN file in the FastNN HDF5 layout."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for fname, (xmin, xmax) in per_filter_bounds.items():
            g = f.create_group(fname)
            g["w1"] = rng.random((16, 6))
            g["b1"] = rng.random((16, 1))
            g["w2"] = rng.random((16, 16))
            g["b2"] = rng.random((16, 1))
            g["w3"] = rng.random((1, 16))
            g["b3"] = rng.random((1, 1))
            g["xmin"] = np.asarray(xmin, dtype=float)
            g["xmax"] = np.asarray(xmax, dtype=float)


class TestLoadNNBoundsCheck:
    """The bounds check must compare per-filter bounds elementwise."""

    def test_coincident_bound_values_accepted(self, tmp_path):
        """A consistent file where two parameters share a bound value
        (afe_min == av_min == 0) must load; the old value-counting check
        rejected it because len(np.unique(xmin)) < 6."""
        path = str(tmp_path / "coincident.h5")
        bounds = (
            [2500.0, -1.0, -4.0, 0.0, 0.0, 2.0],  # afe_min == av_min == 0.0
            [5.0e4, 5.5, 0.5, 0.6, 6.0, 5.0],
        )
        _write_nn_file(path, {"f1": bounds, "f2": bounds})
        nn = FastNN(filters=["f1", "f2"], nnfile=path, verbose=False)
        np.testing.assert_array_equal(nn.xmin, bounds[0])
        np.testing.assert_array_equal(nn.xmax, bounds[1])

    def test_permuted_bounds_rejected(self, tmp_path):
        """Per-filter bounds that are permutations of the same six values
        are inconsistent and must raise; the old check accepted them."""
        path = str(tmp_path / "permuted.h5")
        b1 = (
            [2500.0, -1.0, -4.0, -0.2, 0.0, 2.0],
            [5.0e4, 5.5, 0.5, 0.6, 6.0, 5.0],
        )
        # Teff/logg bounds swapped for the second filter
        b2 = (
            [-1.0, 2500.0, -4.0, -0.2, 0.0, 2.0],
            [5.5, 5.0e4, 0.5, 0.6, 6.0, 5.0],
        )
        _write_nn_file(path, {"f1": b1, "f2": b2})
        with pytest.raises(ValueError, match="different"):
            FastNN(filters=["f1", "f2"], nnfile=path, verbose=False)

    def test_real_file_still_loads(self):
        nn_file = _nn_file()
        if nn_file is None:
            pytest.skip("Neural network data file not available")
        nn = FastNN(filters=["SDSS_g", "SDSS_r"], nnfile=nn_file, verbose=False)
        assert nn.xmin.shape == (6,)


class TestSedBatchSingleFilter:
    """sed_batch must work when the predictor has a single filter."""

    def test_single_filter_batch_matches_scalar(self):
        nn_file = _nn_file()
        if nn_file is None:
            pytest.skip("Neural network data file not available")
        pred = FastNNPredictor(filters=["SDSS_g"], nnfile=nn_file, verbose=False)
        n = 5
        seds = pred.sed_batch(
            logt=np.full(n, 3.76),
            logg=np.full(n, 4.4),
            feh_surf=np.zeros(n),
            logl=np.zeros(n),
            afe=np.zeros(n),
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        assert seds.shape == (n, 1)
        assert np.all(np.isfinite(seds))
        ref = pred.sed(
            logt=3.76,
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        np.testing.assert_allclose(seds[:, 0], ref[0], rtol=1e-12)

    def test_single_valid_star_edge(self):
        """n_valid == 1 (squeeze also drops the sample axis)."""
        nn_file = _nn_file()
        if nn_file is None:
            pytest.skip("Neural network data file not available")
        pred = FastNNPredictor(filters=["SDSS_g"], nnfile=nn_file, verbose=False)
        seds = pred.sed_batch(
            logt=np.array([3.76, 99.0]),  # second star far out of bounds
            logg=np.array([4.4, 4.4]),
            feh_surf=np.zeros(2),
            logl=np.zeros(2),
            afe=np.zeros(2),
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        assert seds.shape == (2, 1)
        assert np.isfinite(seds[0, 0])
        assert np.isnan(seds[1, 0])


class TestIsochronePredictionsArgument:
    """`predictions` must select/reorder output columns, not just be stored."""

    def test_subset_reorder_columns(self):
        iso_file = _iso_file()
        if iso_file is None:
            pytest.skip("MIST isochrone data file not available")
        iso_def = Isochrone(mistfile=iso_file, verbose=False)
        iso_sub = Isochrone(
            mistfile=iso_file, predictions=["logt", "mini"], verbose=False
        )
        ref = iso_def.get_predictions(feh=0.0, afe=0.0, loga=9.0, apply_corr=False)
        sub = iso_sub.get_predictions(feh=0.0, afe=0.0, loga=9.0, apply_corr=False)
        assert sub.shape[1] == 2
        logt_idx = iso_def.pred_labels.index("logt")
        mini_idx = iso_def.pred_labels.index("mini")
        np.testing.assert_array_equal(sub[:, 0], ref[:, logt_idx])
        np.testing.assert_array_equal(sub[:, 1], ref[:, mini_idx])

    def test_corrections_applied_before_reordering(self):
        """Empirical corrections index columns by pred_labels; they must land
        on the right columns even when the output is reordered."""
        iso_file = _iso_file()
        if iso_file is None:
            pytest.skip("MIST isochrone data file not available")
        iso_def = Isochrone(mistfile=iso_file, verbose=False)
        reordered = list(reversed(iso_def.pred_labels))
        iso_rev = Isochrone(mistfile=iso_file, predictions=reordered, verbose=False)
        ref = iso_def.get_predictions(feh=-0.3, afe=0.0, loga=9.5, apply_corr=True)
        rev = iso_rev.get_predictions(feh=-0.3, afe=0.0, loga=9.5, apply_corr=True)
        np.testing.assert_array_equal(rev, ref[:, ::-1])

    def test_unknown_prediction_raises(self):
        iso_file = _iso_file()
        if iso_file is None:
            pytest.skip("MIST isochrone data file not available")
        with pytest.raises(ValueError, match="not available"):
            Isochrone(
                mistfile=iso_file, predictions=["mini", "not_a_label"], verbose=False
            )


class TestIsochroneDefaultPathFallback:
    """Isochrone() must fall back to the pooch cache like EEPTracks does."""

    def test_pooch_cache_used_when_repo_path_missing(self, tmp_path, monkeypatch):
        import brutus.core.populations as bcp

        cache_file = tmp_path / "MIST_1.2_iso_vvcrit0.0.h5"
        real_exists = os.path.exists

        def fake_exists(path):
            if str(path) == str(cache_file):
                return True
            if str(path).endswith("MIST_1.2_iso_vvcrit0.0.h5"):
                return False  # repo-relative default "missing"
            return real_exists(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        import pooch

        monkeypatch.setattr(pooch, "os_cache", lambda name: str(tmp_path))

        opened = {}

        def fake_h5file(path, mode="r"):
            opened["path"] = str(path)
            raise OSError("stop here")

        monkeypatch.setattr(bcp.h5py, "File", fake_h5file)

        with pytest.raises(RuntimeError):
            Isochrone(verbose=False)

        # The loader must have been pointed at the pooch cache copy.
        assert opened["path"] == str(cache_file)


class TestIsochroneAfePaddingMemory:
    """Singleton-afe padding must not materialize a duplicated grid."""

    def test_padded_afe_layers_share_memory(self, real_isochrone):
        iso = real_isochrone
        if iso.pred_grid.shape[1] != 2:
            pytest.skip("file is not singleton-afe")
        assert np.shares_memory(iso.pred_grid[:, 0], iso.pred_grid[:, 1])
        # zero-stride broadcast along the padded axis
        assert iso.pred_grid.strides[1] == 0

    def test_interpolation_still_works(self, real_isochrone):
        preds = real_isochrone.get_predictions(feh=0.0, afe=0.0, loga=9.0)
        assert np.isfinite(preds).any()
        # out-of-tolerance afe still yields NaN (fill_value behavior)
        preds_bad = real_isochrone.get_predictions(feh=0.0, afe=0.3, loga=9.0)
        assert not np.isfinite(preds_bad).any()


class TestGetSedsSmfGrid:
    """get_seds_smf_grid must exactly match per-SMF get_seds calls."""

    @pytest.mark.parametrize("return_dict", [True, False])
    def test_matches_get_seds(self, real_stellarpop, return_dict):
        pop = real_stellarpop
        eep_grid = np.linspace(300.0, 500.0, 40)
        smf_values = [0.0, 0.5, 1.0]
        kw = dict(
            feh=0.0,
            loga=9.5,
            av=0.1,
            rv=3.1,
            eep=eep_grid,
            dist=1000.0,
            mini_bound=0.08,
            return_dict=return_dict,
        )
        results = pop.get_seds_smf_grid(smf_values, **kw)
        assert len(results) == len(smf_values)
        for smf, (sed_g, p_g, p2_g) in zip(smf_values, results):
            sed_r, p_r, p2_r = pop.get_seds(binary_fraction=smf, **kw)
            np.testing.assert_array_equal(sed_g, sed_r)
            if return_dict:
                assert set(p_g) == set(p_r)
                for k in p_r:
                    np.testing.assert_array_equal(p_g[k], p_r[k])
                    np.testing.assert_array_equal(p2_g[k], p2_r[k])
            else:
                np.testing.assert_array_equal(p_g, p_r)
                np.testing.assert_array_equal(p2_g, p2_r)

    def test_slices_are_independent_copies(self, real_stellarpop):
        """Mutating one slice's SEDs must not leak into another slice."""
        pop = real_stellarpop
        eep_grid = np.linspace(300.0, 400.0, 10)
        r = pop.get_seds_smf_grid(
            [0.0, 0.0], feh=0.0, loga=9.5, eep=eep_grid, mini_bound=0.08
        )
        r[0][0][:] = -99.0
        assert not np.array_equal(r[0][0], r[1][0])
