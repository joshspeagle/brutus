#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression and equivalence tests for StarGrid neighbor finding.

Covers the audit fixes to `brutus.core.individual.StarGrid`:

1. Multilinear interpolation must NOT silently collapse onto a far-away
   surviving corner when the query's dominant bracketing corner is missing
   from an incomplete grid (falls back to KD-tree instead).
2. No NaN weights (division by zero) when all surviving corners carry zero
   weight.
3. Both neighbour methods reject queries that omit a multi-valued axis,
   or that give a non-finite or out-of-range label.
4. The O(1) corner-index lookup must be equivalent to the previous
   boolean-scan implementation (and faster).
"""

import itertools
import warnings

import numpy as np
import pytest

from brutus.core.individual import StarGrid


def _make_grid(label_tuples, label_names=("mini", "eep", "feh"), seed=0):
    """Build a StarGrid from a list of label tuples with distinct coeffs."""
    rng = np.random.default_rng(seed)
    n = len(label_tuples)
    labels = np.zeros(n, dtype=[(name, "f8") for name in label_names])
    for i, tup in enumerate(label_tuples):
        labels[i] = tup
    models = np.zeros((n, 3, 3))
    # Distinct, identifiable magnitude coefficients per model
    models[:, :, 0] = 10.0 + np.arange(n)[:, None]
    models[:, :, 1] = rng.uniform(0.5, 2.0, (n, 3))
    models[:, :, 2] = rng.uniform(0.05, 0.2, (n, 3))
    params = np.zeros(n, dtype=[("pidx", "f8")])
    params["pidx"] = np.arange(n, dtype=float)
    return StarGrid(models, labels, params, verbose=False)


@pytest.fixture
def incomplete_grid():
    """Grid where the (mini=1.0, eep=400) corner is missing.

    mini axis {0.5, 1.0}, eep axis {200, 300, 400}, feh axis {0.0};
    the mini=1.0 'track' truncates at eep=300 (mimics real MIST track
    truncation, where the grid is not a complete Cartesian product).
    """
    tuples = [
        (0.5, 200, 0.0),
        (0.5, 300, 0.0),
        (0.5, 400, 0.0),
        (1.0, 200, 0.0),
        (1.0, 300, 0.0),
    ]
    return _make_grid(tuples)


@pytest.fixture
def complete_grid():
    """Complete 2 x 2 x 2 grid."""
    tuples = [
        (m, e, z) for m in (0.5, 1.0) for e in (200.0, 300.0) for z in (-0.5, 0.0)
    ]
    return _make_grid(tuples)


def _reference_multilinear(grid, **kwargs):
    """Previous boolean-scan implementation (reference for equivalence)."""
    req_params = {}
    for key in ["mini", "eep", "feh", "afe", "smf"]:
        if key in kwargs and kwargs[key] is not None:
            req_params[key] = kwargs[key]

    if not req_params:
        return np.array([0]), np.array([1.0])

    bracket_info = {}
    for param, value in req_params.items():
        if param in grid.grid_axes:
            axis_values = grid.grid_axes[param]
            idx = np.searchsorted(axis_values, value)
            if idx == 0:
                idx_low = idx_high = 0
                weight_high = 1.0
            elif idx >= len(axis_values):
                idx_low = idx_high = len(axis_values) - 1
                weight_high = 1.0
            else:
                idx_low = idx - 1
                idx_high = idx
                val_low = axis_values[idx_low]
                val_high = axis_values[idx_high]
                if val_high > val_low:
                    weight_high = (value - val_low) / (val_high - val_low)
                else:
                    weight_high = 0.5
            bracket_info[param] = {
                "indices": [idx_low, idx_high] if idx_low != idx_high else [idx_low],
                "weights": (
                    [1.0 - weight_high, weight_high] if idx_low != idx_high else [1.0]
                ),
                "values": (
                    axis_values[[idx_low, idx_high]]
                    if idx_low != idx_high
                    else axis_values[[idx_low]]
                ),
            }

    param_names = list(bracket_info.keys())
    index_combinations = itertools.product(
        *[bracket_info[p]["indices"] for p in param_names]
    )
    weight_combinations = itertools.product(
        *[bracket_info[p]["weights"] for p in param_names]
    )

    indices, weights = [], []
    for idx_combo, wt_combo in zip(index_combinations, weight_combinations):
        sel = np.ones(grid.nmodels, dtype=bool)
        for param_name, param_idx in zip(param_names, idx_combo):
            param_val = bracket_info[param_name]["values"][
                bracket_info[param_name]["indices"].index(param_idx)
            ]
            sel &= grid.labels[param_name] == param_val
        for param in grid.label_names:
            if param not in req_params and param in [
                "mini",
                "eep",
                "feh",
                "afe",
                "smf",
            ]:
                if param in grid.grid_axes:
                    sel &= grid.labels[param] == grid.grid_axes[param][0]
        grid_idx = np.where(sel)[0]
        if len(grid_idx) > 0:
            indices.append(grid_idx[0])
            weights.append(np.prod(wt_combo))

    if not indices:
        return None  # old code fell back to the KD-tree here

    indices = np.array(indices)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return indices, weights


def _as_weight_map(indices, weights):
    out = {}
    for i, w in zip(indices, weights):
        out[int(i)] = out.get(int(i), 0.0) + float(w)
    return out


class TestMissingCornerFallback:
    """Finding: missing bracketing corner must not silently dominate."""

    def test_dominant_corner_missing_does_not_collapse(self, incomplete_grid):
        # eep=399.9 at mini=1.0: the dominant (weight 0.999) corner
        # (1.0, 400) does not exist. The old code renormalized the surviving
        # weight-0.001 corner (1.0, 300) up to 1.0 -- a silent 100-EEP error.
        idx, w = incomplete_grid._find_neighbors_multilinear(
            mini=1.0, eep=399.9, feh=0.0
        )
        wmap = _as_weight_map(idx, w)
        # The old behavior put weight ~1.0 on model 4 = (1.0, 300)
        assert wmap.get(4, 0.0) < 0.9
        assert np.isclose(sum(wmap.values()), 1.0)

        # And the reference (old) implementation indeed collapses -- i.e.
        # this test fails on the old code
        ref = _reference_multilinear(incomplete_grid, mini=1.0, eep=399.9, feh=0.0)
        ref_map = _as_weight_map(*ref)
        assert ref_map.get(4, 0.0) > 0.999

    def test_minor_corner_missing_renormalizes_with_warning(self, incomplete_grid):
        # eep=300.1 at mini=1.0: surviving corner (1.0, 300) carries weight
        # 0.999 -- renormalization is fine, but must be announced.
        with pytest.warns(UserWarning, match="missing"):
            idx, w = incomplete_grid._find_neighbors_multilinear(
                mini=1.0, eep=300.1, feh=0.0
            )
        wmap = _as_weight_map(idx, w)
        assert wmap.get(4, 0.0) > 0.99


class TestZeroWeightCorners:
    """Corners with zero weight are not looked up."""

    def test_query_on_a_node_next_to_a_missing_corner_does_not_warn(self):
        # The (0.5, 300) corner is missing; a query on the (1.0, 300) node
        # gives it weight 0
        grid = _make_grid([(0.5, 200.0, 0.0), (1.0, 200.0, 0.0), (1.0, 300.0, 0.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            idx, w = grid._find_neighbors_multilinear(mini=1.0, eep=300.0, feh=0.0)
        assert _as_weight_map(idx, w) == {2: 1.0}


class TestZeroWeightNormalization:
    """Finding: zero surviving weight must not produce NaN weights/SEDs."""

    def test_exact_grid_point_with_missing_model(self, incomplete_grid):
        # Query exactly at (1.0, 400): searchsorted brackets give the
        # missing (1.0, 400) corner weight 1.0 and the surviving corners
        # weight 0.0 -> old code divided by zero -> all-NaN output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx, w = incomplete_grid._find_neighbors_multilinear(
                mini=1.0, eep=400.0, feh=0.0
            )
        assert np.all(np.isfinite(w))
        assert np.isclose(np.sum(w), 1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed, params, _ = incomplete_grid.get_seds(mini=1.0, eep=400.0, feh=0.0)
        assert np.all(np.isfinite(sed))
        assert np.isfinite(params["pidx"])


class TestQueryValidation:
    """Both neighbour methods reject queries they cannot answer."""

    @pytest.mark.parametrize("use_multilinear", [True, False])
    def test_omitted_multivalued_axis_raises(self, complete_grid, use_multilinear):
        with pytest.raises(ValueError, match="feh"):
            complete_grid.get_seds(mini=1.0, eep=300.0, use_multilinear=use_multilinear)

    @pytest.mark.parametrize("use_multilinear", [True, False])
    @pytest.mark.parametrize("mini", [np.nan, np.inf, 1.5, 0.1])
    def test_nonfinite_or_out_of_range_label_raises(
        self, complete_grid, use_multilinear, mini
    ):
        # mini spans [0.5, 1.0]; these values used to return the edge model
        with pytest.raises(ValueError, match="mini"):
            complete_grid.get_seds(
                mini=mini, eep=300.0, feh=0.0, use_multilinear=use_multilinear
            )
        with pytest.raises(ValueError, match="mini"):
            complete_grid.get_predictions(
                mini=mini, eep=300.0, feh=0.0, use_multilinear=use_multilinear
            )

    @pytest.mark.parametrize("use_multilinear", [True, False])
    def test_query_within_edge_tolerance_returns_edge_model(
        self, complete_grid, use_multilinear
    ):
        edge, _, _ = complete_grid.get_seds(
            mini=1.0, eep=300.0, feh=0.0, use_multilinear=use_multilinear
        )
        near, _, _ = complete_grid.get_seds(
            mini=1.0 + 1e-9, eep=300.0, feh=0.0, use_multilinear=use_multilinear
        )
        np.testing.assert_allclose(near, edge, rtol=0, atol=1e-6)


class TestLabelNotInGrid:
    """A label that the grid lacks is ignored with a warning, not silently."""

    @pytest.mark.parametrize("use_multilinear", [True, False])
    def test_label_not_in_the_grid_warns(self, complete_grid, use_multilinear):
        with pytest.warns(UserWarning, match="afe"):
            with_afe, _, _ = complete_grid.get_seds(
                mini=1.0, eep=300.0, feh=0.0, afe=0.4, use_multilinear=use_multilinear
            )
        without, _, _ = complete_grid.get_seds(
            mini=1.0, eep=300.0, feh=0.0, use_multilinear=use_multilinear
        )
        np.testing.assert_array_equal(with_afe, without)

    @pytest.mark.parametrize("use_multilinear", [True, False])
    def test_one_value_label_accepts_its_value_and_rejects_others(
        self, use_multilinear
    ):
        tuples = [(m, e, 0.0, 0.0) for m in (0.5, 1.0) for e in (200.0, 300.0)]
        grid = _make_grid(tuples, label_names=("mini", "eep", "feh", "afe"))
        query = dict(mini=1.0, eep=300.0, feh=0.0, use_multilinear=use_multilinear)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with_afe, _, _ = grid.get_seds(afe=0.0, **query)
        np.testing.assert_array_equal(with_afe, grid.get_seds(**query)[0])
        with pytest.raises(ValueError, match="afe"):
            grid.get_seds(afe=0.4, **query)

    def test_smf_on_a_grid_without_smf_warns_once(self, complete_grid):
        with pytest.warns(UserWarning) as record:
            complete_grid.get_seds(mini=1.0, eep=300.0, feh=0.0, smf=0.5)
        assert len(record) == 1 and "smf" in str(record[0].message)


class TestUnspecifiedParameterError:
    """Finding: unspecified multi-valued axes were silently pinned to the
    axis minimum; omission now raises instead (there is no sensible
    default, and a warning is too easy to miss in pipeline logs)."""

    def test_raises_when_multivalued_axis_omitted(self, complete_grid):
        with pytest.raises(ValueError, match="feh"):
            complete_grid.get_seds(mini=1.0, eep=300.0)  # feh unspecified

    def test_error_names_the_axis_range(self, complete_grid):
        with pytest.raises(ValueError, match="not specified"):
            complete_grid.get_seds(mini=1.0, eep=300.0)

    def test_no_error_when_all_specified(self, complete_grid):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            complete_grid.get_seds(mini=1.0, eep=300.0, feh=0.0)

    def test_single_valued_axis_may_be_omitted(self, incomplete_grid):
        # feh axis has a single value; omitting it is unambiguous
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            incomplete_grid.get_seds(mini=1.0, eep=250.0)


class TestMultilinearEquivalence:
    """The O(1) lookup must match the previous boolean-scan implementation."""

    def test_equivalence_on_complete_grid(self):
        rng = np.random.default_rng(42)
        tuples = [
            (m, e, z)
            for m in (0.5, 0.8, 1.0, 1.3)
            for e in (200.0, 300.0, 400.0)
            for z in (-1.0, -0.5, 0.0)
        ]
        grid = _make_grid(tuples)

        queries = []
        # Random interior queries
        for _ in range(50):
            queries.append(
                dict(
                    mini=rng.uniform(0.5, 1.3),
                    eep=rng.uniform(200.0, 400.0),
                    feh=rng.uniform(-1.0, 0.0),
                )
            )
        # Exact grid points and axis values
        for tup in tuples[::5]:
            queries.append(dict(mini=tup[0], eep=tup[1], feh=tup[2]))

        for q in queries:
            new = grid._find_neighbors_multilinear(**q)
            ref = _reference_multilinear(grid, **q)
            assert ref is not None
            new_map = _as_weight_map(*new)
            # Corners with zero weight are skipped
            ref_map = {k: v for k, v in _as_weight_map(*ref).items() if v > 0}
            assert set(new_map) == set(ref_map), q
            for k in ref_map:
                assert np.isclose(new_map[k], ref_map[k], atol=1e-12), q

    def test_first_model_wins_on_duplicate_labels(self):
        # Two models share identical labels; the first must win (matches
        # the previous scan's np.where(sel)[0][0] behavior)
        tuples = [(1.0, 300.0, 0.0), (1.0, 300.0, 0.0), (0.5, 300.0, 0.0)]
        grid = _make_grid(tuples)
        idx, w = grid._find_neighbors_multilinear(mini=1.0, eep=300.0, feh=0.0)
        wmap = _as_weight_map(idx, w)
        assert wmap.get(0, 0.0) == pytest.approx(1.0)
        assert 1 not in wmap


class TestNonFiniteLabels:
    """A model with a NaN label is not a grid value on either path."""

    def test_nan_label_does_not_make_an_axis_multivalued(self):
        grid = _make_grid([(0.5, 200.0, 0.0), (1.0, 200.0, 0.0), (1.0, 300.0, np.nan)])
        assert list(grid.grid_axes["feh"]) == [0.0]
        grid.get_seds(mini=1.0, eep=200.0)  # feh is single-valued: may be omitted

    def test_kdtree_ignores_models_with_nan_labels(self):
        tuples = [(m, e, 0.0) for m in (0.5, 1.0) for e in (200.0, 300.0, 400.0)]
        grid = _make_grid(tuples + [(1.0, np.nan, 0.0)])
        low, _ = grid._find_neighbors_kdtree(mini=1.0, eep=200.0, feh=0.0)
        high, _ = grid._find_neighbors_kdtree(mini=1.0, eep=400.0, feh=0.0)
        assert low[0] == 3 and high[0] == 5
        assert 6 not in low and 6 not in high

    def test_grid_without_a_fully_finite_model_raises(self):
        with pytest.raises(ValueError, match="No model"):
            _make_grid([(0.5, np.nan, 0.0), (1.0, 200.0, np.nan)])

    def test_axis_without_finite_values_raises(self):
        with pytest.raises(ValueError, match="feh"):
            _make_grid([(0.5, 200.0, np.nan), (1.0, 200.0, np.nan)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
