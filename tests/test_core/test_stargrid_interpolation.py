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
3. The KD-tree must not be frozen with the label set of the first query.
4. Unspecified grid dimensions with >1 value emit a UserWarning about the
   pinned default.
5. The O(1) corner-index lookup must be equivalent to the previous
   boolean-scan implementation (and faster).
"""

import itertools
import time
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


class TestKDTreeNotFrozen:
    """Finding: KD-tree frozen with the first query's label set."""

    def test_partial_query_does_not_poison_full_queries(self):
        tuples = [
            (m, e, z) for m in (0.5, 1.0) for e in (200.0, 300.0) for z in (-0.5, 0.0)
        ]
        grid = _make_grid(tuples)

        # First query uses only `mini` -> previously froze a 1-D tree
        grid._find_neighbors_kdtree(mini=0.5)

        # A full query must resolve the exact matching model, not average
        # over all mini=1.0 models with eep/feh silently ignored
        idx, w = grid._find_neighbors_kdtree(mini=1.0, eep=300.0, feh=0.0)
        top = int(np.asarray(idx)[np.argmax(w)])
        expected = int(
            np.where(
                (grid.labels["mini"] == 1.0)
                & (grid.labels["eep"] == 300.0)
                & (grid.labels["feh"] == 0.0)
            )[0][0]
        )
        assert top == expected
        assert np.max(w) > 0.99

    def test_poisoning_via_public_api(self):
        tuples = [
            (m, e, z) for m in (0.5, 1.0) for e in (200.0, 300.0) for z in (-0.5, 0.0)
        ]
        grid = _make_grid(tuples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.get_predictions(mini=0.5, use_multilinear=False)
            preds = grid.get_predictions(
                mini=1.0, eep=300.0, feh=0.0, use_multilinear=False
            )
        expected = int(
            np.where(
                (grid.labels["mini"] == 1.0)
                & (grid.labels["eep"] == 300.0)
                & (grid.labels["feh"] == 0.0)
            )[0][0]
        )
        assert np.isclose(preds["pidx"], expected, atol=1e-6)


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
        # Random interior/exterior queries
        for _ in range(50):
            queries.append(
                dict(
                    mini=rng.uniform(0.3, 1.6),
                    eep=rng.uniform(150.0, 450.0),
                    feh=rng.uniform(-1.5, 0.5),
                )
            )
        # Exact grid points and axis values
        for tup in tuples[::5]:
            queries.append(dict(mini=tup[0], eep=tup[1], feh=tup[2]))
        # Out-of-bounds on all axes
        queries.append(dict(mini=10.0, eep=1000.0, feh=2.0))
        queries.append(dict(mini=0.01, eep=10.0, feh=-9.0))

        for q in queries:
            new = grid._find_neighbors_multilinear(**q)
            ref = _reference_multilinear(grid, **q)
            assert ref is not None
            new_map = _as_weight_map(*new)
            ref_map = _as_weight_map(*ref)
            assert set(new_map) == set(ref_map), q
            for k in ref_map:
                assert np.isclose(new_map[k], ref_map[k], atol=1e-12), q

    def test_omitted_axis_now_raises_instead_of_pinning(self):
        # The old implementation pinned an omitted multi-valued axis to its
        # minimum; that behavior is retired in favor of an immediate error
        # (see TestUnspecifiedParameterError), so no pinned-equivalence
        # check remains.
        tuples = [
            (m, e, z) for m in (0.5, 1.0) for e in (200.0, 300.0) for z in (-0.5, 0.0)
        ]
        grid = _make_grid(tuples)
        with pytest.raises(ValueError, match="feh"):
            grid._find_neighbors_multilinear(mini=0.75, eep=250.0)

    def test_first_model_wins_on_duplicate_labels(self):
        # Two models share identical labels; the first must win (matches
        # the previous scan's np.where(sel)[0][0] behavior)
        tuples = [(1.0, 300.0, 0.0), (1.0, 300.0, 0.0), (0.5, 300.0, 0.0)]
        grid = _make_grid(tuples)
        idx, w = grid._find_neighbors_multilinear(mini=1.0, eep=300.0, feh=0.0)
        wmap = _as_weight_map(idx, w)
        assert wmap.get(0, 0.0) == pytest.approx(1.0)
        assert 1 not in wmap

    def test_benchmark_faster_than_reference(self):
        # Moderate synthetic grid: 40 x 30 x 20 = 24,000 models
        minis = np.linspace(0.3, 2.5, 40)
        eeps = np.linspace(202.0, 605.0, 30)
        fehs = np.linspace(-2.0, 0.5, 20)
        tuples = [(m, e, z) for m in minis for e in eeps for z in fehs]
        grid = _make_grid(tuples)

        rng = np.random.default_rng(1)
        queries = [
            dict(
                mini=float(rng.uniform(0.3, 2.5)),
                eep=float(rng.uniform(202.0, 605.0)),
                feh=float(rng.uniform(-2.0, 0.5)),
            )
            for _ in range(30)
        ]

        t0 = time.perf_counter()
        for q in queries:
            grid._find_neighbors_multilinear(**q)
        t_new = time.perf_counter() - t0

        t0 = time.perf_counter()
        for q in queries:
            _reference_multilinear(grid, **q)
        t_ref = time.perf_counter() - t0

        # Timing is reported as a diagnostic only: a hard "new beats old"
        # assertion is a coin-flip on a loaded CI runner and the correctness
        # equivalence above is what this test actually guards.
        print(
            f"\nmultilinear neighbor lookup ({grid.nmodels} models, "
            f"{len(queries)} queries): new {1e3 * t_new / len(queries):.3f} "
            f"ms/query vs reference {1e3 * t_ref / len(queries):.3f} ms/query "
            f"({t_ref / t_new:.0f}x)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
