"""Shared helpers and fixtures for the brutus test suite.

Tests that need downloaded model data find it with `find_brutus_data_file`
and skip through `requires_data` when it is absent. CI downloads the data
with ``python .github/scripts/fetch_test_data.py``.

Coverage of ``@jit`` functions needs ``NUMBA_DISABLE_JIT=1``; plain test runs
keep JIT on so that the compiled code is what runs.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pooch
import pytest

# Fixtures that load a full data file; tests that use them are marked slow.
SLOW_FIXTURES = {"mist_ps_grid", "bayestar_map", "eep_tracks"}


def find_brutus_data_file(filename):
    """Return the path of a downloaded data file, or None if it is absent.

    The search order is ``$ASTRO_BRUTUS_DATA_DIR``, the Pooch cache, then
    ``data/DATAFILES`` in the repository (where CI links the files).
    """
    dirs = [
        os.environ.get("ASTRO_BRUTUS_DATA_DIR"),
        pooch.os_cache("astro-brutus"),
        Path(__file__).resolve().parents[1] / "data" / "DATAFILES",
    ]
    for directory in dirs:
        if directory is not None:
            path = Path(directory) / filename
            if path.exists():  # False for a broken symlink
                return str(path)
    return None


def requires_data(filename):
    """Return the path of a data file, or skip the calling test if it is absent."""
    path = find_brutus_data_file(filename)
    if path is None:
        pytest.skip(
            f"{filename} not found: set ASTRO_BRUTUS_DATA_DIR or run "
            "python .github/scripts/fetch_test_data.py"
        )
    return path


@contextmanager
def mock_nn_predictor(module):
    """Patch `FastNNPredictor` in `module` and the default NN file lookup."""
    with (
        patch(f"{module}.FastNNPredictor") as predictor,
        patch("brutus.data.loader.find_nn_file", return_value="nn.h5"),
    ):
        yield predictor


def split_labels(labels, label_mask):
    """Split `load_models` labels into grid labels and predictions."""
    names = labels.dtype.names
    grid = [n for n, is_grid in zip(names, label_mask[0]) if is_grid]
    pred = [n for n, is_grid in zip(names, label_mask[0]) if not is_grid]
    return labels[grid], labels[pred]


@pytest.fixture(scope="session")
def mist_ps_grid():
    """`load_models` output for grid_mist_v9.h5 with the five PS1 filters."""
    from brutus.data import load_models
    from brutus.data.filters import ps

    path = requires_data("grid_mist_v9.h5")
    return load_models(path, filters=ps[:5], verbose=False)


@pytest.fixture(scope="session")
def eep_tracks():
    """The MIST EEP tracks, loaded once per session."""
    from brutus.core.individual import EEPTracks

    return EEPTracks(mistfile=requires_data("MIST_1.2_EEPtrk.h5"), verbose=False)


@pytest.fixture(scope="session")
def bayestar_map():
    """The Bayestar19 dust map, loaded once per session."""
    from brutus.dust import Bayestar

    return Bayestar(dustfile=requires_data("bayestar2019_v1.h5"))


def pytest_collection_modifyitems(items):
    for item in items:
        if SLOW_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.slow)
