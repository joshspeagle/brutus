#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Subprocess-based test runner for brutus tutorial notebooks.

Each tutorial runs in an isolated subprocess to avoid namespace pollution.
Tutorials are executed via ``jupyter nbconvert --execute`` with a headless
matplotlib backend and configurable timeouts.

Usage::

    pytest tests/test_tutorials.py -v -m tutorial
    pytest tests/test_tutorials.py -v -k "tutorial_01"
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TUTORIALS_DIR = Path(__file__).resolve().parent.parent / "tutorials"

# Per-tutorial timeout in seconds (generous defaults; override with
# BRUTUS_TUTORIAL_TIMEOUT env var to scale uniformly).
_TIMEOUT_SCALE = float(os.environ.get("BRUTUS_TUTORIAL_TIMEOUT_SCALE", "1.0"))

TUTORIAL_CONFIG = {
    "tutorial_00_data_setup": {
        "timeout": int(120 * _TIMEOUT_SCALE),
        "requires_large_data": False,
    },
    "tutorial_01_individual_stars": {
        "timeout": int(300 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_02_populations": {
        "timeout": int(300 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_03_grids_performance": {
        "timeout": int(600 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_04_galactic_priors": {
        "timeout": int(300 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_05_fitting_individual": {
        "timeout": int(600 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_06_cluster_analysis": {
        "timeout": int(600 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_07_dust_mapping": {
        "timeout": int(300 * _TIMEOUT_SCALE),
        "requires_large_data": False,
    },
    "tutorial_08_photometric_calibration": {
        "timeout": int(300 * _TIMEOUT_SCALE),
        "requires_large_data": True,
    },
    "tutorial_09_utilities": {
        "timeout": int(120 * _TIMEOUT_SCALE),
        "requires_large_data": False,
    },
    "tutorial_10_plotting": {
        "timeout": int(180 * _TIMEOUT_SCALE),
        "requires_large_data": False,
    },
    "tutorial_11_results": {
        "timeout": int(180 * _TIMEOUT_SCALE),
        "requires_large_data": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _notebook_exists(name):
    """Check whether a tutorial notebook file exists."""
    return (TUTORIALS_DIR / f"{name}.ipynb").is_file()


def _check_committed_data(name):
    """
    Check whether a tutorial's required committed data files are present.

    Returns True if all *committed* (small) data files that a tutorial
    needs are present in the tutorials/ directory.
    """
    # Committed data files that live in tutorials/
    committed_files = {
        "Orion_l204.7_b-19.2.h5",
        "Orion_l204.7_b-19.2_mist.h5",
        "Orion_l204.7_b-19.2_bs.h5",
        "NGC_2682.fits",
    }

    # Map of tutorials to their required committed files
    committed_requirements = {
        "tutorial_07_dust_mapping": ["Orion_l204.7_b-19.2_mist.h5"],
        "tutorial_09_utilities": ["Orion_l204.7_b-19.2.h5"],
        "tutorial_10_plotting": ["Orion_l204.7_b-19.2_mist.h5"],
        "tutorial_11_results": ["Orion_l204.7_b-19.2_mist.h5"],
    }

    required = committed_requirements.get(name, [])
    for fname in required:
        if not (TUTORIALS_DIR / fname).is_file():
            return False
    return True


def _run_notebook(notebook_path, timeout):
    """
    Execute a notebook in a subprocess and return (success, output).

    Uses ``jupyter nbconvert --to notebook --execute`` so that
    each notebook runs in a completely isolated process.
    """
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    # Prevent numba JIT from slowing down tutorial execution excessively
    # (tutorials are functional tests, not performance tests)
    env.pop("NUMBA_DISABLE_JIT", None)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout={}".format(timeout),
        "--output",
        "/dev/null",
        str(notebook_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Extra grace period
            env=env,
            cwd=str(TUTORIALS_DIR),
        )
        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"Notebook timed out after {timeout}s"
    except Exception as e:
        return False, f"Execution error: {e}"


# ---------------------------------------------------------------------------
# Test collection
# ---------------------------------------------------------------------------

# Discover all tutorial notebooks
_discovered = sorted(TUTORIALS_DIR.glob("tutorial_*.ipynb"))
_notebook_names = [nb.stem for nb in _discovered]


def _make_tutorial_test(name):
    """Create a test function for a single tutorial notebook."""
    config = TUTORIAL_CONFIG.get(name, {"timeout": 300, "requires_large_data": True})

    @pytest.mark.tutorial
    def test_func():
        nb_path = TUTORIALS_DIR / f"{name}.ipynb"
        if not nb_path.is_file():
            pytest.skip(f"Notebook {name}.ipynb not found")

        # Skip large-data tutorials when data is unavailable
        if config["requires_large_data"]:
            # Quick check: try importing check_data_requirements
            try:
                sys.path.insert(0, str(TUTORIALS_DIR))
                from tutorial_utils import check_data_requirements

                # Extract tutorial number from name
                num_str = name.split("_")[1]
                tut_num = int(num_str)
                available, missing = check_data_requirements(tut_num, verbose=False)
                if not available:
                    pytest.skip(f"Missing data files for {name}: {missing}")
            except Exception:
                pass
            finally:
                if str(TUTORIALS_DIR) in sys.path:
                    sys.path.remove(str(TUTORIALS_DIR))
        else:
            # Non-large-data tutorials still need their committed files
            if not _check_committed_data(name):
                pytest.skip(f"Committed data files missing for {name}")

        success, output = _run_notebook(nb_path, config["timeout"])
        if not success:
            # Truncate output to avoid overwhelming test output
            max_output = 3000
            if len(output) > max_output:
                output = output[:max_output] + "\n... (truncated)"
            pytest.fail(f"Notebook {name} failed:\n{output}")

    test_func.__name__ = f"test_{name}"
    test_func.__qualname__ = f"test_{name}"
    return test_func


# Dynamically create test functions for each discovered notebook
for _name in _notebook_names:
    globals()[f"test_{_name}"] = _make_tutorial_test(_name)

# Also create tests for notebooks that don't exist yet (will be skipped)
for _name in TUTORIAL_CONFIG:
    if _name not in _notebook_names:
        globals()[f"test_{_name}"] = _make_tutorial_test(_name)
