"""Tests of the package-level ``brutus`` namespace."""

import importlib
import subprocess
import sys
from importlib.metadata import version

import pytest

SUBPACKAGES = ["analysis", "core", "data", "dust", "plotting", "priors", "utils"]


def test_import_brutus_raises_when_a_dependency_is_missing():
    """A missing dependency raises an error that names it, not a hidden warning."""
    code = "import sys; sys.modules['healpy'] = None; import brutus"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "healpy" in result.stderr


def test_version_matches_package_metadata():
    """``brutus.__version__`` is the installed version, not a hard-coded copy."""
    import brutus

    assert brutus.__version__ == version("astro-brutus")


@pytest.mark.parametrize(
    "module",
    ["brutus"] + [f"brutus.{name}" for name in SUBPACKAGES],
)
def test_every_name_in_all_resolves(module):
    """Each name in ``__all__`` is an attribute of its package."""
    mod = importlib.import_module(module)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert missing == []
