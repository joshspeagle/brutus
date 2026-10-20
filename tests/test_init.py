"""Tests of the package-level ``brutus`` namespace."""

import functools
import importlib
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

SUBPACKAGES = ["analysis", "core", "data", "dust", "plotting", "priors", "utils"]
API_PAGES = Path(__file__).parents[1] / "docs" / "source" / "api"


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


@functools.cache
def _documented_names():
    """Return the full names of the objects that the API pages document."""
    names = set()
    directive = re.compile(r"\.\. (currentmodule|auto\w+)::\s*([\w.]+)")
    for page in API_PAGES.glob("*.md"):
        module = ""
        for kind, target in directive.findall(page.read_text()):
            if kind == "currentmodule":
                module = target
            elif target.startswith("brutus."):
                names.add(target)
            else:
                names.add(f"{module}.{target}")
    return names


@pytest.mark.parametrize("module", [f"brutus.{name}" for name in SUBPACKAGES])
def test_api_pages_document_every_name_in_all(module):
    """Each public name has an entry on an API page, under its package."""
    mod = importlib.import_module(module)
    missing = [n for n in mod.__all__ if f"{module}.{n}" not in _documented_names()]
    assert missing == []
