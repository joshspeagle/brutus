"""Tests of the package-level ``brutus`` namespace."""

import functools
import importlib
import os
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


_JIT = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="needs numba caching"
)


def _run(code):
    """Run ``code`` in a fresh interpreter with numba's default cache
    location and return its stdout."""
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.slow
@_JIT
def test_cached_kernel_sees_a_change_in_the_module_it_calls(tmp_path):
    """Numba reuses a cache while the kernel's own file is unchanged, even if
    a jitted function it calls changed; the package locators reject it."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(
        "from numba import njit\nfrom pkg.b import g\n\n\n"
        "@njit(cache=True)\ndef f():\n    return g()\n"
    )
    callee = "from numba import njit\n\n\n@njit(cache=True)\ndef g():\n    return {}\n"
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n"
        "from brutus._numba_cache import install_cache_locators\n"
        f"install_cache_locators({str(pkg)!r})\n"
        "import pkg.a; print(pkg.a.f())"
    )
    (pkg / "b.py").write_text(callee.format(1.0))
    assert _run(code).strip() == "1.0"
    (pkg / "b.py").write_text(callee.format(2.0))
    assert _run(code).strip() == "2.0"


@_JIT
def test_every_cached_brutus_kernel_uses_the_package_cache_locator():
    """The locators are installed before brutus decorates any kernel."""
    from numba.core.caching import FunctionCache
    from numba.core.registry import CPUDispatcher

    import brutus
    from brutus._numba_cache import _canonical

    root = _canonical(os.path.dirname(brutus.__file__))
    kernels = [
        obj
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "brutus"
        for obj in vars(module).values()
        if isinstance(obj, CPUDispatcher) and isinstance(obj._cache, FunctionCache)
    ]
    assert kernels
    for kernel in kernels:
        locator = type(kernel._cache._impl.locator)
        assert getattr(locator, "package_root", None) == root, kernel


def test_package_stamp_skips_files_it_cannot_read(tmp_path):
    """A dangling symlink, such as an editor lock file, must not break the
    import; the stamp still sees a change to a code file."""
    from brutus._numba_cache import package_stamp

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".#a.py").symlink_to(tmp_path / "missing")
    first = package_stamp(str(tmp_path))
    (tmp_path / "a.py").write_text("x = 2\n")
    package_stamp.cache_clear()
    assert package_stamp(str(tmp_path)) != first
