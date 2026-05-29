#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the top-level ``brutus`` package (``brutus/__init__.py``):
version consistency, the public API surface, and documentation.
"""

from importlib.metadata import version


def test_version_matches_package_metadata():
    """``brutus.__version__`` is a string and matches the installed metadata.

    Asserting against ``importlib.metadata`` (rather than a hardcoded string)
    both avoids breaking on every release and meaningfully catches a desync
    between ``pyproject.toml`` and ``brutus/__init__.py``.
    """
    import brutus

    assert isinstance(brutus.__version__, str)
    assert brutus.__version__ == version("astro-brutus")


def test_key_public_api_available():
    """The advertised public API is importable from the top level."""
    import brutus

    for name in (
        "Isochrone",
        "EEPTracks",
        "fetch_grids",
        "load_models",
        "magnitude",
        "inv_magnitude",
    ):
        assert hasattr(brutus, name), f"missing public symbol: {name}"


def test_all_is_well_formed():
    """``__all__`` is a list containing the key public symbols."""
    import brutus

    assert isinstance(brutus.__all__, list)
    for name in (
        "__version__",
        "Isochrone",
        "EEPTracks",
        "fetch_grids",
        "fetch_isos",
        "load_models",
        "magnitude",
        "inv_magnitude",
    ):
        assert name in brutus.__all__


def test_module_docstring():
    """The package exposes a usable module docstring."""
    import brutus

    assert brutus.__doc__ is not None
    assert "brutus" in brutus.__doc__.lower()


def test_subpackage_imports():
    """The refactored subpackages expose their key callables."""
    from brutus.core import EEPTracks, Isochrone
    from brutus.data import fetch_grids, load_models
    from brutus.utils import magnitude

    for obj in (Isochrone, EEPTracks, fetch_grids, load_models, magnitude):
        assert callable(obj)
