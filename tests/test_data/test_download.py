#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for ``brutus.data.download._fetch`` symlink handling.

These use only tiny temporary files (no network, no large grids): the Pooch
registry's ``fetch`` is monkeypatched to return a local stand-in file.
"""

import pathlib

from brutus.data import download


def _fake_cache_file(tmp_path):
    """Create a stand-in "downloaded" cache file and return its path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    real = cache / "grid_mist_v9.h5"
    real.write_text("data")
    return real


def test_fetch_replaces_broken_symlink(tmp_path, monkeypatch):
    """A stale/broken symlink at the target path must not crash _fetch.

    Path.exists() follows symlinks and reports False for a dangling link, so
    the old ``if not target_path.exists(): symlink_to(...)`` raised
    FileExistsError because the link path itself already existed.
    """
    monkeypatch.delenv("CI", raising=False)
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(
        download.strato, "fetch", lambda name, progressbar=True: str(real)
    )

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    # Pre-create a broken (dangling) symlink at the target path.
    dangling = tmp_path / "gone.h5"
    dangling.write_text("x")
    link = target_dir / "grid_mist_v9.h5"
    link.symlink_to(dangling)
    dangling.unlink()  # the link now dangles

    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).read_text() == "data"


def test_fetch_falls_back_to_copy_without_symlink_support(tmp_path, monkeypatch):
    """If symlink creation fails, _fetch must copy instead of raising.

    Reproduces filesystems without symlink support (some Windows / network /
    container mounts) by making Path.symlink_to raise OSError.
    """
    monkeypatch.delenv("CI", raising=False)
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(
        download.strato, "fetch", lambda name, progressbar=True: str(real)
    )

    def _no_symlink(*args, **kwargs):
        raise OSError(1, "operation not permitted")

    monkeypatch.setattr(pathlib.Path, "symlink_to", _no_symlink)

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()

    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).read_text() == "data"


def test_fetch_existing_valid_symlink_is_noop(tmp_path, monkeypatch):
    """A pre-existing valid symlink must be left intact and resolve correctly."""
    monkeypatch.delenv("CI", raising=False)
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(
        download.strato, "fetch", lambda name, progressbar=True: str(real)
    )

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    link = target_dir / "grid_mist_v9.h5"
    link.symlink_to(real)

    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).is_symlink()
    assert pathlib.Path(out).read_text() == "data"
