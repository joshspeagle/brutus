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


def test_fetch_replaces_truncated_target_copy(tmp_path, monkeypatch):
    """A pre-existing regular file whose size differs from the verified cache
    copy (e.g. a truncated leftover from an interrupted copy) must be
    replaced, not returned as-is forever."""
    monkeypatch.delenv("CI", raising=False)
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(
        download.strato, "fetch", lambda name, progressbar=True: str(real)
    )

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    truncated = target_dir / "grid_mist_v9.h5"
    truncated.write_text("da")  # partial copy of "data"

    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).read_text() == "data"


def test_fetch_interrupted_copy_leaves_no_partial_target(tmp_path, monkeypatch):
    """When the copy fallback is interrupted, no partial file may remain at
    the final target path (a leftover truncated file used to permanently
    short-circuit every future fetch)."""
    import shutil

    monkeypatch.delenv("CI", raising=False)
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(
        download.strato, "fetch", lambda name, progressbar=True: str(real)
    )

    def _no_symlink(*args, **kwargs):
        raise OSError(1, "operation not permitted")

    monkeypatch.setattr(pathlib.Path, "symlink_to", _no_symlink)

    real_copy2 = shutil.copy2

    def _interrupted_copy2(src, dst, **kwargs):
        # Write a partial file at the destination, then die mid-copy.
        pathlib.Path(dst).write_text("da")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copy2", _interrupted_copy2)

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    target = target_dir / "grid_mist_v9.h5"

    try:
        download._fetch("grid_mist_v9.h5", str(target_dir))
    except KeyboardInterrupt:
        pass
    assert not target.exists(), "partial copy left at final target path"

    # A subsequent fetch with a working copy must produce the full file.
    monkeypatch.setattr(shutil, "copy2", real_copy2)
    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).read_text() == "data"


def test_ci_env_does_not_skip_hash_verification(tmp_path, monkeypatch):
    """CI=true (ambient in GitHub Actions/GitLab/etc.) must NOT bypass
    pooch's SHA verification; only the explicit brutus opt-in may."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("BRUTUS_SKIP_HASH_CHECK", raising=False)
    real = _fake_cache_file(tmp_path)

    # Put a (corrupt) file in the pooch cache location; if the CI shortcut
    # were still active, _fetch would return it without calling fetch().
    monkeypatch.setattr(download.strato, "path", real.parent)
    calls = []

    def _fetch_spy(name, progressbar=True):
        calls.append(name)
        return str(real)

    monkeypatch.setattr(download.strato, "fetch", _fetch_spy)

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    download._fetch("grid_mist_v9.h5", str(target_dir))
    assert calls == ["grid_mist_v9.h5"], "pooch fetch (hash check) was bypassed"


def test_brutus_skip_hash_check_optin(tmp_path, monkeypatch):
    """BRUTUS_SKIP_HASH_CHECK=1 uses an existing cache file without fetching."""
    monkeypatch.setenv("BRUTUS_SKIP_HASH_CHECK", "1")
    real = _fake_cache_file(tmp_path)
    monkeypatch.setattr(download.strato, "path", real.parent)

    def _fetch_boom(name, progressbar=True):
        raise AssertionError("fetch should not be called for cached file")

    monkeypatch.setattr(download.strato, "fetch", _fetch_boom)

    target_dir = tmp_path / "DATAFILES"
    target_dir.mkdir()
    out = download._fetch("grid_mist_v9.h5", str(target_dir))
    assert pathlib.Path(out).read_text() == "data"
