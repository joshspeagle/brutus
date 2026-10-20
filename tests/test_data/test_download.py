"""Tests of the fetch functions in brutus.data.download."""

import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

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


class TestDataDownloadFunctions:
    """Unit tests for data downloading functions."""

    @patch("brutus.data.download._fetch")
    def test_fetch_isos_basic(self, mock_fetch):
        """Test basic fetch_isos functionality."""
        from brutus.data.download import fetch_isos

        # Mock the _fetch function
        mock_fetch.return_value = Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")

        # Test default parameters
        result = fetch_isos()

        mock_fetch.assert_called_once_with("MIST_1.2_iso_vvcrit0.0.h5", ".")
        assert result == Path("/fake/path/MIST_1.2_iso_vvcrit0.0.h5")

    @patch("brutus.data.download._fetch")
    def test_fetch_isos_rotating(self, mock_fetch):
        """Test fetch_isos with rotating models."""
        from brutus.data.download import fetch_isos

        mock_fetch.return_value = Path("/fake/path/MIST_1.2_iso_vvcrit0.4.h5")

        result = fetch_isos(target_dir="/tmp", iso="MIST_1.2_vvcrit0.4")

        mock_fetch.assert_called_once_with("MIST_1.2_iso_vvcrit0.4.h5", "/tmp")
        assert result == Path("/fake/path/MIST_1.2_iso_vvcrit0.4.h5")

    def test_fetch_isos_invalid(self):
        """Test fetch_isos with invalid iso parameter."""
        from brutus.data.download import fetch_isos

        with pytest.raises(
            ValueError, match="The specified isochrone file does not exist"
        ):
            fetch_isos(iso="invalid_iso")

    @patch("brutus.data.download._fetch")
    def test_fetch_tracks_basic(self, mock_fetch):
        """Test basic fetch_tracks functionality."""
        from brutus.data.download import fetch_tracks

        mock_fetch.return_value = Path("/fake/path/MIST_1.2_EEPtrk.h5")

        result = fetch_tracks(target_dir="/data")

        mock_fetch.assert_called_once_with("MIST_1.2_EEPtrk.h5", "/data")
        assert result == Path("/fake/path/MIST_1.2_EEPtrk.h5")

    def test_fetch_tracks_invalid(self):
        """Test fetch_tracks with invalid track parameter."""
        from brutus.data.download import fetch_tracks

        with pytest.raises(ValueError, match="The specified track file does not exist"):
            fetch_tracks(track="invalid_track")

    @patch("brutus.data.download._fetch")
    def test_fetch_dustmaps_basic(self, mock_fetch):
        """Test basic fetch_dustmaps functionality."""
        from brutus.data.download import fetch_dustmaps

        mock_fetch.return_value = Path("/fake/path/bayestar2019_v1.h5")

        result = fetch_dustmaps()

        mock_fetch.assert_called_once_with("bayestar2019_v1.h5", ".")

    def test_fetch_dustmaps_invalid(self):
        """Test fetch_dustmaps with invalid dustmap parameter."""
        from brutus.data.download import fetch_dustmaps

        with pytest.raises(
            ValueError, match="The specified dustmap file does not exist"
        ):
            fetch_dustmaps(dustmap="invalid_dustmap")

    @patch("brutus.data.download._fetch")
    def test_fetch_grids_versions(self, mock_fetch):
        """Test fetch_grids with different versions."""
        from brutus.data.download import fetch_grids

        # Test v9 (default)
        mock_fetch.return_value = Path("/fake/path/grid_mist_v9.h5")
        fetch_grids(grid="mist_v9")
        mock_fetch.assert_called_with("grid_mist_v9.h5", ".")

        # Test v8
        mock_fetch.return_value = Path("/fake/path/grid_mist_v8.h5")
        fetch_grids(grid="mist_v8")
        mock_fetch.assert_called_with("grid_mist_v8.h5", ".")

        # Test bayestar
        mock_fetch.return_value = Path("/fake/path/grid_bayestar_v5.h5")
        fetch_grids(grid="bayestar_v5")
        mock_fetch.assert_called_with("grid_bayestar_v5.h5", ".")

    def test_fetch_grids_invalid(self):
        """Test fetch_grids with invalid grid parameter."""
        from brutus.data.download import fetch_grids

        with pytest.raises(ValueError, match="The specified grid file does not exist"):
            fetch_grids(grid="invalid_grid")

    @patch("brutus.data.download._fetch")
    def test_fetch_offsets_versions(self, mock_fetch):
        """Test fetch_offsets with different versions."""
        from brutus.data.download import fetch_offsets

        # Test v9 (default)
        mock_fetch.return_value = Path("/fake/path/offsets_mist_v9.txt")
        fetch_offsets(grid="mist_v9")
        mock_fetch.assert_called_with("offsets_mist_v9.txt", ".")

        # Test bayestar (note: uses bs_v9 filename)
        mock_fetch.return_value = Path("/fake/path/offsets_bs_v9.txt")
        fetch_offsets(grid="bayestar_v5")
        mock_fetch.assert_called_with("offsets_bs_v9.txt", ".")

    @patch("brutus.data.download._fetch")
    def test_fetch_nns_basic(self, mock_fetch):
        """Test basic fetch_nns functionality."""
        from brutus.data.download import fetch_nns

        mock_fetch.return_value = Path("/fake/path/nn_c3k.h5")

        result = fetch_nns()

        mock_fetch.assert_called_once_with("nn_c3k.h5", ".")

    def test_fetch_nns_invalid(self):
        """Test fetch_nns with invalid model parameter."""
        from brutus.data.download import fetch_nns

        with pytest.raises(
            ValueError, match="The specified neural network file does not exist"
        ):
            fetch_nns(model="invalid_model")


class TestDataIntegration:
    """Integration tests for data workflows."""

    @patch("brutus.data.download._fetch")
    def test_typical_data_workflow(self, mock_fetch):
        """Test a typical data download and load workflow."""
        # This test simulates a typical user workflow without actual downloads

        # Mock download functions
        mock_fetch.side_effect = [
            Path("/fake/grid_mist_v9.h5"),
            Path("/fake/offsets_mist_v9.txt"),
        ]

        from brutus.data import fetch_grids, fetch_offsets

        # Download model grid
        grid_path = fetch_grids(target_dir="/data", grid="mist_v9")
        assert grid_path == Path("/fake/grid_mist_v9.h5")

        # Download offsets
        offset_path = fetch_offsets(target_dir="/data", grid="mist_v9")
        assert offset_path == Path("/fake/offsets_mist_v9.txt")

        # Verify expected calls
        assert mock_fetch.call_count == 2

    def test_documentation_examples(self):
        """Test that documentation examples work (with mocking)."""
        with patch("brutus.data.download._fetch") as mock_fetch:
            mock_fetch.return_value = Path("/fake/MIST_1.2_iso_vvcrit0.0.h5")

            # Example from fetch_isos docstring
            from brutus.data import fetch_isos

            iso_path = fetch_isos(target_dir="./data/")

            assert iso_path == Path("/fake/MIST_1.2_iso_vvcrit0.0.h5")
