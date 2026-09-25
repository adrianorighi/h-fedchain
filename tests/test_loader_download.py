"""Tests for PTBXLLoader.download() / _extract_zip() — offline only (no network)."""

import zipfile
from pathlib import Path

import pytest

from dataset.loader import PTBXLLoader

WRAPPED = "ptb-xl-a-large-publicly-available-electrocardiography-dataset"


def _uri(path):
    return Path(path).as_uri()


def _make_wrapped_zip(path):
    """Zip with the real physionet layout: single top dir -> 1.0.3/ -> records500/..."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{WRAPPED}/1.0.3/records500/00000/00001/00001.hea", "hdr")
        zf.writestr(f"{WRAPPED}/1.0.3/records500/00000/00001/00001.dat", b"\x00\x01")
        zf.writestr(f"{WRAPPED}/1.0.3/RECORDS", "records500/00000/00001\n")
    return str(path)


def _make_flat_zip(path):
    """Zip with records already at the archive root (no wrapper dir)."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("records500/00000/00001/00001.hea", "hdr")
        zf.writestr("records500/00000/00001/00001.dat", b"\x00\x01")
    return str(path)


def _make_no_hea_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("some/dir/readme.txt", "no records here")
    return str(path)


class TestExtractZip:
    def test_unwraps_nested_root(self, tmp_path):
        zip_path = _make_wrapped_zip(tmp_path / "wrapped.zip")
        cache = tmp_path / "cache"
        loader = PTBXLLoader(cache_dir=str(cache))

        loader._extract_zip(zip_path)

        assert (cache / "records500/00000/00001/00001.hea").exists()
        assert not (cache / WRAPPED).exists()
        assert not (cache / ".extract_tmp").exists()

    def test_flat_layout_no_op_nesting(self, tmp_path):
        zip_path = _make_flat_zip(tmp_path / "flat.zip")
        cache = tmp_path / "cache"
        loader = PTBXLLoader(cache_dir=str(cache))

        loader._extract_zip(zip_path)

        assert (cache / "records500/00000/00001/00001.hea").exists()
        assert not (cache / ".extract_tmp").exists()

    def test_rejects_zip_slip_entry(self, tmp_path):
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("../evil.txt")
            zf.writestr(info, "pwned")
        cache = tmp_path / "cache"
        loader = PTBXLLoader(cache_dir=str(cache))

        with pytest.raises(ValueError, match="unsafe"):
            loader._extract_zip(str(zip_path))

        assert not (cache / "evil.txt").exists()
        assert not (tmp_path / "evil.txt").exists()
        assert not (cache / ".extract_tmp").exists()

    def test_no_hea_raises_runtime_error(self, tmp_path):
        zip_path = _make_no_hea_zip(tmp_path / "norecs.zip")
        cache = tmp_path / "cache"
        loader = PTBXLLoader(cache_dir=str(cache))

        with pytest.raises(RuntimeError, match="extraction produced no wfdb records"):
            loader._extract_zip(zip_path)

        assert not (cache / ".extract_tmp").exists()


class TestDownload:
    def test_download_from_file_url(self, tmp_path, monkeypatch):
        zip_path = _make_wrapped_zip(tmp_path / "wrapped.zip")
        cache = tmp_path / "cache"
        monkeypatch.setattr("dataset.loader.PTBXL_URL", _uri(zip_path))
        loader = PTBXLLoader(cache_dir=str(cache))

        assert loader.download() is True
        assert loader.is_available()
        assert (cache / "records500/00000/00001/00001.hea").exists()
        assert not (cache / ".extract_tmp").exists()
        leftovers = [p.name for p in cache.iterdir() if p.name.endswith((".part", ".zip"))]
        assert leftovers == []

    def test_download_short_circuits_when_available(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "marker.txt").write_text("already here")

        def _no_network(*args, **kwargs):
            raise AssertionError("network must not be touched")

        monkeypatch.setattr("dataset.loader.urllib.request.urlopen", _no_network)
        loader = PTBXLLoader(cache_dir=str(cache))

        assert loader.download() is True

    def test_download_failure_raises_and_cleans_part_file(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist.zip"
        monkeypatch.setattr("dataset.loader.PTBXL_URL", _uri(str(missing)))
        cache = tmp_path / "cache"
        loader = PTBXLLoader(cache_dir=str(cache))

        with pytest.raises(RuntimeError, match="download"):
            loader.download()

        assert not loader.is_available()
        assert not (cache / ".download.zip.part").exists()
