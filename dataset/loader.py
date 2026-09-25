import os
import shutil
import urllib.request
import zipfile
from pathlib import PurePosixPath
from typing import Optional

import numpy as np

CACHE_DIR = os.path.expanduser("~/.ptbxl")
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-1.0.3.zip"

CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60
MAX_WRAPPER_DEPTH = 5


class PTBXLLoader:
    """Loads PTB-XL ECG dataset. Caches locally after first download."""

    def __init__(self, cache_dir: str = CACHE_DIR, max_records: Optional[int] = None):
        self.cache_dir = cache_dir
        self.max_records = max_records
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._available = False

    def is_available(self) -> bool:
        """Check if PTB-XL is cached and ready."""
        return os.path.isdir(self.cache_dir) and bool(os.listdir(self.cache_dir))

    def download(self, force: bool = False) -> bool:
        """Download the PTB-XL zip and extract it into ``cache_dir``.

        The zip is streamed to ``<cache_dir>/.download.zip.part`` and only
        extracted once complete, so an interrupted transfer can never leave a
        half-extracted layout that fools :meth:`is_available`.

        Returns:
            True when the dataset is present (already cached, or downloaded).

        Raises:
            RuntimeError: wrapping the original error if download or
                extraction fails. Re-raising (instead of returning False)
                keeps the failure visible at the call site — a silent False
                would let scripts continue and later fail inside ``load()``
                with a misleading "not downloaded" message.
        """
        if self.is_available() and not force:
            return True

        os.makedirs(self.cache_dir, exist_ok=True)
        part_path = os.path.join(self.cache_dir, ".download.zip.part")
        try:
            request = urllib.request.Request(
                PTBXL_URL, headers={"User-Agent": "h-fedchain/ptbxl-loader"}
            )
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                with open(part_path, "wb") as out:
                    shutil.copyfileobj(response, out, length=CHUNK_SIZE)
            self._extract_zip(part_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download PTB-XL dataset: {exc}") from exc
        finally:
            if os.path.exists(part_path):
                os.remove(part_path)
        return True

    def _extract_zip(self, zip_path: str) -> None:
        """Extract ``zip_path`` into ``cache_dir``, unwrapping the physionet root.

        Extraction happens in ``<cache_dir>/.extract_tmp`` so a failed or
        malicious archive never poisons :meth:`is_available`.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        extract_dir = os.path.join(self.cache_dir, ".extract_tmp")
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                self._validate_zip_entries(zf)
                zf.extractall(extract_dir)

            root = self._locate_dataset_root(extract_dir)
            if not self._contains_hea(root):
                raise RuntimeError("extraction produced no wfdb records")

            self._move_children(root, self.cache_dir)

            if not self._contains_hea(self.cache_dir):
                raise RuntimeError("extraction produced no wfdb records")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    @staticmethod
    def _validate_zip_entries(zf: zipfile.ZipFile) -> None:
        for info in zf.infolist():
            name = info.filename
            entry = PurePosixPath(name)
            if entry.is_absolute() or ".." in entry.parts or os.path.isabs(name):
                raise ValueError(f"unsafe zip entry rejected: {name!r}")

    @staticmethod
    def _locate_dataset_root(extract_dir: str) -> str:
        """Descend single-directory wrapper levels to the directory holding records."""
        current = extract_dir
        for _ in range(MAX_WRAPPER_DEPTH):
            entries = os.listdir(current)
            if any(name.endswith(".hea") for name in entries):
                break
            dirs = [e for e in entries if os.path.isdir(os.path.join(current, e))]
            if (
                len(entries) == 1
                and len(dirs) == 1
                and not dirs[0].startswith("records")
            ):
                current = os.path.join(current, dirs[0])
                continue
            break
        return current

    @staticmethod
    def _contains_hea(path: str) -> bool:
        for _dirpath, _dirnames, filenames in os.walk(path):
            if any(name.endswith(".hea") for name in filenames):
                return True
        return False

    def _move_children(self, src: str, dst: str) -> None:
        for name in os.listdir(src):
            source = os.path.join(src, name)
            target = os.path.join(dst, name)
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)
            shutil.move(source, target)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load PTB-XL data.

        Returns:
            data: numpy array of shape (N, 12, 1000) — ECG signals
            labels: numpy array of shape (N,) — integer class labels

        Raises:
            FileNotFoundError: if dataset not downloaded
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"PTB-XL not found at {self.cache_dir}. "
                "Run download() first."
            )
        if self._data is not None:
            return self._data, self._labels

        import wfdb

        db = wfdb.get_dbs()
        records = wfdb.io.get_record_list("ptb-xl")
        if self.max_records:
            records = records[: self.max_records]

        signals_list = []
        labels_list = []

        for rec_name in records:
            try:
                record = wfdb.rdrecord(os.path.join(self.cache_dir, rec_name))
                sig = record.p_signal
                target_len = 1000
                if sig.shape[0] != target_len:
                    x_old = np.linspace(0, 1, sig.shape[0])
                    x_new = np.linspace(0, 1, target_len)
                    sig_resampled = np.zeros((target_len, 12))
                    for ch in range(12):
                        sig_resampled[:, ch] = np.interp(x_new, x_old, sig[:, ch])
                    sig = sig_resampled
                signals_list.append(sig.T)
                labels_list.append(0)
            except Exception:
                continue

        self._data = np.array(signals_list, dtype=np.float32)
        self._labels = np.array(labels_list, dtype=np.int32)
        return self._data, self._labels

    def generate_synthetic(
        self, num_records: int = 100, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic ECG-like data for testing when real data unavailable."""
        rng = np.random.default_rng(42 if seed is None else seed)
        data = rng.normal(0, 1, (num_records, 12, 1000)).astype(np.float32) * 0.5
        t = np.linspace(0, 10, 1000)
        for i in range(num_records):
            for ch in range(12):
                data[i, ch] += 0.3 * np.sin(2 * np.pi * 1.5 * t + rng.random() * 2 * np.pi)
        labels = rng.integers(0, 5, size=num_records).astype(np.int32)
        return data, labels
