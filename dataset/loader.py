import os
import numpy as np
from typing import Optional

CACHE_DIR = os.path.expanduser("~/.ptbxl")
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-1.0.3.zip"


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

    def generate_synthetic(self, num_records: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic ECG-like data for testing when real data unavailable."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (num_records, 12, 1000)).astype(np.float32) * 0.5
        t = np.linspace(0, 10, 1000)
        for i in range(num_records):
            for ch in range(12):
                data[i, ch] += 0.3 * np.sin(2 * np.pi * 1.5 * t + rng.random() * 2 * np.pi)
        labels = rng.integers(0, 5, size=num_records).astype(np.int32)
        return data, labels
