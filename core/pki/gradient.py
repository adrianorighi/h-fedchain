"""Canonical signed message for gradient submissions (fog + edge + simulator)."""
import hashlib
import numpy as np


def gradient_to_f32(data) -> np.ndarray:
    """Coerce gradient data to a float32 ndarray.

    Accepts the legacy pickled-list form and the compact raw-float32-bytes
    wire form; both hash to the same canonical bytes (float32 → tolist →
    float32 round-trips exactly), so signatures verify across forms.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def gradient_data_hash(data) -> bytes:
    return hashlib.sha256(gradient_to_f32(data).tobytes()).digest()


def gradient_signed_message(node_id: str, round_num: int, data) -> bytes:
    return f"{node_id}:{round_num}".encode() + gradient_data_hash(data)
