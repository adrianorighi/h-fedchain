from dataset.encrypted_store import EncryptedLocalStore
import os


def test_encrypt_decrypt_roundtrip():
    store = EncryptedLocalStore("d0", b"k" * 32)
    original = b"clinical_data_12345"
    encrypted = store.encrypt(original)
    assert encrypted != original
    decrypted = store.decrypt(encrypted)
    assert decrypted == original


def test_different_keys_produce_different_ciphertext():
    store1 = EncryptedLocalStore("d0", b"k" * 32)
    store2 = EncryptedLocalStore("d0", b"j" * 32)
    ct1 = store1.encrypt(b"same_data")
    ct2 = store2.encrypt(b"same_data")
    assert ct1 != ct2


def test_save_load_file(tmp_path):
    store = EncryptedLocalStore("d0", b"k" * 32)
    path = tmp_path / "test_data.bin"
    store.save_data(b"patient_data", str(path))
    assert path.exists()
    loaded = store.load_data(str(path))
    assert loaded == b"patient_data"


def test_load_nonexistent_raises(tmp_path):
    from dataset.edge_worker import EdgeWorker
    import numpy as np
    worker = EdgeWorker(device_id="d0", indices=[0, 1],
                       all_data=np.zeros((2, 12000)),
                       all_labels=np.zeros(2, dtype=int),
                       use_encryption=True)
    import pytest
    with pytest.raises(FileNotFoundError):
        worker.load_local_data(str(tmp_path / "nonexistent.bin"))
