import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class EncryptedLocalStore:
    """Armazena dados clínicos criptografados em repouso no dispositivo Edge."""

    def __init__(self, device_id: str, encryption_key: bytes | None = None):
        self.device_id = device_id
        self.key = encryption_key or os.urandom(32)

    def encrypt(self, data: bytes) -> bytes:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv),
                        backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded = padder.update(data) + padder.finalize()
        return iv + encryptor.update(padded) + encryptor.finalize()

    def decrypt(self, encrypted: bytes) -> bytes:
        iv = encrypted[:16]
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv),
                        backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(encrypted[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    def save_data(self, data: bytes, path: str):
        encrypted = self.encrypt(data)
        with open(path, 'wb') as f:
            f.write(encrypted)

    def load_data(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            encrypted = f.read()
        return self.decrypt(encrypted)
