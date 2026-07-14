from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def generate_keypair() -> tuple[bytes, bytes]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    sk = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    vk = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return sk, vk


def sign(sk: bytes, message: bytes) -> bytes:
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(sk)
    return private_key.sign(message)


def verify(vk: bytes, message: bytes, signature: bytes) -> bool:
    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(vk)
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False
