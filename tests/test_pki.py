import pytest
from core.pki import generate_keypair, sign, verify


def test_generate_keypair_returns_sk_vk():
    sk, vk = generate_keypair()
    assert len(sk) == 32
    assert len(vk) == 32
    assert sk != vk


def test_sign_and_verify():
    sk, vk = generate_keypair()
    msg = b"hello, world"
    sig = sign(sk, msg)
    assert verify(vk, msg, sig) is True


def test_verify_rejects_tampered_message():
    sk, vk = generate_keypair()
    msg = b"hello"
    sig = sign(sk, msg)
    assert verify(vk, b"wrong", sig) is False


def test_verify_rejects_tampered_signature():
    sk, vk = generate_keypair()
    msg = b"hello"
    sig = sign(sk, msg)
    bad_sig = b"\x00" * 64
    assert verify(vk, msg, bad_sig) is False


def test_verify_with_wrong_key():
    sk1, _ = generate_keypair()
    _, vk2 = generate_keypair()
    msg = b"hello"
    sig = sign(sk1, msg)
    assert verify(vk2, msg, sig) is False
