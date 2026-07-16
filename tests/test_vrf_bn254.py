import pytest
from py_ecc.bn128 import b, is_on_curve

from core.pki.ed25519 import generate_keypair
from core.vrf.bn254 import (
    hash_to_point,
    vrf_evaluate,
    vrf_verify,
    _serialize_point,
    _deserialize_point,
)


def test_vrf_hash_to_point():
    pt = hash_to_point(b"test_seed")
    assert is_on_curve(pt, b) is True


def test_vrf_deterministic():
    sk, _ = generate_keypair()
    alpha = b"round_42"
    g1, p1 = vrf_evaluate(sk, alpha)
    g2, p2 = vrf_evaluate(sk, alpha)
    assert g1 == g2
    assert p1 == p2


def test_vrf_different_seeds():
    sk, _ = generate_keypair()
    g1, _ = vrf_evaluate(sk, b"seed_a")
    g2, _ = vrf_evaluate(sk, b"seed_b")
    assert g1 != g2


def test_vrf_verify_accepts_valid():
    sk, vk = generate_keypair()
    alpha = b"round_42"
    gamma, proof = vrf_evaluate(sk, alpha)
    assert vrf_verify(vk, alpha, gamma, proof) is True


def test_vrf_verify_rejects_bad_proof():
    sk, vk = generate_keypair()
    alpha = b"round_42"
    gamma, _ = vrf_evaluate(sk, alpha)
    bad_proof = b"tampered"
    assert vrf_verify(vk, alpha, gamma, bad_proof) is False


def test_vrf_verify_rejects_wrong_key():
    sk1, _ = generate_keypair()
    _, vk2 = generate_keypair()
    alpha = b"round_42"
    gamma, proof = vrf_evaluate(sk1, alpha)
    assert vrf_verify(vk2, alpha, gamma, proof) is False


def test_vrf_verify_rejects_wrong_alpha():
    sk, vk = generate_keypair()
    gamma, proof = vrf_evaluate(sk, b"round_42")
    assert vrf_verify(vk, b"round_43", gamma, proof) is False


def test_serialize_roundtrip():
    pt = hash_to_point(b"round_42")
    data = _serialize_point(pt)
    assert len(data) == 64
    restored = _deserialize_point(data)
    assert int(restored[0]) == int(pt[0])
    assert int(restored[1]) == int(pt[1])
    assert is_on_curve(restored, b) is True
