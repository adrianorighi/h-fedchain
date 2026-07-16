from hashlib import sha256

from py_ecc.bn128 import (
    FQ,
    b,
    curve_order,
    field_modulus,
    is_on_curve,
    multiply,
)

from core.pki.ed25519 import sign, verify as ed25519_verify

HASH_TO_CURVE_ITERATIONS = 256
FIELD_MODULUS = field_modulus
BN128_B = int(b)
SERIALIZED_LEN = 64


def _serialize_point(pt: tuple[FQ, FQ]) -> bytes:
    return int(pt[0]).to_bytes(32, "big") + int(pt[1]).to_bytes(32, "big")


def _deserialize_point(data: bytes) -> tuple[FQ, FQ]:
    assert len(data) == SERIALIZED_LEN
    x = int.from_bytes(data[:32], "big")
    y = int.from_bytes(data[32:], "big")
    return (FQ(x), FQ(y))


def hash_to_point(message: bytes) -> tuple[FQ, FQ]:
    seed = int.from_bytes(sha256(message).digest(), "big")
    x = FQ(seed % FIELD_MODULUS)
    for _ in range(HASH_TO_CURVE_ITERATIONS):
        ix = int(x)
        y_sq = (pow(ix, 3, FIELD_MODULUS) + BN128_B) % FIELD_MODULUS
        if pow(y_sq, (FIELD_MODULUS - 1) // 2, FIELD_MODULUS) == 1:
            y = FQ(pow(y_sq, (FIELD_MODULUS + 1) // 4, FIELD_MODULUS))
            pt = (x, y)
            if is_on_curve(pt, b):
                return pt
        x = FQ((ix + 1) % FIELD_MODULUS)
    raise ValueError("Could not hash to curve — exceeded iterations")


def _derive_bn254_scalar(sk_ed25519: bytes) -> int:
    return int.from_bytes(sha256(sk_ed25519).digest(), "big") % curve_order


def vrf_evaluate(sk_ed25519: bytes, alpha: bytes) -> tuple[bytes, bytes]:
    scalar = _derive_bn254_scalar(sk_ed25519)
    h = hash_to_point(alpha)
    gamma = multiply(h, scalar)
    gamma_bytes = _serialize_point(gamma)
    proof = sign(sk_ed25519, gamma_bytes + alpha)
    return gamma_bytes, proof


def vrf_verify(
    vk_ed25519: bytes, alpha: bytes, gamma_bytes: bytes, proof: bytes
) -> bool:
    gamma = _deserialize_point(gamma_bytes)
    if not is_on_curve(gamma, b):
        return False
    return ed25519_verify(vk_ed25519, gamma_bytes + alpha, proof)
