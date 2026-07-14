from hashlib import sha256
from cryptography.hazmat.primitives.asymmetric import ed25519 as ed25519_key
from cryptography.hazmat.primitives import serialization
from hfc_types.messages import VRFMessage
from core.pki.ed25519 import sign, verify as ed25519_verify


def _derive_vk(sk: bytes) -> bytes:
    private_key = ed25519_key.Ed25519PrivateKey.from_private_bytes(sk)
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


class VRFLeaderElection:
    def evaluate(self, sk: bytes, seed: bytes) -> tuple[bytes, bytes]:
        vk = _derive_vk(sk)
        y = sha256(vk + seed).digest()
        proof = sign(sk, y + seed)
        return y, proof

    def verify(self, vk: bytes, seed: bytes, y: bytes, proof: bytes) -> bool:
        expected_y = sha256(vk + seed).digest()
        if y != expected_y:
            return False
        return ed25519_verify(vk, y + seed, proof)

    def elect(
        self,
        candidates: list[VRFMessage],
        seed: bytes,
        vk_map: dict[str, bytes],
    ) -> str:
        if not candidates:
            raise ValueError("No valid candidates")
        valid: list[VRFMessage] = []
        for c in candidates:
            vk = vk_map.get(c.node_id)
            if vk is None:
                continue
            if self.verify(vk, seed, c.y, c.proof):
                valid.append(c)
        if not valid:
            raise ValueError("No valid candidates")
        return min(valid, key=lambda c: c.y).node_id
