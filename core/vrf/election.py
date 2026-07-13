from hashlib import sha256
from hfc_types.messages import VRFMessage


class VRFLeaderElection:
    def evaluate(self, sk: bytes, seed: bytes) -> tuple[bytes, bytes]:
        y = sha256(sk + seed).digest()
        proof = sha256(b"vrf_proof:" + sk + seed).digest()
        return y, proof

    def verify(
        self, vk: bytes, seed: bytes, y: bytes, proof: bytes
    ) -> bool:
        expected_y = sha256(vk + seed).digest()
        expected_proof = sha256(b"vrf_proof:" + vk + seed).digest()
        return y == expected_y and proof == expected_proof

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
