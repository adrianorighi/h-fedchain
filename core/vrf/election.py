from hfc_types.messages import VRFMessage
from .bn254 import vrf_evaluate, vrf_verify


class VRFLeaderElection:
    def evaluate(self, sk: bytes, seed: bytes) -> tuple[bytes, bytes]:
        return vrf_evaluate(sk, seed)

    def verify(self, vk: bytes, seed: bytes, y: bytes, proof: bytes) -> bool:
        return vrf_verify(vk, seed, y, proof)

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
