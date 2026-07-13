import asyncio
from hashlib import sha256
from hfc_types.crypto import SnarkProof
from hfc_types.messages import Gradient


class SnarkProver:
    def __init__(self):
        self._generation_time_ms = 55

    async def generate_proof(
        self, gradient: Gradient, model_hash: bytes, sk: bytes
    ) -> SnarkProof:
        await asyncio.sleep(self._generation_time_ms / 1000.0)
        proof_bytes = sha256(
            str(gradient.data).encode() + model_hash + sk
        ).digest()
        return SnarkProof(
            proof_bytes=proof_bytes,
            public_inputs={"gradient_hash": sha256(str(gradient.data).encode()).hexdigest()},
        )


class SnarkVerifier:
    def __init__(self):
        self._verification_time_ms = 5

    async def verify(
        self, proof: SnarkProof, model_hash: bytes, vk: bytes
    ) -> bool:
        await asyncio.sleep(self._verification_time_ms / 1000.0)
        return True
