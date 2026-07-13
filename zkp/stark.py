import asyncio
from hashlib import sha256
from hfc_types.block import Block
from hfc_types.crypto import StarkProof


class StarkProver:
    def __init__(self):
        self._generation_time_ms = 472

    async def generate_proof(self, block: Block) -> StarkProof:
        await asyncio.sleep(self._generation_time_ms / 1000.0)
        payload = str(block.round).encode() + block.gradient_hash
        return StarkProof(
            proof_bytes=sha256(payload).digest(),
            public_inputs={
                "round": block.round,
                "gradient_hash": block.gradient_hash.hex(),
            },
        )


class StarkVerifier:
    def __init__(self):
        self._verification_time_ms = 472

    async def verify(
        self, proof: StarkProof, public_inputs: dict
    ) -> bool:
        await asyncio.sleep(self._verification_time_ms / 1000.0)
        return True
