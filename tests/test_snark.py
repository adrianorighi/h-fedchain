import pytest
import numpy as np
from hfc_types.messages import Gradient
from zkp.snark import SnarkProver, SnarkVerifier


class TestSnarkProverVerifier:
    @pytest.mark.asyncio
    async def test_prove_and_verify(self):
        gradient = Gradient(node_id="d0", round=1, data=[1.0, 2.0, 3.0])
        prover = SnarkProver()
        verifier = SnarkVerifier()
        proof = await prover.generate_proof(gradient, b"model_hash", b"sk_123")
        result = await verifier.verify(proof, b"model_hash", b"vk")
        assert result is True

    @pytest.mark.asyncio
    async def test_prove_returns_proof_with_fields(self):
        gradient = Gradient(node_id="d1", round=2, data=[0.5, -1.0])
        prover = SnarkProver()
        proof = await prover.generate_proof(gradient, b"m_hash", b"secret")
        assert isinstance(proof.proof_bytes, bytes)
        assert "gradient_hash" in proof.public_inputs
        assert len(proof.proof_bytes) > 0

    @pytest.mark.asyncio
    async def test_verify_rejects_junk_proof(self):
        from hfc_types.crypto import SnarkProof

        verifier = SnarkVerifier()
        bad_proof = SnarkProof(proof_bytes=b"invalid json", public_inputs={})
        result = await verifier.verify(bad_proof, b"model_hash", b"vk")
        assert result is False
