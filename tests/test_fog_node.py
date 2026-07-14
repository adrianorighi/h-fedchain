import pytest
import numpy as np
from simulator.fog_node import FogNode
from simulator.network import EmulatedNetwork
from hfc_types.messages import Gradient, GradientWithProof
from hfc_types.crypto import SnarkProof
from core.pki import generate_keypair
from zkp.snark import SnarkProver, SnarkVerifier


@pytest.fixture
def fog_node():
    net = EmulatedNetwork(latency_ms=1.0)
    net.add_node("n0")
    return FogNode(
        node_id="n0",
        sk=b"test_sk",
        vk=b"test_vk",
        peers=["n0"],
        n=1,
        f=0,
        network=net,
    )


@pytest.mark.asyncio
async def test_fog_node_rejects_invalid_snark(fog_node):
    fog_node.set_variant("snark")
    model_hash = b"test_model_hash"
    bad_proof = SnarkProof(proof_bytes=b"junk", public_inputs={})
    grads = [
        GradientWithProof(
            gradient=Gradient(node_id="d0", round=1, data=[1.0, 2.0]),
            snark_proof=bad_proof,
        )
    ]
    result = await fog_node.process_round(grads, b"seed", 1, model_hash=model_hash)
    assert result is None or len(result.accepted_devices) == 0


@pytest.mark.asyncio
async def test_fog_node_accepts_valid_gradient(fog_node):
    fog_node.set_variant("snark")
    sk, vk = generate_keypair()
    fog_node._vk_map["d0"] = vk
    model_hash = b"test_model_hash"
    grad = Gradient(node_id="d0", round=1, data=[1.0, 2.0])
    prover = SnarkProver()
    proof = await prover.generate_proof(grad, model_hash, sk)
    grads = [GradientWithProof(gradient=grad, snark_proof=proof)]
    result = await fog_node.process_round(grads, b"seed", 1, model_hash=model_hash)
    assert result is not None
    assert "d0" in result.accepted_devices


@pytest.mark.asyncio
async def test_fog_node_tracks_adversarial_count(fog_node):
    fog_node.set_variant("no_zkp")
    grads = [
        Gradient(node_id="d0", round=1, data=[1.0]),
        Gradient(node_id="adv_d1", round=1, data=[100.0]),
        Gradient(node_id="adv_d2", round=1, data=[200.0]),
    ]
    fog_node.n = 1
    fog_node.f = 0
    result = await fog_node.process_round(grads, b"seed", 1)
    assert result is not None
    assert result.total_adversarial == 2
    assert result.rejected_adversarial == 2
    assert result.rejected_honest == 0


@pytest.mark.asyncio
async def test_fog_node_without_snark_accepts_all(fog_node):
    fog_node.set_variant("no_zkp")
    grads = [
        Gradient(node_id="d0", round=1, data=[1.0]),
        Gradient(node_id="d1", round=1, data=[2.0]),
    ]
    fog_node.n = 2
    fog_node.f = 0
    result = await fog_node.process_round(grads, b"seed", 1)
    assert result is not None
    assert len(result.accepted_devices) == 2
