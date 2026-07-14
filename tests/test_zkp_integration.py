import pytest
from dataclasses import replace
from simulator.fog_node import FogNode
from simulator.orchestrator import Orchestrator
from hfc_types.block import Block, LedgerEntry


@pytest.mark.asyncio
async def test_stark_proof_generated_for_stark_variant():
    orch = Orchestrator(
        nodes_per_cluster=4,
        devices_per_cluster=4,
        f=1,
        latency_ms=1.0,
        variant="stark",
    )
    orch.setup()
    result = await orch.run_round(round_num=1)
    assert result["qc_emitted"] is True
    block = orch.nodes[0].ledger._entries[-1].block
    if orch.variant in ("stark", "full"):
        assert block.stark_proof is not None


@pytest.mark.asyncio
async def test_no_stark_proof_for_no_zkp_variant():
    orch = Orchestrator(
        nodes_per_cluster=4,
        devices_per_cluster=4,
        f=1,
        latency_ms=1.0,
        variant="no_zkp",
    )
    orch.setup()
    result = await orch.run_round(round_num=1)
    assert result["qc_emitted"] is True
    block = orch.nodes[0].ledger._entries[-1].block
    assert block.stark_proof is None


@pytest.mark.asyncio
async def test_no_stark_proof_for_snark_variant():
    orch = Orchestrator(
        nodes_per_cluster=4,
        devices_per_cluster=4,
        f=1,
        latency_ms=1.0,
        variant="snark",
    )
    orch.setup()
    result = await orch.run_round(round_num=1)
    block = orch.nodes[0].ledger._entries[-1].block
    assert block.stark_proof is None


@pytest.mark.asyncio
async def test_fog_node_verify_block_valid_stark():
    from zkp.stark import StarkProver, StarkVerifier
    from core.pki import generate_keypair
    node = FogNode(
        node_id="n0",
        sk=b"\x01" * 32,
        vk=b"\x02" * 32,
        peers=["n1", "n2", "n3"],
        n=4, f=1,
        network=None,
    )
    node.network = None
    node.set_variant("stark")
    block = Block(
        round=5,
        gradient_hash=b"gh",
        qc_commit=None,
        stark_proof=None,
        accepted_devices=["dev1"],
        rejected_devices=[],
        timestamp=100.0,
        prev_hash=b"\x00" * 32,
    )
    prover = StarkProver()
    proof = await prover.generate_proof(block)
    block_with_proof = replace(block, stark_proof=proof)
    valid = await node.verify_block(block_with_proof)
    assert valid is True


@pytest.mark.asyncio
async def test_fog_node_verify_block_no_proof():
    node = FogNode(
        node_id="n0",
        sk=b"\x01" * 32,
        vk=b"\x02" * 32,
        peers=["n1", "n2", "n3"],
        n=4, f=1,
        network=None,
    )
    node.network = None
    node.set_variant("stark")
    block = Block(
        round=5,
        gradient_hash=b"gh",
        qc_commit=None,
        stark_proof=None,
        accepted_devices=["dev1"],
        rejected_devices=[],
        timestamp=100.0,
        prev_hash=b"\x00" * 32,
    )
    valid = await node.verify_block(block)
    assert valid is True  # No proof to verify
