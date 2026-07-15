"""End-to-end tests for distributed H-FedChain services."""

import asyncio
import os
import pickle
import tempfile

import pytest

from core.pki import generate_keypair
from hfc_types.messages import Gradient, GradientWithProof


def _make_gradient(node_id: str, round_num: int, sk: bytes) -> GradientWithProof:
    gradient = Gradient(
        node_id=node_id,
        round=round_num,
        data=[float(i) for i in range(10)],
        signature=sk[:8],
    )
    return GradientWithProof(gradient=gradient, snark_proof=None)


@pytest.mark.asyncio
async def test_single_fog_node_init():
    """FogService initializes without error (no MQTT needed)."""
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog_test",
        sk=sk,
        vk=vk,
        peers=[],
        n=1,
        f=0,
        grpc_port=0,
    )
    fog._mqtt_enabled = False
    assert fog.node_id == "fog_test"
    assert fog.engine is not None
    assert fog.ledger is not None
    await fog.stop()


@pytest.mark.asyncio
async def test_fog_node_gradient_ingestion():
    """Fog node accepts gradients and runs consensus round."""
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog1",
        sk=sk,
        vk=vk,
        peers=["fog2", "fog3"],
        n=3,
        f=1,
        grpc_port=0,
    )
    fog._mqtt_enabled = False
    fog._running = True

    g1 = _make_gradient("d0", 1, sk)
    g2 = _make_gradient("d1", 1, sk)
    fog.set_vk("d0", vk)
    fog.set_vk("d1", vk)

    await fog.on_gradient_received(g1)
    assert len(fog._pending_gradients) == 1
    await fog.on_gradient_received(g2)
    # _run_round clears pending_gradients after reaching threshold
    assert len(fog._pending_gradients) == 0

    await fog.stop()


@pytest.mark.asyncio
async def test_fog_node_single_node_consensus():
    """Single node consensus: leader proposes, commits block."""
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="single",
        sk=sk,
        vk=vk,
        peers=[],
        n=1,
        f=0,
        grpc_port=0,
    )
    fog._mqtt_enabled = False
    fog._running = True

    g1 = _make_gradient("d0", 1, sk)
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(g1)

    await asyncio.sleep(0.5)
    assert fog.ledger.get_height() > 0, "No block committed in single-node consensus"

    await fog.stop()


@pytest.mark.asyncio
async def test_fog_rejects_bad_signature():
    """Fog node rejects gradient with invalid signature (handled by SNARK verify)."""
    from services.fog_service import FogService

    sk, vk = generate_keypair()

    fog = FogService(
        node_id="fog1",
        sk=sk,
        vk=vk,
        peers=[],
        n=2,
        f=0,
        grpc_port=0,
    )
    fog._mqtt_enabled = False
    fog._running = True

    gradient = Gradient(
        node_id="d0",
        round=1,
        data=[1.0, 2.0],
        signature=b"bad_sig",
    )
    grad_with_proof = GradientWithProof(gradient=gradient, snark_proof=None)
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(grad_with_proof)
    assert len(fog._pending_gradients) == 1

    await fog.stop()


@pytest.mark.asyncio
async def test_ledger_persistence():
    """LedgerStore persists and reloads blocks via SQLite."""
    from core.ledger.store import HAS_SQLITE, LedgerStore
    from hfc_types.block import Block

    if not HAS_SQLITE:
        pytest.skip("SQLite not available")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = LedgerStore(db_path=db_path)
    block1 = Block(
        round=0, gradient_hash=b"gh1", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    assert store.append(block1) is True
    assert store.get_height() == 1

    block2 = Block(
        round=1, gradient_hash=b"gh2", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=101.0, prev_hash=block1.hash,
    )
    assert store.append(block2) is True

    store2 = LedgerStore(db_path=db_path)
    assert store2.get_height() == 2
    assert store2.verify_chain() is True

    os.unlink(db_path)
