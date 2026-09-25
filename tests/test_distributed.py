"""End-to-end tests for distributed H-FedChain services."""

import asyncio
import os
import pickle
import tempfile
import time

import pytest

from core.pki import generate_keypair
from hfc_types.messages import Gradient, GradientWithProof


def _make_gradient(node_id: str, round_num: int, sk: bytes) -> GradientWithProof:
    from core.pki import gradient_signed_message, sign as pki_sign
    data = [float(i) for i in range(10)]
    gradient = Gradient(
        node_id=node_id, round=round_num, data=data,
        signature=pki_sign(sk, gradient_signed_message(node_id, round_num, data)),
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
    """Fog node rejects gradient with invalid Ed25519 signature (PKI verify)."""
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
    assert len(fog._pending_gradients) == 0

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


def _mk_consensus_fog(node_id, peers, **kwargs):
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    fog = FogService(
        node_id=node_id, sk=sk, vk=vk, peers=list(peers), n=3, f=1,
        grpc_port=0, cloud_address="127.0.0.1:1", **kwargs,
    )
    fog._mqtt_enabled = False
    fog._running = True
    return fog


async def _pump(fog, peer_id, queue, targets):
    while True:
        msg = await queue.get()
        await targets[peer_id]._on_consensus_message(msg, fog.node_id)


def _wire_fogs(fogs):
    targets = {f.node_id: f for f in fogs}
    pumps = []
    for src in fogs:
        for peer in src.peers:
            queue = asyncio.Queue()
            src._peer_queues[peer] = queue
            pumps.append(asyncio.create_task(_pump(src, peer, queue, targets)))
    return pumps


async def _exchange_kx(fogs):
    from proto import hfedchain_pb2

    for dst in fogs:
        for src in fogs:
            if src is dst:
                continue
            msg = hfedchain_pb2.ConsensusMessage(
                key_exchange=hfedchain_pb2.KeyExchange(
                    node_id=src.node_id, vk=src.engine.vk
                )
            )
            await dst._on_consensus_message(msg, src.node_id)


async def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_three_fog_round_commits_everywhere():
    fogs = [
        _mk_consensus_fog("fog1", ["fog2", "fog3"]),
        _mk_consensus_fog("fog2", ["fog1", "fog3"]),
        _mk_consensus_fog("fog3", ["fog1", "fog2"]),
    ]
    pumps = _wire_fogs(fogs)
    try:
        await _exchange_kx(fogs)
        dev_sk, _ = generate_keypair()
        for fog in fogs:
            fog._pending_gradients = [
                _make_gradient("d0", 1, dev_sk),
                _make_gradient("d1", 1, dev_sk),
            ]

        await asyncio.gather(*[f._run_round() for f in fogs])

        assert await _wait_until(
            lambda: all(f.ledger.get_height() == 1 for f in fogs)
        ), "not every fog committed the round"
        hashes = {f.ledger._entries[-1].block.hash for f in fogs}
        assert len(hashes) == 1
        for fog in fogs:
            assert fog.ledger.get_height() == 1
    finally:
        for pump in pumps:
            pump.cancel()


@pytest.mark.asyncio
async def test_view_change_recovers_stalled_round():
    fogs = [
        _mk_consensus_fog("fog1", ["fog2", "fog3"], vote_timeout_s=0.3),
        _mk_consensus_fog("fog2", ["fog1", "fog3"], vote_timeout_s=0.3),
        _mk_consensus_fog("fog3", ["fog1", "fog2"], vote_timeout_s=0.3),
    ]
    by_id = {f.node_id: f for f in fogs}
    pumps = _wire_fogs(fogs)
    original_handler = by_id["fog3"]._on_consensus_message

    async def drop_votes(msg, sender_id):
        if msg.WhichOneof("msg") == "vote":
            return
        await original_handler(msg, sender_id)

    by_id["fog3"]._on_consensus_message = drop_votes
    try:
        await _exchange_kx(fogs)
        dev_sk, _ = generate_keypair()
        for fog in fogs:
            fog._pending_gradients = [
                _make_gradient("d0", 2, dev_sk),
                _make_gradient("d1", 2, dev_sk),
            ]

        await asyncio.gather(*[f._run_round() for f in fogs])

        assert 2 in by_id["fog3"]._vc_sent, "leader must broadcast VC"

        assert await _wait_until(
            lambda: all(f.ledger.get_height() == 1 for f in fogs)
        ), "view change did not recover the round"
        hashes = {f.ledger._entries[-1].block.hash for f in fogs}
        assert len(hashes) == 1
        for fog in fogs:
            assert 2 in fog._vc_sent, f"{fog.node_id} did not echo VC"
            assert 2 in fog._vc_applied, f"{fog.node_id} did not apply VC quorum"
    finally:
        for pump in pumps:
            pump.cancel()


@pytest.mark.asyncio
async def test_round_commits_with_only_one_followers_votes():
    """Task 6 Fix 3 — the leader counts its own vote: quorum 2 = self + 1,
    so the round must commit even when only ONE follower's votes reach the
    leader (the other follower's votes are dropped)."""
    fogs = [
        _mk_consensus_fog("fog1", ["fog2", "fog3"], vote_timeout_s=0.5),
        _mk_consensus_fog("fog2", ["fog1", "fog3"], vote_timeout_s=0.5),
        _mk_consensus_fog("fog3", ["fog1", "fog2"], vote_timeout_s=0.5),
    ]
    by_id = {f.node_id: f for f in fogs}
    pumps = _wire_fogs(fogs)

    # Round 1 leader = sorted([fog1, fog2, fog3])[1 % 3] == "fog2".
    leader = by_id["fog2"]
    original_handler = leader._on_consensus_message

    async def drop_fog3_votes(msg, sender_id):
        if msg.WhichOneof("msg") == "vote" and msg.vote.node_id == "fog3":
            return
        await original_handler(msg, sender_id)

    leader._on_consensus_message = drop_fog3_votes
    try:
        await _exchange_kx(fogs)
        dev_sk, _ = generate_keypair()
        for fog in fogs:
            fog._pending_gradients = [
                _make_gradient("d0", 1, dev_sk),
                _make_gradient("d1", 1, dev_sk),
            ]

        await asyncio.gather(*[f._run_round() for f in fogs])

        assert await _wait_until(
            lambda: all(f.ledger.get_height() == 1 for f in fogs)
        ), "self-vote + one follower must reach quorum and commit everywhere"
        hashes = {f.ledger._entries[-1].block.hash for f in fogs}
        assert len(hashes) == 1
    finally:
        for pump in pumps:
            pump.cancel()
