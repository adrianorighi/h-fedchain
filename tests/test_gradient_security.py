"""Security regression tests for gradient verification (Task 4 review fixes).

Covers:
1. Signed message binds gradient CONTENT (tamper rejection)
2. Round freshness at ingress (stale round rejection + mixed-round batches)
3. Per-round dedup by device
4. Pinned/validated vk ingestion (KX peer scoping, registration first-writer-wins)
5. Quorum vote dedup (distinct voters only)
6. Diagnosability (warning logged when quorum not reached)
"""

import asyncio
import logging
import pickle

import numpy as np
import pytest

from core.pki import (
    generate_keypair,
    gradient_signed_message,
    sign as pki_sign,
    verify as pki_verify,
)
from hfc_types.messages import Gradient, GradientWithProof
from proto import hfedchain_pb2
from services.fog_service import FogService


def _mkfog(peers=("fog2",), **kwargs):
    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog1", sk=sk, vk=vk, peers=list(peers),
        n=3, f=1, grpc_port=0, **kwargs,
    )
    fog._mqtt_enabled = False
    return fog


def _mkgrad(node_id, round_num, sk, data):
    data = list(data)
    sig = pki_sign(sk, gradient_signed_message(node_id, round_num, data))
    return GradientWithProof(gradient=Gradient(
        node_id=node_id, round=round_num, data=data, signature=sig))


def _mkgrad_bytes(node_id, round_num, sk, data: bytes):
    sig = pki_sign(sk, gradient_signed_message(node_id, round_num, data))
    return GradientWithProof(gradient=Gradient(
        node_id=node_id, round=round_num, data=data, signature=sig))


# ----------------------------------------------------------------------
# Fix 1 — signed message binds gradient content
# ----------------------------------------------------------------------

def test_signed_message_round_trips_float32_and_binds_fields():
    arr = np.array([0.1, -2.5, 1e-8], dtype=np.float32)
    # float32 array and its tolist() form must hash identically
    assert gradient_signed_message("d0", 1, arr.tolist()) == \
        gradient_signed_message("d0", 1, arr)
    # content is bound
    assert gradient_signed_message("d0", 1, [0.1, -2.5]) != \
        gradient_signed_message("d0", 1, [0.1, -2.6])
    # node_id and round are bound
    assert gradient_signed_message("d0", 1, [1.0]) != \
        gradient_signed_message("d1", 1, [1.0])
    assert gradient_signed_message("d0", 1, [1.0]) != \
        gradient_signed_message("d0", 2, [1.0])


@pytest.mark.asyncio
async def test_tampered_gradient_data_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    original = [0.1, 0.2, 0.3]
    captured_sig = pki_sign(sk, gradient_signed_message("d0", 1, original))
    tampered = GradientWithProof(gradient=Gradient(
        node_id="d0", round=1, data=[0.9, 0.2, 0.3],
        signature=captured_sig))
    await fog.on_gradient_received(tampered)
    assert fog._pending_gradients == []


def test_edge_worker_signature_binds_content():
    from dataset.edge_worker import EdgeWorker

    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 12, 1000)).astype(np.float32)
    labels = rng.integers(0, 5, size=5)
    worker = EdgeWorker("d0", list(range(5)), data, labels,
                        is_adversarial=False)
    g = worker.train_round(worker.model.get_weights(), round_num=3).gradient
    assert pki_verify(
        worker.vk, gradient_signed_message("d0", 3, g.data), g.signature)
    tampered = list(g.data)
    tampered[0] += 1.0
    assert not pki_verify(
        worker.vk, gradient_signed_message("d0", 3, tampered), g.signature)


@pytest.mark.asyncio
async def test_orchestrator_synthetic_gradients_signed_with_content():
    from simulator.orchestrator import Orchestrator

    orch = Orchestrator(
        num_clusters=1, nodes_per_cluster=3, devices_per_cluster=4, f=1,
        adversarial_ratio=0.0,
    )
    orch.setup()
    grads = await orch._generate_gradients(round_num=2)
    assert grads
    for gwp in grads:
        g = gwp.gradient
        cert = orch._certificates[g.node_id]
        msg = gradient_signed_message(g.node_id, g.round, g.data)
        assert pki_verify(cert.public_key, msg, g.signature), \
            f"signature of {g.node_id} does not bind gradient content"
        tampered = list(g.data)
        tampered[0] += 1.0
        assert not pki_verify(
            cert.public_key,
            gradient_signed_message(g.node_id, g.round, tampered),
            g.signature,
        )


# ----------------------------------------------------------------------
# Fix 2 — round freshness at ingress + mixed-round batch guard
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_round_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    await fog.engine.start_round(3, is_leader=False)

    await fog.on_gradient_received(_mkgrad("d0", 3, sk, [1.0]))  # == current
    assert fog._pending_gradients == []
    await fog.on_gradient_received(_mkgrad("d0", 2, sk, [1.0]))  # older
    assert fog._pending_gradients == []

    await fog.on_gradient_received(_mkgrad("d0", 4, sk, [1.0]))
    assert len(fog._pending_gradients) == 1


@pytest.mark.asyncio
async def test_run_round_uses_max_round_and_filters_mixed_batch():
    fog = _mkfog()
    fog._running = True
    sk, vk = generate_keypair()
    for did in ("d0", "d1", "d2"):
        fog.set_vk(did, vk)

    async def _noop_leader(*args, **kwargs):
        pass

    fog._run_leader_consensus = _noop_leader
    fog._pending_gradients = [
        _mkgrad("d0", 1, sk, [1.0, 2.0]),
        _mkgrad("d1", 2, sk, [3.0, 4.0]),
        _mkgrad("d2", 2, sk, [5.0, 6.0]),
    ]
    await fog._run_round()
    assert fog._last_aggregate is not None
    assert fog._last_aggregate.round == 2
    assert "d0" not in fog._last_aggregate.accepted_devices
    assert fog._pending_gradients == []


# ----------------------------------------------------------------------
# Fix 3 — per-round dedup by device
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_device_same_round_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [0.0]))
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [1.0]))
    assert len(fog._pending_gradients) == 1
    assert fog._pending_gradients[0].gradient.data == [0.0]


# ----------------------------------------------------------------------
# Fix 4 — pinned/validated vk ingestion
# ----------------------------------------------------------------------

def test_spoofed_key_exchange_ignored():
    fog = _mkfog(peers=("fog2",))
    fog._peer_queues["fog2"] = asyncio.Queue()
    msg = hfedchain_pb2.ConsensusMessage(
        key_exchange=hfedchain_pb2.KeyExchange(
            node_id="evil", vk=b"\x11" * 32))

    async def go():
        await fog._on_consensus_message(msg, "x")
    asyncio.run(go())
    assert fog._vk_map == {}


def test_key_exchange_from_peer_with_malformed_vk_ignored():
    fog = _mkfog(peers=("fog2",))
    fog._peer_queues["fog2"] = asyncio.Queue()
    msg = hfedchain_pb2.ConsensusMessage(
        key_exchange=hfedchain_pb2.KeyExchange(node_id="fog2", vk=b"short"))

    async def go():
        await fog._on_consensus_message(msg, "x")
    asyncio.run(go())
    assert fog._vk_map == {}
    assert fog._peer_queues["fog2"].empty()


def test_registration_overwrite_keeps_first_vk():
    fog = _mkfog()
    _, vk1 = generate_keypair()
    _, vk2 = generate_keypair()
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk1}))
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk2}))
    assert fog._vk_map["d0"] == vk1


def test_registration_same_vk_is_noop():
    fog = _mkfog()
    _, vk = generate_keypair()
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk}))
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk}))
    assert fog._vk_map == {"d0": vk}


def test_registration_malformed_vk_ignored():
    fog = _mkfog()
    _, vk = generate_keypair()
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk}))
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": b"short"}))
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d1", "vk": b"short"}))
    fog._on_mqtt_registration(pickle.dumps({"node_id": 7, "vk": vk}))
    fog._on_mqtt_registration(pickle.dumps(["not", "a", "dict"]))
    assert fog._vk_map == {"d0": vk}


@pytest.mark.asyncio
async def test_malformed_vk_in_map_does_not_raise():
    fog = _mkfog()
    sk, _ = generate_keypair()
    fog.set_vk("d0", b"short")          # poisoned entry (programmatic)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [1.0]))
    assert fog._pending_gradients == []


# ----------------------------------------------------------------------
# Task 6 Fix 6 — MQTT peer-vk spoof + unscoped voter ids
# ----------------------------------------------------------------------

def test_registration_for_peer_id_ignored(caplog):
    """Peer vks come only from key exchange; an MQTT registration that
    claims a peer id must be ignored entirely."""
    fog = _mkfog(peers=("fog2",))
    _, vk = generate_keypair()

    with caplog.at_level(logging.WARNING):
        fog._on_mqtt_registration(pickle.dumps({"node_id": "fog2", "vk": vk}))

    assert "fog2" not in fog._vk_map, (
        "registration must not populate a peer's vk"
    )
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("fog2" in w for w in warnings), (
        f"expected a warning for the ignored peer registration, got: {warnings}"
    )


@pytest.mark.asyncio
async def test_vote_from_non_peer_id_not_counted():
    """The collector only counts votes from peers (the voters): a vote
    carrying an arbitrary registered device id must not reach quorum."""
    fog = _mkfog(peers=("fog2",), vote_timeout_s=0.3)
    vsk, vvk = generate_keypair()
    fog.set_vk("registered_device", vvk)  # MQTT-registered, not a voter

    sig = pki_sign(vsk, b"1" + b"hash" + b"PREPARE")
    vote = hfedchain_pb2.VoteMessage(
        node_id="registered_device", round=1, phase="prepare",
        block_hash=b"hash", signature=sig,
    )

    async def inject():
        await asyncio.sleep(0.05)
        fog._pending_votes.append(vote)
        fog._vote_event.set()

    asyncio.create_task(inject())
    result = await fog._collect_quorum_votes(1, b"hash", "prepare", 1)
    assert result is None, (
        "a vote from a non-peer id must not count toward quorum"
    )


@pytest.mark.asyncio
async def test_vote_from_peer_counted_toward_quorum():
    fog = _mkfog(peers=("fog2",), vote_timeout_s=0.3)
    f2_sk, f2_vk = generate_keypair()
    fog.set_vk("fog2", f2_vk)

    sig = pki_sign(f2_sk, b"1" + b"hash" + b"PREPARE")
    vote = hfedchain_pb2.VoteMessage(
        node_id="fog2", round=1, phase="prepare",
        block_hash=b"hash", signature=sig,
    )

    async def inject():
        await asyncio.sleep(0.05)
        fog._pending_votes.append(vote)
        fog._vote_event.set()

    asyncio.create_task(inject())
    result = await fog._collect_quorum_votes(1, b"hash", "prepare", 1)
    assert result is not None, "a peer's valid vote must count toward quorum"
    assert result.signatures == [("fog2", sig)]


# ----------------------------------------------------------------------
# Fix 5 + 6 — vote dedup and diagnosability
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vote_dedup_requires_distinct_voters(caplog):
    fog = _mkfog(vote_timeout_s=0.3)
    vsk, vvk = generate_keypair()
    fog.set_vk("v1", vvk)

    sig = pki_sign(vsk, b"1" + b"hash" + b"PREPARE")
    vote = hfedchain_pb2.VoteMessage(
        node_id="v1", round=1, phase="prepare",
        block_hash=b"hash", signature=sig)

    async def inject():
        await asyncio.sleep(0.05)
        fog._pending_votes.extend([vote, vote])   # same voter replayed twice
        fog._vote_event.set()

    asyncio.get_event_loop().create_task(inject())
    with caplog.at_level(logging.WARNING):
        result = await fog._collect_quorum_votes(1, b"hash", "prepare", 2)
    assert result is None

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert warnings, "expected a warning when quorum is not reached"
    msg = warnings[0].lower()
    assert "quorum" in msg
    assert "prepare" in msg
    assert "round 1" in msg


# ======================================================================
# Round-2 review fixes
# ======================================================================

# ----------------------------------------------------------------------
# Round-2 Fix 1 — KX first-writer-wins (pin like registration does)
# ----------------------------------------------------------------------

def _kx(node_id: str, vk: bytes):
    return hfedchain_pb2.ConsensusMessage(
        key_exchange=hfedchain_pb2.KeyExchange(node_id=node_id, vk=vk))


def test_kx_conflicting_vk_keeps_original(caplog):
    fog = _mkfog(peers=("fog2",))
    fog._peer_queues["fog2"] = asyncio.Queue()
    vk1, vk2 = b"\x01" * 32, b"\x02" * 32

    async def go():
        await fog._on_consensus_message(_kx("fog2", vk1), "p")
        while not fog._peer_queues["fog2"].empty():
            fog._peer_queues["fog2"].get_nowait()
        await fog._on_consensus_message(_kx("fog2", vk2), "p")

    with caplog.at_level(logging.WARNING):
        asyncio.run(go())

    assert fog._vk_map["fog2"] == vk1, "conflicting KX must not re-key a peer"
    assert fog._peer_queues["fog2"].empty(), "must not echo on a re-key attempt"
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("fog2" in w for w in warnings), "expected a pinning warning"


def test_kx_same_vk_is_noop_no_second_echo(caplog):
    fog = _mkfog(peers=("fog2",))
    fog._peer_queues["fog2"] = asyncio.Queue()
    vk = b"\x03" * 32

    async def go():
        await fog._on_consensus_message(_kx("fog2", vk), "p")
        while not fog._peer_queues["fog2"].empty():
            fog._peer_queues["fog2"].get_nowait()
        await fog._on_consensus_message(_kx("fog2", vk), "p")

    with caplog.at_level(logging.WARNING):
        asyncio.run(go())

    assert fog._vk_map["fog2"] == vk
    assert fog._peer_queues["fog2"].empty(), "same vk must not re-echo"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_kx_new_peer_still_echoes_once():
    """Pinning must not break the original single-echo behavior."""
    fog = _mkfog(peers=("fog2",))
    fog._peer_queues["fog2"] = asyncio.Queue()

    async def go():
        await fog._on_consensus_message(_kx("fog2", b"\x04" * 32), "p")

    asyncio.run(go())
    assert fog._vk_map["fog2"] == b"\x04" * 32
    assert not fog._peer_queues["fog2"].empty()


# ----------------------------------------------------------------------
# Round-2 Fix 3 — round-scoped dedup (same device, different round)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_device_different_round_both_accepted():
    fog = _mkfog(gradient_threshold=10)
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [1.0]))
    await fog.on_gradient_received(_mkgrad("d0", 2, sk, [2.0]))
    rounds = sorted(item.gradient.round for item in fog._pending_gradients)
    assert rounds == [1, 2], "dedup must be per (device, round), not per device"


@pytest.mark.asyncio
async def test_duplicate_device_same_round_still_rejected_with_new_dedup():
    fog = _mkfog(gradient_threshold=10)
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [1.0]))
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, [2.0]))
    assert len(fog._pending_gradients) == 1
    assert fog._pending_gradients[0].gradient.data == [1.0]


# ----------------------------------------------------------------------
# Round-2 Fix 4 — flat numeric data validation at ingress
# (validated BEFORE signature verification: bad data is rejected
#  regardless of signature validity)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nested_gradient_data_rejected_despite_valid_signature():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    nested = [[0.1, 0.2], [0.3, 0.4]]          # signable (np.asarray → 2D)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, nested))
    assert fog._pending_gradients == []


@pytest.mark.asyncio
async def test_non_numeric_data_rejected_before_signature_verify(caplog):
    fog = _mkfog()
    _, vk = generate_keypair()
    fog.set_vk("d0", vk)
    g = GradientWithProof(gradient=Gradient(
        node_id="d0", round=1, data=["a", "b"], signature=b"x"))
    with caplog.at_level(logging.WARNING):
        await fog.on_gradient_received(g)
    assert fog._pending_gradients == []
    warnings = [r.getMessage().lower() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("data" in w for w in warnings), \
        "expected a data-validation warning, got: %r" % warnings
    assert not any("pki" in w for w in warnings), \
        "data must be validated before signature verification"


@pytest.mark.asyncio
async def test_empty_gradient_data_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    await fog.on_gradient_received(_mkgrad("d0", 1, sk, []))
    assert fog._pending_gradients == []


# ======================================================================
# Fix 5 — float32-bytes gradient payloads (wire format)
# ======================================================================

def test_signed_message_list_and_bytes_forms_are_equivalent():
    """float32 → tolist() → float32 is exact, so the canonical hash of a
    list and of its raw float32 bytes are identical: a signature made over
    either representation must verify over the other."""
    arr = np.array([0.1, -2.5, 1e-8], dtype=np.float32)
    raw = arr.tobytes()

    assert gradient_signed_message("d0", 1, arr.tolist()) == \
        gradient_signed_message("d0", 1, raw)

    sk, vk = generate_keypair()
    sig_list = pki_sign(sk, gradient_signed_message("d0", 1, arr.tolist()))
    assert pki_verify(vk, gradient_signed_message("d0", 1, raw), sig_list)

    sig_bytes = pki_sign(sk, gradient_signed_message("d0", 1, raw))
    assert pki_verify(vk, gradient_signed_message("d0", 1, arr.tolist()),
                      sig_bytes)


@pytest.mark.asyncio
async def test_bytes_data_gradient_accepted_at_ingress():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    # signed over the LIST form; delivered as BYTES (cross-representation)
    await fog.on_gradient_received(_mkgrad_bytes(
        "d0", 1, sk, data.tobytes()))
    assert len(fog._pending_gradients) == 1, (
        "a bytes-data gradient with a valid signature must reach pending"
    )


@pytest.mark.asyncio
async def test_mixed_list_and_bytes_same_round_accepted():
    """Same-round gradients in either wire format must coexist when they
    have the same element count — and a different element count must still
    be rejected (size compares ELEMENTS, not bytes/len)."""
    fog = _mkfog(gradient_threshold=10)
    sk, vk = generate_keypair()
    for did in ("d0", "d1", "d2"):
        fog.set_vk(did, vk)

    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    await fog.on_gradient_received(_mkgrad_bytes("d0", 1, sk, data.tobytes()))
    await fog.on_gradient_received(_mkgrad("d1", 1, sk, data.tolist()))
    assert len(fog._pending_gradients) == 2, (
        "bytes and list payloads with equal element counts must coexist"
    )

    other = np.array([1.0, 2.0], dtype=np.float32)
    await fog.on_gradient_received(_mkgrad_bytes("d2", 1, sk, other.tobytes()))
    assert len(fog._pending_gradients) == 2, (
        "a different element count must still be rejected"
    )


@pytest.mark.asyncio
async def test_run_round_aggregates_bytes_gradients():
    fog = _mkfog(gradient_threshold=1)
    fog._running = True

    async def _noop_leader(*args, **kwargs):
        pass

    fog._run_leader_consensus = _noop_leader
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    await fog.on_gradient_received(_mkgrad_bytes("d0", 1, sk, data.tobytes()))

    assert fog._last_aggregate is not None, (
        "a bytes gradient must flow all the way into _run_round aggregation"
    )
    assert fog._last_aggregate.round == 1
    assert np.allclose(
        np.asarray(fog._last_aggregate.gradient.data, dtype=np.float32), data
    )
    assert fog._pending_gradients == []
