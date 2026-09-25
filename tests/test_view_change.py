import asyncio
import pickle

import pytest

from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.view_change import ViewChangeHandler
from core.pki import generate_keypair, sign as pki_sign, verify as pki_verify
from hfc_types.block import Block
from hfc_types.messages import Gradient, GradientWithProof, MessageType


@pytest.fixture
def vc():
    return ViewChangeHandler(n=5, f=1)


def test_should_change_view_on_timeout(vc):
    assert vc.should_change_view(timeout=True) is True


def test_no_change_without_timeout(vc):
    assert vc.should_change_view(timeout=False) is False


def test_next_leader_rotation(vc):
    node_ids = ["n0", "n1", "n2", "n3", "n4"]
    leaders = []
    for i in range(5):
        leaders.append(vc.next_leader(node_ids))
    assert len(set(leaders)) == 5
    assert leaders[0] == "n1"


def test_record_highest_qc(vc):
    vc.record_highest_qc(round=5, qc=b"qc_data")
    assert vc._highest_qc == (5, b"qc_data")


def test_record_highest_qc_ignores_older(vc):
    vc.record_highest_qc(round=5, qc=b"higher")
    vc.record_highest_qc(round=3, qc=b"lower")
    assert vc._highest_qc == (5, b"higher")


def test_create_view_change_with_qc(vc):
    sk, vk = generate_keypair()
    vc.record_highest_qc(round=1, qc=b"some_qc")
    msg = vc.create_view_change(node_id="n3", new_view=6, sk=sk, round=6)
    assert msg.new_view == 6
    assert msg.node_id == "n3"
    assert msg.round == 6
    assert msg.highest_qc == (1, b"some_qc")
    assert len(msg.signature) > 0


def test_create_view_change_without_qc(vc):
    sk, vk = generate_keypair()
    msg = vc.create_view_change(node_id="n0", new_view=2, sk=sk, round=3)
    assert msg.highest_qc is None


def test_create_new_view(vc):
    qcs = [b"qc1", b"qc2"]
    msg = vc.create_new_view(leader_id="n1", new_view=6, qc_set=qcs)
    assert msg.new_view == 6
    assert len(msg.qc_set) == 2
    assert msg.leader_id == "n1"


def test_view_change_message_has_round_field(vc):
    from core.hotstuff.messages import ViewChangeMessage

    fields = [f.name for f in ViewChangeMessage.__dataclass_fields__.values()]
    assert fields == ["node_id", "new_view", "highest_qc", "signature", "round"]


def test_create_view_change_signs_round_and_verifies(vc):
    sk, vk = generate_keypair()
    vc.record_highest_qc(round=3, qc=b"qc_blob")
    msg = vc.create_view_change(node_id="n2", new_view=7, sk=sk, round=5)
    assert msg.round == 5
    payload = str(5).encode() + b"qc_blob"
    assert pki_verify(vk, payload, msg.signature)


def test_create_view_change_without_qc_signs_round(vc):
    sk, vk = generate_keypair()
    msg = vc.create_view_change(node_id="n0", new_view=2, sk=sk, round=9)
    assert pki_verify(vk, str(9).encode(), msg.signature)


def _mkblock(round_num=1, timestamp=100.0, gradient_hash=b"gh"):
    return Block(
        round=round_num, gradient_hash=gradient_hash, qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=timestamp, prev_hash=b"\x00" * 32,
    )


@pytest.fixture
def engine():
    sk, vk = generate_keypair()
    return HotStuffEngine(
        node_id="n0", sk=sk, vk=vk, peers=["n1", "n2"], n=3, f=1,
    )


@pytest.mark.asyncio
async def test_on_prepare_signs_proposal_round_when_stale(engine):
    block = _mkblock(round_num=5)
    vote = await engine.on_prepare(block)
    assert vote is not None
    msg = str(5).encode() + block.hash + MessageType.PREPARE.name.encode()
    assert pki_verify(engine.vk, msg, vote.signature)


@pytest.mark.asyncio
async def test_on_pre_commit_signs_qc_round_when_stale(engine):
    qc = QuorumCertifier().collect(
        round=7, block_hash=b"bh", msg_type=MessageType.PREPARE,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    vote = await engine.on_pre_commit(qc)
    assert vote is not None
    msg = str(7).encode() + qc.block_hash + MessageType.PRE_COMMIT.name.encode()
    assert pki_verify(engine.vk, msg, vote.signature)


@pytest.mark.asyncio
async def test_on_commit_signs_qc_round_when_stale(engine):
    qc = QuorumCertifier().collect(
        round=7, block_hash=b"bh", msg_type=MessageType.PRE_COMMIT,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    vote = await engine.on_commit(qc)
    assert vote is not None
    msg = str(7).encode() + qc.block_hash + MessageType.COMMIT.name.encode()
    assert pki_verify(engine.vk, msg, vote.signature)


@pytest.mark.asyncio
async def test_on_prepare_resend_same_round_same_hash_returns_vote(engine):
    block = _mkblock(round_num=1)
    vote1 = await engine.on_prepare(block)
    vote2 = await engine.on_prepare(block)
    assert vote1 is not None
    assert vote2 is not None
    assert vote1.signature == vote2.signature
    assert vote2.block_hash == vote1.block_hash


@pytest.mark.asyncio
async def test_on_prepare_same_round_different_hash_rejected(engine):
    block_a = _mkblock(round_num=1, timestamp=100.0)
    block_b = _mkblock(round_num=1, timestamp=200.0)
    assert await engine.on_prepare(block_a) is not None
    assert await engine.on_prepare(block_b) is None
    assert await engine.on_prepare(block_a) is not None


@pytest.mark.asyncio
async def test_on_pre_commit_resend_and_conflicting_hash(engine):
    qc = QuorumCertifier().collect(
        round=2, block_hash=b"h1", msg_type=MessageType.PREPARE,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    other = QuorumCertifier().collect(
        round=2, block_hash=b"h2", msg_type=MessageType.PREPARE,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    assert await engine.on_pre_commit(qc) is not None
    assert await engine.on_pre_commit(qc) is not None
    assert await engine.on_pre_commit(other) is None


@pytest.mark.asyncio
async def test_on_qc_commit_rejects_wrong_block_hash(engine):
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    qc = QuorumCertifier().collect(
        round=3, block_hash=b"\xaa" * 32, msg_type=MessageType.COMMIT,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    assert await engine.on_qc_commit(qc) is None
    assert engine.state != "DECIDED"


@pytest.mark.asyncio
async def test_on_qc_commit_rejects_wrong_round(engine):
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    qc = QuorumCertifier().collect(
        round=4, block_hash=block.hash, msg_type=MessageType.COMMIT,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    assert await engine.on_qc_commit(qc) is None
    assert engine.state != "DECIDED"


@pytest.mark.asyncio
async def test_on_qc_commit_accepts_valid_qc_with_vk_map(engine):
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    keys = {f"m{i}": generate_keypair() for i in range(2)}
    vk_map = {nid: vk for nid, (sk, vk) in keys.items()}
    msg = str(3).encode() + block.hash + MessageType.COMMIT.name.encode()
    sigs = [(nid, pki_sign(sk, msg)) for nid, (sk, _) in keys.items()]
    quorum = QuorumCertifier.quorum_size(engine.n)
    qc = QuorumCertifier().collect(
        3, block.hash, MessageType.COMMIT, sigs, quorum, vk_map,
    )
    entry = await engine.on_qc_commit(qc, vk_map=vk_map)
    assert entry is not None
    assert entry.block.hash == block.hash
    assert engine.state == "DECIDED"


@pytest.mark.asyncio
async def test_on_qc_commit_rejects_invalid_signature(engine):
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    quorum = QuorumCertifier.quorum_size(engine.n)
    qc = QuorumCertifier().collect(
        3, block.hash, MessageType.COMMIT,
        [(f"m{i}", b"bad") for i in range(quorum)], quorum, None,
    )
    vk_map = {f"m{i}": generate_keypair()[1] for i in range(quorum)}
    assert await engine.on_qc_commit(qc, vk_map=vk_map) is None
    assert engine.state != "DECIDED"


# ----------------------------------------------------------------------
# Task 6 Fix 2 — commit certificate must be COMMIT-phase; QCs verified
# before voting
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_qc_commit_rejects_prepare_phase_qc(engine):
    """A PREPARE-phase QC must never be treated as a commit certificate."""
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    qc = QuorumCertifier().collect(
        round=3, block_hash=block.hash, msg_type=MessageType.PREPARE,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    assert await engine.on_qc_commit(qc) is None
    assert engine.state != "DECIDED"


@pytest.mark.asyncio
async def test_on_qc_commit_rejects_pre_commit_phase_qc(engine):
    await engine.start_round(3, is_leader=True)
    block = _mkblock(round_num=3)
    await engine.propose(block)
    qc = QuorumCertifier().collect(
        round=3, block_hash=block.hash, msg_type=MessageType.PRE_COMMIT,
        signatures=[("n1", b"s"), ("n2", b"s")], quorum_size=2,
    )
    assert await engine.on_qc_commit(qc) is None
    assert engine.state != "DECIDED"


def _real_sig_qc(round_num, block_hash, msg_type, quorum=2):
    """QC with real signatures over the QC's own msg_type (so signature
    verification passes and only the phase check can reject it)."""
    keys = {f"m{i}": generate_keypair() for i in range(quorum)}
    vk_map = {nid: vk for nid, (sk, vk) in keys.items()}
    msg = str(round_num).encode() + block_hash + msg_type.name.encode()
    sigs = [(nid, pki_sign(sk, msg)) for nid, (sk, _) in keys.items()]
    qc = QuorumCertifier().collect(
        round_num, block_hash, msg_type, sigs, quorum,
    )
    return qc, vk_map


@pytest.mark.asyncio
async def test_on_pre_commit_rejects_invalid_signature_with_vk_map(engine):
    await engine.start_round(3, is_leader=False)
    vk_map = {"m0": generate_keypair()[1], "m1": generate_keypair()[1]}
    qc = QuorumCertifier().collect(
        round=3, block_hash=b"bh", msg_type=MessageType.PREPARE,
        signatures=[("m0", b"bad0"), ("m1", b"bad1")], quorum_size=2,
    )
    assert await engine.on_pre_commit(qc, vk_map=vk_map) is None, (
        "invalid-signature QC must not produce a pre_commit vote"
    )


@pytest.mark.asyncio
async def test_on_pre_commit_rejects_wrong_phase_with_vk_map(engine):
    qc, vk_map = _real_sig_qc(3, b"bh", MessageType.PRE_COMMIT)
    assert await engine.on_pre_commit(qc, vk_map=vk_map) is None, (
        "only a PREPARE-phase QC may drive a pre_commit vote"
    )


@pytest.mark.asyncio
async def test_on_pre_commit_accepts_valid_prepare_qc_with_vk_map(engine):
    qc, vk_map = _real_sig_qc(3, b"bh", MessageType.PREPARE)
    vote = await engine.on_pre_commit(qc, vk_map=vk_map)
    assert vote is not None
    assert vote.phase == "pre_commit"


@pytest.mark.asyncio
async def test_on_commit_rejects_wrong_phase_with_vk_map(engine):
    qc, vk_map = _real_sig_qc(3, b"bh", MessageType.PREPARE)
    assert await engine.on_commit(qc, vk_map=vk_map) is None, (
        "only a PRE_COMMIT-phase QC may drive a commit vote"
    )


@pytest.mark.asyncio
async def test_on_commit_rejects_invalid_signature_with_vk_map(engine):
    vk_map = {"m0": generate_keypair()[1], "m1": generate_keypair()[1]}
    qc = QuorumCertifier().collect(
        round=3, block_hash=b"bh", msg_type=MessageType.PRE_COMMIT,
        signatures=[("m0", b"bad0"), ("m1", b"bad1")], quorum_size=2,
    )
    assert await engine.on_commit(qc, vk_map=vk_map) is None


@pytest.mark.asyncio
async def test_fog_qc_broadcast_with_invalid_signature_casts_no_vote():
    """Fog must pass vk_map when processing inbound QC broadcasts: a QC
    with bad signatures produces no vote back to the leader."""
    from proto import hfedchain_pb2

    fog = _mkfog(peers=["fog2"], n=3, f=1)
    fog._peer_queues["fog2"] = asyncio.Queue()
    _, f2_vk = generate_keypair()
    fog.set_vk("fog2", f2_vk)

    qc = QuorumCertifier().collect(
        round=1, block_hash=b"bh", msg_type=MessageType.PREPARE,
        signatures=[("fog1", b"bad"), ("fog2", b"bad")], quorum_size=2,
    )
    msg = hfedchain_pb2.ConsensusMessage(
        qc=hfedchain_pb2.QcBroadcast(
            leader_id="fog2", round=1, phase="pre_commit",
            quorum_certificate=pickle.dumps(qc),
        )
    )
    await fog._on_consensus_message(msg, "fog2")
    assert fog._peer_queues["fog2"].empty(), (
        "an invalid QC must not elicit a vote"
    )


def _mkfog(**kwargs):
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    defaults = dict(
        node_id="fog1", sk=sk, vk=vk, peers=[], n=1, f=0,
        grpc_port=0, cloud_address="127.0.0.1:1",
    )
    defaults.update(kwargs)
    fog = FogService(**defaults)
    fog._mqtt_enabled = False
    return fog


def _mkgrad(node_id, round_num, sk, data):
    from core.pki import gradient_signed_message

    data = list(data)
    sig = pki_sign(sk, gradient_signed_message(node_id, round_num, data))
    return GradientWithProof(gradient=Gradient(
        node_id=node_id, round=round_num, data=data, signature=sig))


@pytest.mark.asyncio
async def test_single_node_commit_qc_has_valid_real_signature():
    fog = _mkfog()
    fog._running = True
    sk, _ = generate_keypair()
    fog._pending_gradients = [_mkgrad("d0", 1, sk, [1.0, 2.0, 3.0])]

    await fog._run_round()

    assert fog.ledger.get_height() == 1
    qc = fog.ledger._entries[-1].block.qc_commit
    quorum = fog.quorum_certifier.quorum_size(fog.n)
    assert qc.is_valid(quorum, {fog.node_id: fog.engine.vk})


def test_canonical_nodes_order_independent():
    fog_a = _mkfog(node_id="fog1", peers=["fog3", "fog2"], n=3, f=1)
    fog_b = _mkfog(node_id="fog3", peers=["fog2", "fog1"], n=3, f=1)
    assert fog_a._canonical_nodes() == fog_b._canonical_nodes()
    round_num = 2
    leader_a = fog_a._canonical_nodes()[round_num % 3]
    leader_b = fog_b._canonical_nodes()[round_num % 3]
    assert leader_a == leader_b


# ----------------------------------------------------------------------
# Task 6 Fix 4 — deterministic VC leader (no shared mutable view state)
# ----------------------------------------------------------------------

def test_vc_leader_identical_across_nodes():
    fog_a = _mkfog(node_id="fog1", peers=["fog3", "fog2"], n=3, f=1)
    fog_b = _mkfog(node_id="fog2", peers=["fog1", "fog3"], n=3, f=1)
    fog_c = _mkfog(node_id="fog3", peers=["fog2", "fog1"], n=3, f=1)
    for r in range(6):
        leaders = {fog_a._vc_leader(r), fog_b._vc_leader(r), fog_c._vc_leader(r)}
        assert len(leaders) == 1, (
            f"round {r}: nodes elected different VC leaders {leaders}"
        )


def test_vc_leader_is_successor_of_round_robin_leader():
    fog = _mkfog(node_id="fog1", peers=["fog2", "fog3"], n=3, f=1)
    nodes = ["fog1", "fog2", "fog3"]
    for r in range(6):
        orig = nodes[r % len(nodes)]
        expected = nodes[(nodes.index(orig) + 1) % len(nodes)]
        assert fog._vc_leader(r) == expected, f"round {r}"


def test_vc_leader_single_node_is_self():
    fog = _mkfog(node_id="solo", peers=[], n=1, f=0)
    assert fog._vc_leader(3) == "solo"


@pytest.mark.asyncio
async def test_send_view_change_broadcasts_valid_signature_and_dedupe():
    from proto import hfedchain_pb2

    fog = _mkfog(peers=["fog2"], n=2, f=0)
    fog._peer_queues["fog2"] = asyncio.Queue()

    await fog._send_view_change(4)

    msg = fog._peer_queues["fog2"].get_nowait()
    assert msg.WhichOneof("msg") == "view_change"
    vc = msg.view_change
    assert vc.node_id == "fog1"
    assert vc.round == 4
    assert vc.highest_qc == b""
    assert pki_verify(fog.engine.vk, str(4).encode(), vc.signature)

    await fog._send_view_change(4)
    assert fog._peer_queues["fog2"].empty()
    assert fog._vc_sent == {4}


@pytest.mark.asyncio
async def test_view_change_from_peer_validated_and_echoed():
    fog = _mkfog(peers=["fog2", "fog3"], n=3, f=1)
    fog._peer_queues["fog2"] = asyncio.Queue()
    fog._peer_queues["fog3"] = asyncio.Queue()
    f2_sk, f2_vk = generate_keypair()
    fog.set_vk("fog2", f2_vk)
    from proto import hfedchain_pb2

    msg = hfedchain_pb2.ConsensusMessage(
        view_change=hfedchain_pb2.ViewChangeMessage(
            node_id="fog2", round=1, new_view=2, highest_qc=b"",
            signature=pki_sign(f2_sk, str(1).encode()),
        )
    )
    await fog._on_consensus_message(msg, "fog2")

    assert 1 in fog._vc_sent
    for peer in ("fog2", "fog3"):
        echo = fog._peer_queues[peer].get_nowait()
        assert echo.WhichOneof("msg") == "view_change"
        assert echo.view_change.node_id == "fog1"
        assert echo.view_change.round == 1
    assert fog._pending_vc[1] == {"fog2", "fog1"}


@pytest.mark.asyncio
async def test_view_change_invalid_signature_ignored():
    fog = _mkfog(peers=["fog2"], n=3, f=1)
    fog._peer_queues["fog2"] = asyncio.Queue()
    _, f2_vk = generate_keypair()
    fog.set_vk("fog2", f2_vk)
    from proto import hfedchain_pb2

    msg = hfedchain_pb2.ConsensusMessage(
        view_change=hfedchain_pb2.ViewChangeMessage(
            node_id="fog2", round=1, new_view=2, highest_qc=b"",
            signature=b"garbage",
        )
    )
    await fog._on_consensus_message(msg, "fog2")
    assert fog._vc_sent == set()
    assert fog._pending_vc == {}
    assert fog._peer_queues["fog2"].empty()


@pytest.mark.asyncio
async def test_view_change_from_non_peer_ignored():
    fog = _mkfog(peers=["fog2"], n=3, f=1)
    fog._peer_queues["fog2"] = asyncio.Queue()
    f2_sk, f2_vk = generate_keypair()
    fog.set_vk("evil", f2_vk)
    from proto import hfedchain_pb2

    msg = hfedchain_pb2.ConsensusMessage(
        view_change=hfedchain_pb2.ViewChangeMessage(
            node_id="evil", round=1, new_view=2, highest_qc=b"",
            signature=pki_sign(f2_sk, str(1).encode()),
        )
    )
    await fog._on_consensus_message(msg, "evil")
    assert fog._vc_sent == set()
    assert fog._pending_vc == {}


@pytest.mark.asyncio
async def test_vc_quorum_applies_next_leader_once():
    fog = _mkfog(peers=["fog2", "fog3"], n=3, f=1)
    for peer in ("fog2", "fog3"):
        fog._peer_queues[peer] = asyncio.Queue()
    f2_sk, f2_vk = generate_keypair()
    f3_sk, f3_vk = generate_keypair()
    fog.set_vk("fog2", f2_vk)
    fog.set_vk("fog3", f3_vk)
    from proto import hfedchain_pb2

    def vc_msg(node_id, sk):
        return hfedchain_pb2.ConsensusMessage(
            view_change=hfedchain_pb2.ViewChangeMessage(
                node_id=node_id, round=1, new_view=2, highest_qc=b"",
                signature=pki_sign(sk, str(1).encode()),
            )
        )

    await fog._on_consensus_message(vc_msg("fog2", f2_sk), "fog2")
    assert 1 in fog._vc_applied
    # Task 6 Fix 4: VC leader is derived deterministically from
    # (round, canonical list) — no shared mutable view state is advanced.
    assert fog.view_change.current_view == 1
    assert fog._vc_leader(1) == "fog3"

    await fog._on_consensus_message(vc_msg("fog3", f3_sk), "fog3")
    assert fog.view_change.current_view == 1
    assert fog._vc_applied == {1}, "VC must be applied at most once per round"


@pytest.mark.asyncio
async def test_qc_watch_sends_view_change_on_timeout():
    fog = _mkfog(peers=["fog2"], n=3, f=1, vote_timeout_s=0.05)
    fog._peer_queues["fog2"] = asyncio.Queue()
    fog._running = True

    fog._qc_watchers[7] = asyncio.create_task(fog._qc_watch(7))
    await asyncio.sleep(0.3)

    assert 7 in fog._vc_sent


@pytest.mark.asyncio
async def test_qc_watch_stops_when_round_completes():
    fog = _mkfog(peers=["fog2"], n=3, f=1, vote_timeout_s=0.05)
    fog._peer_queues["fog2"] = asyncio.Queue()
    fog._running = True

    fog._qc_watchers[7] = asyncio.create_task(fog._qc_watch(7))
    fog._mark_round_completed(7)
    await asyncio.sleep(0.3)

    assert 7 not in fog._vc_sent
    assert 7 in fog._completed_rounds


# ----------------------------------------------------------------------
# Task 6 Fix 5 — re-drive respects _round_in_flight + re-triggers backlog
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_re_drive_blocked_while_round_in_flight():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    fog.engine._last_proposal = _mkblock(round_num=1)
    fog._delta_by_round[1] = (b"\x01" * 4, 1)
    fog._round_in_flight = True
    fog._pending_votes = ["sentinel"]

    assert await fog.re_drive_existing_round() is False, (
        "a re-drive must not run while a round is in flight"
    )
    assert fog._round_in_flight is True, "guard must remain its caller's"
    assert fog._pending_votes == ["sentinel"], (
        "a blocked re-drive must not reset the in-flight round's votes"
    )


@pytest.mark.asyncio
async def test_re_drive_reschedules_backlog_after_drive():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    fog.engine._last_proposal = _mkblock(round_num=1)
    fog._delta_by_round[1] = (b"\x01" * 4, 2)
    fog._pending_gradients = ["g1"]  # >= threshold: backlog present

    calls = []

    async def fake_run():
        calls.append(fog._round_in_flight)

    fog._run_round = fake_run

    async def cloud_noop(block, delta_bytes, n_devices):
        pass

    fog._send_to_cloud = cloud_noop

    ok = await fog.re_drive_existing_round()
    assert ok is True
    assert fog._round_in_flight is False, "guard released after the drive"
    await asyncio.sleep(0)  # let the scheduled backlog run

    assert calls == [False], (
        "backlog must be re-scheduled after the re-drive releases the guard"
    )


# ----------------------------------------------------------------------
# Task 6 Fix M3 — re-drive refuses cleanly when no delta is available
# ----------------------------------------------------------------------

async def test_re_drive_refuses_when_no_delta_available(caplog):
    import logging

    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    fog.engine._last_proposal = _mkblock(round_num=1)
    fog._delta_by_round.clear()
    fog._last_delta_w = None
    fog._last_n_devices = 0

    drives = []
    clouds = []

    async def fake_drive(*args, **kwargs):
        drives.append(args)
        return True

    async def fake_cloud(*args):
        clouds.append(args)

    fog._drive_consensus = fake_drive
    fog._send_to_cloud = fake_cloud

    with caplog.at_level(logging.WARNING, logger="services.fog_service"):
        result = await fog.re_drive_existing_round()

    assert result is False, "a re-drive without any delta must refuse"
    assert drives == [], "_drive_consensus must not run without a delta"
    assert clouds == [], "_send_to_cloud must not run without a delta"
    assert "No delta available for round 1" in caplog.text
    assert "cannot re-drive" in caplog.text
    assert fog._round_in_flight is False, "guard must not be left set"
