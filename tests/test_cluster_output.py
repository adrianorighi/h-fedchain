"""ClusterOutput / delta-bytes tests (Task 5).

Covers:
1. Real float32 delta bytes committed to the block (gradient_hash = sha256(delta))
2. _build_cluster_output field correctness (cluster_id, n_devices, delta_w)
3. Defaults on a fresh fog (empty delta, n_devices == 0)
4. Delta = mean over Multi-Krum selected gradients (deterministic planted set)
5. Block accepted_devices/rejected_devices = Multi-Krum selected vs rejected
6. Delta is set before consensus runs (even when consensus fails)
7. Delta bound to the round being sent (captured before consensus awaits)
7b. Binding survives mid-flight instance corruption during consensus
8. Aggregation ValueError keeps pending gradients (no escape, no loss)
9. Mixed-size gradient for the same round rejected at ingress
10. Sub-threshold run drops stale rounds but retains valid gradients
11. _run_round re-entrancy guard (second caller returns immediately)
12. Backlog arriving while a run stalls is rescheduled after the run
13. _delta_by_round recorded per round and window-pruned (Task 6 Fix 1)
14. Re-drive ships the round's own delta even when instance state was
    overwritten by a later round (Task 6 Fix 1)
"""

import asyncio
import hashlib
import logging

import numpy as np

from core.pki import (
    generate_keypair,
    gradient_signed_message,
    sign as pki_sign,
)
from hfc_types.block import Block
from hfc_types.messages import Gradient, GradientWithProof
from services.fog_service import FogService


def _mkfog(**kwargs):
    sk, vk = generate_keypair()
    defaults = dict(
        node_id="fog1", sk=sk, vk=vk, peers=[], n=5, f=1,
        grpc_port=0, cloud_address="127.0.0.1:1",
    )
    defaults.update(kwargs)
    fog = FogService(**defaults)
    fog._mqtt_enabled = False
    return fog


def _mkgrad(node_id, round_num, sk, data):
    data = list(data)
    sig = pki_sign(sk, gradient_signed_message(node_id, round_num, data))
    return GradientWithProof(gradient=Gradient(
        node_id=node_id, round=round_num, data=data, signature=sig))


def _mkblock(round_num=1):
    return Block(
        round=round_num, gradient_hash=b"gh", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=0.0, prev_hash=b"\x00" * 32,
    )


def _planted_gradients():
    sk, _ = generate_keypair()
    return [
        _mkgrad("d0", 1, sk, [0.0] * 10),
        _mkgrad("d1", 1, sk, [0.1] * 10),
        _mkgrad("d2", 1, sk, [0.2] * 10),
        _mkgrad("d3", 1, sk, [0.3] * 10),
        _mkgrad("d4", 1, sk, [100.0] * 10),  # outlier, Multi-Krum must drop
    ]


def _expected_delta(planted, selected):
    return np.mean(
        np.stack([
            np.asarray(planted[i].gradient.data, dtype=np.float32)
            for i in selected
        ]),
        axis=0,
    ).astype(np.float32).tobytes()


# ----------------------------------------------------------------------
# 1 — real delta bytes committed to the block
# ----------------------------------------------------------------------

async def test_mean_delta_bytes():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()
    planted = [_mkgrad("d0", 1, sk, [i * 0.5 for i in range(10)])]
    fog._pending_gradients = list(planted)

    await fog._run_round()

    expected = _expected_delta(planted, [0])
    assert fog._last_delta_w == expected
    assert fog._last_n_devices == 1
    assert fog.ledger.get_height() == 1
    assert (
        fog.ledger._entries[-1].block.gradient_hash
        == hashlib.sha256(fog._last_delta_w).digest()
    )


# ----------------------------------------------------------------------
# 2 + 3 — _build_cluster_output fields and defaults
# ----------------------------------------------------------------------

async def test_build_cluster_output():
    fog = _mkfog(cluster_id="c7")
    known = bytes(range(40))

    output = fog._build_cluster_output(_mkblock(round_num=9), known, 3)

    assert output.cluster_id == "c7"
    assert output.round == 9
    assert output.delta_w == known
    assert output.n_devices == 3
    assert output.gradient_hash == b"gh"  # block.gradient_hash, for cloud check


async def test_build_cluster_output_defaults_empty_delta():
    fog = _mkfog()

    output = fog._build_cluster_output(_mkblock(), b"", 0)

    assert output.delta_w == b""
    assert output.n_devices == 0


# ----------------------------------------------------------------------
# 4 + 5 — delta is mean of selected; block carries selected vs rejected
# ----------------------------------------------------------------------

async def test_multi_node_delta_is_mean_of_selected():
    fog = _mkfog(n=5, f=1, gradient_threshold=1)
    planted = _planted_gradients()
    fog._pending_gradients = list(planted)

    await fog._run_round()

    selected = fog.multikrum.select(
        [np.array(g.gradient.data) for g in planted], fog.f)
    assert selected == [0, 1, 2, 3], "outlier (d4) must be rejected by Multi-Krum"
    assert fog._last_delta_w == _expected_delta(planted, selected)
    assert fog._last_n_devices == len(selected)


async def test_block_accepted_rejected_lists():
    fog = _mkfog(n=1, f=1, gradient_threshold=1)
    planted = _planted_gradients()
    fog._pending_gradients = list(planted)

    await fog._run_round()

    selected = fog.multikrum.select(
        [np.array(g.gradient.data) for g in planted], fog.f)
    block = fog.ledger._entries[-1].block
    assert block.accepted_devices == [
        planted[i].gradient.node_id for i in selected
    ]
    assert block.rejected_devices == [
        g.gradient.node_id for i, g in enumerate(planted)
        if i not in selected
    ]


# ----------------------------------------------------------------------
# 6 — delta computed before consensus (consensus itself may fail)
# ----------------------------------------------------------------------

async def test_delta_set_before_consensus_runs():
    fog = _mkfog(
        peers=["fog2", "fog3"], n=3, f=1,
        gradient_threshold=1, vote_timeout_s=0.2,
    )
    sk, _ = generate_keypair()
    planted = [_mkgrad("d0", 0, sk, [float(i) for i in range(10)])]
    fog._pending_gradients = list(planted)

    await fog._run_round()

    assert fog.ledger.get_height() == 0, "peers are offline, no commit expected"
    assert fog._last_delta_w == _expected_delta(planted, [0])
    assert fog._last_n_devices == 1


# ----------------------------------------------------------------------
# 7 — delta bound to the round being sent (Fix 1)
# ----------------------------------------------------------------------

async def test_sent_delta_matches_committed_block():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()
    fog._pending_gradients = [_mkgrad("d0", 1, sk, [i * 0.5 for i in range(10)])]

    captured = []

    async def recorder(block, delta_bytes, n_devices):
        captured.append((block, delta_bytes, n_devices))

    fog._send_to_cloud = recorder
    await fog._run_round()

    assert len(captured) == 1, "expected exactly one cloud submission"
    block, delta_bytes, n_devices = captured[0]
    assert hashlib.sha256(delta_bytes).digest() == block.gradient_hash
    assert n_devices == 1
    assert fog.ledger.get_height() == 1


# ----------------------------------------------------------------------
# 7b — binding survives instance-state corruption during consensus
# ----------------------------------------------------------------------

async def test_binding_survives_midflight_instance_corruption():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()
    fog._pending_gradients = [_mkgrad("d0", 1, sk, [i * 0.5 for i in range(10)])]

    captured = []

    async def recorder(block, delta_bytes, n_devices):
        captured.append((block, delta_bytes, n_devices))

    fog._send_to_cloud = recorder

    # Corrupt the instance state at an await point inside consensus,
    # after the round's delta was captured but before the cloud send.
    original_start_round = fog.engine.start_round

    async def corrupting_start_round(*args, **kwargs):
        result = await original_start_round(*args, **kwargs)
        fog._last_delta_w = b"corrupted" * 8
        fog._last_n_devices = 999
        return result

    fog.engine.start_round = corrupting_start_round

    await fog._run_round()

    assert fog._last_delta_w == b"corrupted" * 8, "corruption hook must have run"
    assert len(captured) == 1, "expected exactly one cloud submission"
    block, delta_bytes, n_devices = captured[0]
    assert delta_bytes != b"corrupted" * 8, (
        "send must use the bound delta, not instance state at send time"
    )
    assert hashlib.sha256(delta_bytes).digest() == block.gradient_hash
    assert n_devices == 1, "send must use the bound device count, not 999"
    assert fog.ledger.get_height() == 1


# ----------------------------------------------------------------------
# 8 — aggregation ValueError keeps pending gradients (Fix 3)
# ----------------------------------------------------------------------

async def test_multikrum_value_error_keeps_pending(caplog):
    fog = _mkfog(n=3, f=1, gradient_threshold=3)
    sk, _ = generate_keypair()
    planted = [
        _mkgrad("d0", 1, sk, [1.0] * 10),
        _mkgrad("d1", 1, sk, [2.0] * 10),
        _mkgrad("d2", 1, sk, [3.0] * 10),  # 3f=3 >= n=3 -> ValueError
    ]
    fog._pending_gradients = list(planted)

    with caplog.at_level(logging.ERROR):
        await fog._run_round()  # must not raise

    assert len(fog._pending_gradients) == 3, "valid gradients must be restored"
    assert fog.ledger.get_height() == 0, "no round must be committed"
    errors = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.ERROR]
    assert errors, "expected an error log for the failed aggregation"
    assert "round 1" in errors[0]
    assert "f=1" in errors[0]


# ----------------------------------------------------------------------
# 9 — mixed-size gradient for the same round rejected at ingress (Fix 4)
# ----------------------------------------------------------------------

async def test_mixed_size_gradient_same_round_rejected():
    fog = _mkfog(gradient_threshold=10)
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    fog.set_vk("d1", vk)

    await fog.on_gradient_received(_mkgrad("d0", 2, sk, [0.0] * 10))
    assert len(fog._pending_gradients) == 1

    await fog.on_gradient_received(_mkgrad("d1", 2, sk, [0.0] * 8))
    assert len(fog._pending_gradients) == 1, "size-mismatched gradient must be rejected"
    assert fog._pending_gradients[0].gradient.node_id == "d0"


# ----------------------------------------------------------------------
# 10 — sub-threshold run drops stale rounds, retains valid (Fix 3)
# ----------------------------------------------------------------------

async def test_sub_threshold_run_retains_valid_drops_stale():
    fog = _mkfog(gradient_threshold=3)
    sk, _ = generate_keypair()
    fog._pending_gradients = [
        _mkgrad("d0", 1, sk, [1.0] * 10),  # stale round
        _mkgrad("d1", 2, sk, [2.0] * 10),
        _mkgrad("d2", 2, sk, [3.0] * 10),
    ]

    await fog._run_round()

    rounds = sorted(item.gradient.round for item in fog._pending_gradients)
    assert rounds == [2, 2], (
        "stale round must be dropped, valid round-2 gradients retained, "
        f"got {rounds}"
    )


# ----------------------------------------------------------------------
# 11 — _run_round re-entrancy guard (Fix 2)
# ----------------------------------------------------------------------

async def test_reentrancy_guard_blocks_second_run():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()
    planted = [_mkgrad("d0", 1, sk, [1.0] * 10)]
    fog._pending_gradients = list(planted)
    fog._round_in_flight = True

    await fog._run_round()

    assert fog._round_in_flight is True, "guard must be left to its caller"
    assert fog._pending_gradients == planted, "pending must be untouched"
    assert fog.ledger.get_height() == 0, "no round must run while in flight"


# ----------------------------------------------------------------------
# 12 — backlog arriving while a run stalls is rescheduled (Fix 2)
# ----------------------------------------------------------------------

async def test_backlog_during_stalled_run_is_rescheduled():
    from hfc_types.crypto import SnarkProof

    fog = _mkfog(n=1, f=0, gradient_threshold=1, variant="snark")
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    fog.set_vk("d1", vk)

    g1 = _mkgrad("d0", 2, sk, [1.0] * 10)
    g1.snark_proof = SnarkProof(proof_bytes=b"invalid", public_inputs={})
    fog._pending_gradients = [g1]

    async def yielding_verify(*args, **kwargs):
        await asyncio.sleep(0)  # yield: lets a gradient arrive mid-attempt
        return False            # proof invalid -> this attempt stalls

    fog.snark_verifier.verify = yielding_verify

    run = asyncio.create_task(fog._run_round())
    await asyncio.sleep(0)  # run is now suspended inside yielding_verify

    # Arrival meets the threshold but hits the in-flight guard and returns;
    # the in-flight attempt does not see it (already snapshotted) and stalls.
    await fog.on_gradient_received(_mkgrad("d1", 2, sk, [2.0] * 10))
    await run
    await asyncio.sleep(0.05)  # let the rescheduled run finish

    assert fog._pending_gradients == [], "backlog must not be left unprocessed"
    assert fog.ledger.get_height() == 1, "rescheduled run must commit a block"


# ----------------------------------------------------------------------
# 13 — _delta_by_round recorded per round and window-pruned (Task 6 Fix 1)
# ----------------------------------------------------------------------

async def test_delta_by_round_recorded_per_round():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()
    fog._pending_gradients = [_mkgrad("d0", 1, sk, [i * 0.5 for i in range(10)])]

    await fog._run_round()

    assert 1 in fog._delta_by_round, "round delta must be recorded for re-drives"
    delta_bytes, n_devices = fog._delta_by_round[1]
    assert n_devices == 1
    assert hashlib.sha256(delta_bytes).digest() == (
        fog.ledger._entries[-1].block.gradient_hash
    )


async def test_delta_by_round_window_stays_bounded():
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    sk, _ = generate_keypair()

    fog._pending_gradients = [_mkgrad("d0", 1, sk, [1.0] * 10)]
    await fog._run_round()
    assert 1 in fog._delta_by_round

    # A later round must prune the old entry (small window: keep recent only).
    fog._pending_gradients = [_mkgrad("d0", 5, sk, [2.0] * 10)]
    await fog._run_round()
    assert 5 in fog._delta_by_round
    assert 1 not in fog._delta_by_round, "stale round deltas must be pruned"


# ----------------------------------------------------------------------
# 14 — re-drive ships the round's own delta (Task 6 Fix 1)
# ----------------------------------------------------------------------

async def test_re_drive_ships_round_own_delta_not_instance_state():
    """A re-drive after a later round overwrote _last_delta_w must still
    submit THIS round's delta (end-to-end integrity with cloud check)."""
    fog = _mkfog(n=1, f=0, gradient_threshold=1)
    own_delta = np.arange(10, dtype=np.float32).tobytes()
    block = Block(
        round=1, gradient_hash=hashlib.sha256(own_delta).digest(),
        qc_commit=None, stark_proof=None, accepted_devices=[],
        rejected_devices=[], timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    fog.engine._last_proposal = block
    fog._delta_by_round[1] = (own_delta, 3)
    # Later round(s) clobbered the shared instance state.
    fog._last_delta_w = b"junk" * 16
    fog._last_n_devices = 999

    captured = []

    async def recorder(block_arg, delta_bytes, n_devices):
        captured.append((block_arg, delta_bytes, n_devices))

    fog._send_to_cloud = recorder

    ok = await fog.re_drive_existing_round()

    assert ok is True, "re-drive of an uncompleted round must succeed"
    assert len(captured) == 1, "expected exactly one cloud submission"
    sent_block, delta_bytes, n_devices = captured[0]
    assert delta_bytes == own_delta, (
        "re-drive must ship the round's own delta, not the corrupted "
        "instance state"
    )
    assert n_devices == 3, "re-drive must ship the round's own device count"
    assert hashlib.sha256(delta_bytes).digest() == sent_block.gradient_hash
