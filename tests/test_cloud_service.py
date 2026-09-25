import asyncio
import hashlib
import logging
import pickle

import numpy as np
import pytest

from services.cloud_service import CloudService


class _StubMqtt:
    """Records publish calls; never touches a broker."""

    def __init__(self):
        self.published = []
        self.started = False
        self.stopped = False

    def publish(self, topic, payload, retain=False):
        self.published.append({"topic": topic, "payload": payload,
                               "retain": retain})
        return True

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def _install_stub_mqtt(cloud) -> _StubMqtt:
    stub = _StubMqtt()
    cloud.mqtt = stub
    return stub


def test_cloud_service_init():
    service = CloudService()
    assert service is not None
    assert service.n_expected_clusters == 1
    assert service.grpc_port == 50052
    assert service.learning_rate == 0.01
    assert service.converged is False
    assert service.global_weights is None
    assert service._mqtt_enabled is True
    assert service._last_global_hash == b"\x00" * 32


# ----------------------------------------------------------------------
# ClusterOutput.gradient_hash end-to-end integrity check (Task 6 Fix 1)
# ----------------------------------------------------------------------

def _output(cluster_id, round_num, delta_w, gradient_hash=b"", n_devices=4):
    from proto import hfedchain_pb2

    return hfedchain_pb2.ClusterOutput(
        cluster_id=cluster_id,
        round=round_num,
        delta_w=delta_w,
        gradient_hash=gradient_hash,
        n_devices=n_devices,
    )


@pytest.mark.asyncio
async def test_cloud_drops_output_with_mismatched_gradient_hash(caplog):
    """A payload whose sha256(delta_w) differs from the committed
    gradient_hash must be dropped for the round (not aggregated)."""
    cloud = CloudService(n_expected_clusters=1)
    genuine = np.ones(4, dtype=np.float32).tobytes()
    tampered = np.zeros(4, dtype=np.float32).tobytes()
    out = _output(
        "c1", 1, tampered,
        gradient_hash=hashlib.sha256(genuine).digest(),
    )

    with caplog.at_level(logging.WARNING):
        resp = await cloud.submit_cluster_output(out)

    assert resp.n_active_clusters == 0, "mismatched output must not aggregate"
    assert not cloud._round_outputs.get(1), (
        "mismatched output must not be collected for the round"
    )
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("gradient_hash" in w for w in warnings), (
        f"expected a gradient_hash mismatch warning, got: {warnings}"
    )


@pytest.mark.asyncio
async def test_cloud_accepts_output_with_matching_gradient_hash():
    cloud = CloudService(n_expected_clusters=1)
    genuine = np.ones(4, dtype=np.float32).tobytes()
    out = _output(
        "c1", 1, genuine,
        gradient_hash=hashlib.sha256(genuine).digest(),
        n_devices=5,
    )

    resp = await cloud.submit_cluster_output(out)

    assert resp.n_active_clusters == 1
    assert resp.delta_w_inter == genuine


@pytest.mark.asyncio
async def test_cloud_skips_check_when_gradient_hash_empty(caplog):
    """proto3 default (empty) gradient_hash skips the integrity check so
    legacy/fog-less producers keep working (lenient, logged at debug)."""
    cloud = CloudService(n_expected_clusters=1)
    genuine = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, genuine, n_devices=2)

    with caplog.at_level(logging.DEBUG):
        resp = await cloud.submit_cluster_output(out)

    assert resp.n_active_clusters == 1
    debugs = [r.getMessage() for r in caplog.records
              if r.levelno < logging.WARNING]
    assert any("gradient_hash" in d for d in debugs), (
        f"expected a debug log for the skipped check, got: {debugs}"
    )


@pytest.mark.asyncio
async def test_mismatched_cluster_not_counted_toward_expected():
    """Dropped output must not count toward n_expected_clusters/weights."""
    cloud = CloudService(n_expected_clusters=2)
    genuine = np.ones(4, dtype=np.float32).tobytes()
    tampered = np.full(4, 7.0, dtype=np.float32).tobytes()

    bad = _output(
        "c1", 1, tampered,
        gradient_hash=hashlib.sha256(genuine).digest(),
    )
    await cloud.submit_cluster_output(bad)
    assert cloud._round_outputs.get(1) is None or "c1" not in cloud._round_outputs[1]

    good = _output(
        "c2", 1, genuine,
        gradient_hash=hashlib.sha256(genuine).digest(),
    )
    resp = await cloud.submit_cluster_output(good)
    # only one valid output of two expected: no aggregation yet
    assert resp.n_active_clusters == 0
    assert set(cloud._round_outputs[1]) == {"c2"}


# ----------------------------------------------------------------------
# Task 7: pi_inter, validation gate, WORM, MQTT global broadcast
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aggregate_round_returns_pi_inter_and_broadcasts_delta_hash():
    """Full aggregation: non-empty pi_inter in the response and WORM, and a
    retained models/global broadcast whose delta_hash binds the delta and
    whose schema/learning_rate make the payload self-describing (edges apply
    w -= lr * delta with exactly this lr, so the mirror stays consistent)."""
    cloud = CloudService(n_expected_clusters=1)
    stub = _install_stub_mqtt(cloud)
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output(
        "c1", 1, delta,
        gradient_hash=hashlib.sha256(delta).digest(),
        n_devices=5,
    )

    resp = await cloud.submit_cluster_output(out)

    assert resp.delta_w_inter == delta
    assert resp.pi_inter, "response must carry a non-empty pi_inter proof"
    assert resp.n_active_clusters == 1
    assert resp.round == 1

    entry = cloud.worm.get_entry(1)
    assert entry is not None, "aggregated round must be persisted to WORM"
    assert entry.pi_inter is not None, "WORM entry must carry pi_inter"
    assert entry.pi_inter.proof_bytes == resp.pi_inter
    assert entry.accepted is True

    assert len(stub.published) == 1, "aggregated round must broadcast once"
    msg = stub.published[0]
    assert msg["topic"] == "models/global"
    assert msg["retain"] is True
    payload = pickle.loads(msg["payload"])
    assert payload["round"] == 1
    assert payload["delta_w_inter"] == delta
    assert payload["delta_hash"] == hashlib.sha256(delta).digest()
    assert "model_hash" not in payload, (
        "model_hash elsewhere means sha256(weights); the delta must not "
        "collide with that meaning"
    )
    assert payload["learning_rate"] == pytest.approx(0.01)
    assert payload["schema"] == 1
    assert payload["pi_inter"] == resp.pi_inter
    assert payload["n_active_clusters"] == 1
    assert payload["converged"] is False

    assert cloud._last_global_hash != b"\x00" * 32, (
        "hash chain must advance past the genesis prev_hash"
    )


@pytest.mark.asyncio
async def test_first_round_gate_initializes_zero_mirror():
    """The validation gate must see a zeroed mirror on round 1 and the
    accepted mirror must be the lr-scaled update under key "global"."""
    cloud = CloudService(n_expected_clusters=1)
    _install_stub_mqtt(cloud)
    calls = []
    orig_validate = cloud.validation_gate.validate

    def spy(w_old, w_new, loss=None):
        calls.append((w_old, w_new))
        return orig_validate(w_old, w_new, loss)

    cloud.validation_gate.validate = spy
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    await cloud.submit_cluster_output(out)

    assert len(calls) == 1
    w_old, w_new = calls[0]
    assert w_old is not None and set(w_old) == {"global"}, (
        "gate must receive the initialized mirror, not None"
    )
    assert w_old["global"].dtype == np.float32
    np.testing.assert_array_equal(w_old["global"], np.zeros(4, dtype=np.float32))

    assert set(cloud.global_weights) == {"global"}
    assert cloud.global_weights["global"].dtype == np.float32
    np.testing.assert_allclose(
        cloud.global_weights["global"],
        -0.01 * np.ones(4, dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_gate_rejection_records_worm_entry_but_still_broadcasts(caplog):
    """A gate rejection must still broadcast the delta so edge devices keep
    progressing (liveness) and must be recorded in WORM with accepted=False
    (compliance) — without updating the cloud's mirror."""
    cloud = CloudService(n_expected_clusters=1)
    stub = _install_stub_mqtt(cloud)
    cloud.validation_gate.delta_conv = 0.0  # delta_norm (0.02) > 0 → reject
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    with caplog.at_level(logging.WARNING):
        resp = await cloud.submit_cluster_output(out)

    assert cloud.worm.get_height() == 1, "rejections must be compliance records"
    entry = cloud.worm.get_entry(1)
    assert entry is not None
    assert entry.accepted is False
    assert entry.delta_w_inter == delta
    assert entry.pi_inter is not None
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("delta_norm" in w for w in warnings), (
        f"expected a rejection warning carrying the reason, got: {warnings}"
    )
    assert len(stub.published) == 1, (
        "rejected round must still broadcast for edge liveness"
    )
    payload = pickle.loads(stub.published[0]["payload"])
    assert payload["delta_w_inter"] == delta
    assert resp.delta_w_inter == delta
    assert resp.n_active_clusters == 1
    np.testing.assert_allclose(
        cloud.global_weights["global"],
        np.zeros(4, dtype=np.float32),
        err_msg="a rejected round must not move the cloud mirror",
    )


# ----------------------------------------------------------------------
# Task 7 review fixes: gate-rejected rounds must not re-aggregate/fork
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gate_rejected_round_retry_is_idempotent(caplog):
    """A gate-rejected round leaves no *accepted* WORM record, but a fog
    retry must still be a no-op: no second aggregation, no second broadcast,
    no chain advance — and the rejection itself is recorded in WORM."""
    cloud = CloudService(n_expected_clusters=1)
    stub = _install_stub_mqtt(cloud)
    cloud.validation_gate.delta_conv = 0.0  # delta_norm (0.02) > 0 → reject
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    with caplog.at_level(logging.WARNING):
        resp1 = await cloud.submit_cluster_output(out)
    assert resp1.delta_w_inter == delta
    hash_after_first = cloud._last_global_hash
    assert hash_after_first != b"\x00" * 32, "first pass must advance the chain"

    agg_calls = []
    orig_agg = cloud._aggregate_round

    async def spy_agg(round_num):
        agg_calls.append(round_num)
        return await orig_agg(round_num)

    cloud._aggregate_round = spy_agg

    resp2 = await cloud.submit_cluster_output(out)

    assert agg_calls == [], (
        "a gate-rejected round must not re-aggregate on a fog retry"
    )
    assert cloud._last_global_hash == hash_after_first, (
        "a retry must not advance the chain a second time"
    )
    assert len(stub.published) == 1, "a retry must not broadcast again"
    assert resp2.delta_w_inter == resp1.delta_w_inter
    assert resp2.pi_inter == resp1.pi_inter
    assert resp2.n_active_clusters == resp1.n_active_clusters == 1
    assert resp2.round == 1
    assert 1 not in cloud._round_outputs

    entry = cloud.worm.get_entry(1)
    assert entry is not None, "the rejection must be recorded in WORM"
    assert entry.accepted is False
    assert entry.delta_w_inter == delta


@pytest.mark.asyncio
async def test_cross_round_prev_hash_linkage():
    """Round N+1's inter block must hash-chain onto round N's block — even
    when round N was gate-rejected and the fog retried it (a retry must not
    fork the chain into a second, differently-linked block)."""
    cloud = CloudService(n_expected_clusters=1, mqtt_enabled=False)
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    blocks = []
    orig_gen = cloud.stark_prover.generate_proof

    async def spy_gen(block):
        blocks.append(block)
        return await orig_gen(block)

    cloud.stark_prover.generate_proof = spy_gen

    cloud.validation_gate.delta_conv = 0.0  # round 1 rejected
    await cloud.submit_cluster_output(out)
    round1_hash = cloud._last_global_hash
    assert round1_hash != b"\x00" * 32

    # Fog retry of the rejected round 1 must not fork the chain.
    await cloud.submit_cluster_output(out)
    assert cloud._last_global_hash == round1_hash, (
        "retrying a rejected round must not advance the chain"
    )
    assert [b.round for b in blocks] == [1], (
        "exactly one inter block must exist for round 1"
    )

    cloud.validation_gate.delta_conv = 100.0  # round 2 accepted
    await cloud.submit_cluster_output(
        _output("c1", 2, delta, gradient_hash=hashlib.sha256(delta).digest())
    )

    assert [b.round for b in blocks] == [1, 2]
    assert blocks[1].prev_hash == blocks[0].hash, (
        "round 2 must link to round 1's block hash"
    )
    assert blocks[1].prev_hash == round1_hash


@pytest.mark.asyncio
async def test_broadcast_warns_when_publish_reports_undelivered(caplog):
    """A publish() that reports a dropped message (disconnected client) must
    surface a warning naming the round — silent publish loss is a bug."""
    class _UndeliveredMqtt(_StubMqtt):
        def publish(self, topic, payload, retain=False):
            super().publish(topic, payload, retain=retain)
            return False

    cloud = CloudService(n_expected_clusters=1)
    cloud.mqtt = _UndeliveredMqtt()
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    with caplog.at_level(logging.WARNING):
        resp = await cloud.submit_cluster_output(out)

    assert resp.n_active_clusters == 1
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any(
        "not delivered" in w and "round 1" in w for w in warnings
    ), f"expected an undelivered-broadcast warning, got: {warnings}"


@pytest.mark.asyncio
async def test_worm_persistence_across_store_instances(tmp_path):
    """worm_db_path must survive into a fresh WormStore reading the same file."""
    from core.ledger.worm_store import HAS_SQLITE, WormStore

    if not HAS_SQLITE:
        pytest.skip("SQLite not available")

    db_path = str(tmp_path / "worm.db")
    cloud = CloudService(
        n_expected_clusters=1, mqtt_enabled=False, worm_db_path=db_path,
    )
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 7, delta, gradient_hash=hashlib.sha256(delta).digest())

    resp = await cloud.submit_cluster_output(out)

    reopened = WormStore(db_path=db_path)
    entry = reopened.get_entry(7)
    assert entry is not None, "entry must be readable from a fresh store"
    assert entry.delta_w_inter == delta
    assert entry.pi_inter is not None
    assert entry.pi_inter.proof_bytes == resp.pi_inter
    assert entry.n_active_clusters == 1


@pytest.mark.asyncio
async def test_idempotent_resubmit_returns_stored_output():
    """A fog retry of an already-aggregated round returns the stored
    response without re-collecting, re-aggregating or re-appending."""
    cloud = CloudService(n_expected_clusters=1)
    stub = _install_stub_mqtt(cloud)
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    resp1 = await cloud.submit_cluster_output(out)
    assert resp1.pi_inter

    agg_calls = []
    orig_agg = cloud._aggregate_round

    async def spy_agg(round_num):
        agg_calls.append(round_num)
        return await orig_agg(round_num)

    cloud._aggregate_round = spy_agg

    resp2 = await cloud.submit_cluster_output(out)

    assert agg_calls == [], "a finished round must not be re-aggregated"
    assert resp2.delta_w_inter == resp1.delta_w_inter
    assert resp2.pi_inter == resp1.pi_inter
    assert resp2.n_active_clusters == resp1.n_active_clusters == 1
    assert resp2.round == 1
    assert 1 not in cloud._round_outputs
    assert cloud.worm.get_height() == 1
    assert len(stub.published) == 1, "retry must not broadcast twice"


@pytest.mark.asyncio
async def test_mqtt_disabled_skips_publish():
    """mqtt_enabled=False must not attempt any broadcast."""
    cloud = CloudService(n_expected_clusters=1, mqtt_enabled=False)
    stub = _install_stub_mqtt(cloud)
    delta = np.ones(4, dtype=np.float32).tobytes()
    out = _output("c1", 1, delta, gradient_hash=hashlib.sha256(delta).digest())

    resp = await cloud.submit_cluster_output(out)

    assert resp.n_active_clusters == 1
    assert resp.pi_inter
    assert stub.published == [], "disabled MQTT must not publish"


@pytest.mark.asyncio
async def test_weight_normalization_excludes_empty_delta_clusters():
    """Weights are normalized over clusters with a non-empty delta only, so
    their sum stays 1 and an empty-delta cluster does not shrink the rest."""
    cloud = CloudService(
        n_expected_clusters=2, mqtt_enabled=False,
    )
    delta = np.full(4, 3.0, dtype=np.float32).tobytes()
    cloud._round_outputs[5] = {
        "c1": {"delta_w": b"", "n_devices": 10},
        "c2": {"delta_w": delta, "n_devices": 10},
    }

    resp = await cloud._aggregate_round(5)

    assert resp.n_active_clusters == 2
    assert resp.delta_w_inter == delta, (
        "the only included delta must be weighted 1.0 (weights sum to 1)"
    )


# ----------------------------------------------------------------------
# MQTT lifecycle (start/stop)
# ----------------------------------------------------------------------

async def _wait_grpc_up(cloud, stub):
    for _ in range(200):
        if cloud._grpc_server is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("cloud gRPC server never came up")


@pytest.mark.asyncio
async def test_cloud_start_and_stop_manage_mqtt_lifecycle():
    """start() must bring MQTT up (when enabled) and stop() must tear it down."""
    cloud = CloudService(grpc_port=0)
    stub = _StubMqtt()
    cloud.mqtt = stub

    task = asyncio.create_task(cloud.start())
    try:
        await _wait_grpc_up(cloud, stub)  # MQTT is attempted before gRPC
        assert stub.started, "start() must start MQTT when enabled"
    finally:
        await cloud.stop()
        await task
    assert stub.stopped, "stop() must stop MQTT"


@pytest.mark.asyncio
async def test_cloud_start_skips_mqtt_when_disabled():
    """mqtt_enabled=False must never touch the MQTT client."""
    cloud = CloudService(grpc_port=0, mqtt_enabled=False)
    stub = _StubMqtt()
    cloud.mqtt = stub

    task = asyncio.create_task(cloud.start())
    try:
        await _wait_grpc_up(cloud, stub)
        assert stub.started is False, "disabled MQTT must not start"
    finally:
        await cloud.stop()
        await task
    assert stub.stopped is False, "disabled MQTT must not be stopped"
