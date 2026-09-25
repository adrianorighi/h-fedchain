"""Task 9 — monolithic EdgeRunner: registrations, persistent keys, round loop.

All tests inject a FakeMqttClient — no broker, no network. SNARK is disabled
everywhere (py_ecc pairing may hang in this environment) and the dataset is
always synthetic (``use_dataset=False``).
"""

import asyncio
import hashlib
import pickle

import numpy as np
import pytest

from core.pki import gradient_signed_message, verify as pki_verify
from dataset.model import flatten_weights
from services.edge_runner import EdgeRunner

TOTAL_PARAMS = 768_389  # 12000*64 + 64 + 64*5 + 5
CLUSTER = "c1"


class FakeMqttClient:
    """In-memory stand-in for core.network.mqtt.MqttClient.

    Records every publish (topic, payload, retain), stores subscribe
    handlers, and always reports success. ``on_publish`` lets a test answer
    a gradient batch with a models/global payload mid-``run()``.
    """

    def __init__(self):
        self.published: list[tuple[str, bytes, bool]] = []
        self.handlers: dict[str, list] = {}
        self.started = False
        self.stopped = False
        self.on_publish = None

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def subscribe(self, topic, handler):
        self.handlers.setdefault(topic, []).append(handler)

    def publish(self, topic, payload, retain=False) -> bool:
        self.published.append((topic, payload, retain))
        if self.on_publish is not None:
            self.on_publish(topic, payload)
        return True

    def deliver(self, topic: str, payload: bytes):
        for handler in self.handlers.get(topic, []):
            handler(payload)

    def published_to(self, topic: str) -> list[bytes]:
        return [p for t, p, _ in self.published if t == topic]


def make_runner(tmp_path, **overrides) -> tuple[EdgeRunner, FakeMqttClient]:
    opts = dict(
        device_ids=None,
        num_devices=2,
        cluster_id=CLUSTER,
        mqtt_broker="localhost",
        mqtt_port=1883,
        rounds=2,
        learning_rate=0.01,
        use_snark=False,
        use_dataset=False,
        key_dir=str(tmp_path / "keys"),
        model_timeout_s=5.0,
        dirichlet_alpha=0.5,
        max_records=20,
    )
    opts.update(overrides)
    fake = FakeMqttClient()
    opts["mqtt_client"] = fake
    return EdgeRunner(**opts), fake


def model_payload(round_num, delta: np.ndarray, *, lr=0.01,
                  converged=False, pi_inter=None) -> bytes:
    raw = np.asarray(delta, dtype=np.float32).tobytes()
    return pickle.dumps({
        "schema": 1,
        "round": round_num,
        "delta_w_inter": raw,
        "delta_hash": hashlib.sha256(raw).digest(),
        "learning_rate": lr,
        "pi_inter": pi_inter,
        "n_active_clusters": 1,
        "converged": converged,
    })


def gradient_rounds(fake: FakeMqttClient) -> list[int]:
    return sorted({
        pickle.loads(p).gradient.round
        for t, p, _ in fake.published
        if t == f"gradients/{CLUSTER}"
    })


def auto_answer(fake: FakeMqttClient, deltas: dict[int, np.ndarray],
                converged_round: int | None = None):
    """Deliver a models/global payload the first time round ``r``'s
    gradient batch shows up."""
    delivered: set[int] = set()

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        r = pickle.loads(payload).gradient.round
        if r in delivered or r not in deltas:
            return
        delivered.add(r)
        fake.deliver("models/global", model_payload(
            r, deltas[r], converged=(r == converged_round),
        ))

    fake.on_publish = on_publish


# ----------------------------------------------------------------------
# 1 — setup publishes retained registrations
# ----------------------------------------------------------------------

async def test_setup_publishes_retained_registrations(tmp_path):
    runner, fake = make_runner(tmp_path)
    await runner.setup()

    assert fake.started, "setup() must start the MQTT client"
    assert "models/global" in fake.handlers, (
        "setup() must subscribe to models/global"
    )

    regs = [(pickle.loads(p), retain) for t, p, retain in fake.published
            if t.startswith(f"register/{CLUSTER}/")]
    assert len(regs) == 2, f"expected one registration per device, got {len(regs)}"
    vks = {w.device_id: w.vk for w in runner.workers}
    for reg, retain in regs:
        assert retain is True, "registrations must be retained"
        assert set(reg.keys()) == {"node_id", "vk"}
        assert isinstance(reg["vk"], bytes) and len(reg["vk"]) == 32
        assert reg["vk"] == vks[reg["node_id"]]
    assert {reg["node_id"] for reg, _ in regs} == set(vks)


# ----------------------------------------------------------------------
# 1b — retained registrations survive a fog restart (per-device topics)
# ----------------------------------------------------------------------

async def test_retained_registrations_survive_fog_restart(tmp_path):
    """MQTT retains ONE message per topic: all devices publishing retained
    to register/{cluster} would leave only the last device's vk after a
    broker/fog restart. Each device must own its topic, and replaying ALL
    retained messages into a fresh fog must restore every vk."""
    from core.pki import generate_keypair
    from services.fog_service import FogService

    runner, fake = make_runner(tmp_path)
    await runner.setup()

    retained: dict[str, bytes] = {}
    for t, p, retain in fake.published:
        if t.startswith("register/"):
            assert retain is True, "registrations must be retained"
            retained[t] = p  # broker keeps the LAST payload per topic

    assert len(retained) == 2, (
        "each device must publish retained to its own topic so no "
        f"registration overwrites another; got topics: {sorted(retained)}"
    )
    expected_topics = {f"register/{CLUSTER}/{w.device_id}"
                       for w in runner.workers}
    assert set(retained) == expected_topics, (
        f"expected per-device topics {sorted(expected_topics)}, "
        f"got {sorted(retained)}"
    )

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog1", sk=sk, vk=vk, peers=[], n=1, f=0,
        grpc_port=0, mqtt_enabled=False,
    )
    for payload in retained.values():
        fog._on_mqtt_registration(payload)

    vks = {w.device_id: w.vk for w in runner.workers}
    assert fog._vk_map == vks, (
        "replaying every retained registration into a fresh fog must "
        f"restore ALL vks, got {sorted(fog._vk_map)}"
    )
    await runner.stop()


# ----------------------------------------------------------------------
# 2 — key persistence across setups
# ----------------------------------------------------------------------

async def test_device_keys_persist_across_setups(tmp_path):
    key_dir = tmp_path / "keys"

    r1, _ = make_runner(tmp_path, key_dir=str(key_dir))
    await r1.setup()
    vks1 = [w.vk for w in r1.workers]

    r2, _ = make_runner(tmp_path, key_dir=str(key_dir))
    await r2.setup()
    vks2 = [w.vk for w in r2.workers]

    assert vks1 == vks2, "same key_dir must yield the same device vks"
    assert (key_dir / "d0.key").exists(), "keys must be persisted as files"

    r3, _ = make_runner(tmp_path, key_dir=str(tmp_path / "fresh"))
    await r3.setup()
    vks3 = [w.vk for w in r3.workers]
    assert vks3 != vks1, "a fresh key_dir must yield fresh keys"


# ----------------------------------------------------------------------
# 3 — full round applies the broadcast delta
# ----------------------------------------------------------------------

async def test_round_applies_broadcast_delta(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=2)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    delta = np.linspace(-1.0, 1.0, TOTAL_PARAMS, dtype=np.float32)
    auto_answer(fake, {1: delta, 2: np.zeros(TOTAL_PARAMS, np.float32)},
                converged_round=2)

    await runner.run()

    expected = init_flat - 0.01 * delta.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "w_after == w_init - lr * delta (float32 wire format)"
    )
    assert gradient_rounds(fake) == [1, 2]
    assert runner.rounds_completed == 2
    await runner.stop()


# ----------------------------------------------------------------------
# 4 — tampered delta_hash is skipped
# ----------------------------------------------------------------------

async def test_tampered_delta_hash_is_skipped(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=2)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    d2 = np.full(TOTAL_PARAMS, 0.5, dtype=np.float32)
    delivered: set[int] = set()

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        r = pickle.loads(payload).gradient.round
        if r in delivered:
            return
        delivered.add(r)
        if r == 1:
            raw = np.ones(TOTAL_PARAMS, dtype=np.float32).tobytes()
            fake.deliver("models/global", pickle.dumps({
                "schema": 1, "round": 1,
                "delta_w_inter": raw,
                "delta_hash": b"\x00" * 32,  # tampered
                "learning_rate": 0.01, "pi_inter": None,
                "n_active_clusters": 1, "converged": False,
            }))
        else:
            fake.deliver("models/global", model_payload(r, d2))

    fake.on_publish = on_publish
    await runner.run()

    assert runner.rounds_completed == 2, "the run must continue past round 1"
    expected = init_flat - 0.01 * d2.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "round 1's tampered update must be skipped; only round 2 applied"
    )


# ----------------------------------------------------------------------
# 5 — wrong-size delta (valid hash) → ValueError path, run continues
# ----------------------------------------------------------------------

async def test_wrong_size_delta_is_skipped(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=2)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    d2 = np.full(TOTAL_PARAMS, 0.25, dtype=np.float32)
    delivered: set[int] = set()

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        r = pickle.loads(payload).gradient.round
        if r in delivered:
            return
        delivered.add(r)
        if r == 1:
            fake.deliver("models/global",
                         model_payload(r, np.arange(10, dtype=np.float32)))
        else:
            fake.deliver("models/global", model_payload(r, d2))

    fake.on_publish = on_publish
    await runner.run()

    assert runner.rounds_completed == 2, "a size mismatch must not kill the run"
    expected = init_flat - 0.01 * d2.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "round 1's wrong-size delta must be skipped; only round 2 applied"
    )


# ----------------------------------------------------------------------
# 6 — timeout
# ----------------------------------------------------------------------

async def test_missing_model_raises_timeout(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=1, model_timeout_s=0.3)
    await runner.setup()

    with pytest.raises(TimeoutError, match="not received within"):
        await runner.run()

    await runner.stop()


# ----------------------------------------------------------------------
# 7 — registration payload accepted by the fog's rules
# ----------------------------------------------------------------------

async def test_registration_payload_accepted_by_fog(tmp_path):
    from core.pki import generate_keypair
    from services.fog_service import FogService

    runner, fake = make_runner(tmp_path)
    await runner.setup()

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog1", sk=sk, vk=vk, peers=["fog2", "fog3"],
        n=3, f=1, grpc_port=0, mqtt_enabled=False,
    )
    for t, p, _ in fake.published:
        if t.startswith(f"register/{CLUSTER}/"):
            fog._on_mqtt_registration(p)

    vks = {w.device_id: w.vk for w in runner.workers}
    assert set(fog._vk_map.keys()) == set(vks.keys())
    for device_id, device_vk in vks.items():
        assert fog._vk_map[device_id] == device_vk
    await runner.stop()


# ----------------------------------------------------------------------
# 8 — published gradient signature verifies against the registered vk
# ----------------------------------------------------------------------

async def test_gradient_signature_verifies_against_registered_vk(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=1)
    auto_answer(fake, {1: np.zeros(TOTAL_PARAMS, np.float32)})
    await runner.setup()
    await runner.run()

    grads = fake.published_to(f"gradients/{CLUSTER}")
    assert len(grads) == 2, "one gradient per device per round"
    vks = {w.device_id: w.vk for w in runner.workers}
    for payload in grads:
        g = pickle.loads(payload).gradient
        signed = gradient_signed_message(g.node_id, g.round, g.data)
        assert g.signature is not None
        assert pki_verify(vks[g.node_id], signed, g.signature), (
            f"gradient from {g.node_id} must verify against its registered vk"
        )
    await runner.stop()


# ----------------------------------------------------------------------
# Fix 1 — adversarial devices keep real signatures + adv_ naming
# ----------------------------------------------------------------------

async def test_adversarial_devices_reach_fog_pki_gate(tmp_path):
    """Adversarial devices poison DATA only: their gradients must carry the
    real device signature (so the fog's PKI gate lets them through to
    Multi-Krum) and auto-generated ids must start with adv_ (so the fog's
    adversarial accounting counts them)."""
    from services.fog_service import FogService
    from core.pki import generate_keypair

    runner, fake = make_runner(
        tmp_path, rounds=1, num_devices=4, adversarial_ratio=0.5,
        max_records=40,
    )
    auto_answer(fake, {1: np.zeros(TOTAL_PARAMS, np.float32)})
    await runner.setup()
    await runner.run()

    adv_ids = [w.device_id for w in runner.workers if w.is_adversarial]
    assert adv_ids, "the ratio must yield at least one adversarial device"
    assert all(d.startswith("adv_") for d in adv_ids), (
        f"auto-generated adversarial ids must start with 'adv_', got {adv_ids}"
    )

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="fog1", sk=sk, vk=vk, peers=[], n=1, f=0,
        grpc_port=0, mqtt_enabled=False, gradient_threshold=10,
    )
    for t, p, _ in fake.published:
        if t.startswith("register/"):
            fog._on_mqtt_registration(p)

    grads = fake.published_to(f"gradients/{CLUSTER}")
    assert len(grads) == 4, "one gradient per device"
    vks = {w.device_id: w.vk for w in runner.workers}
    for payload in grads:
        g = pickle.loads(payload).gradient
        signed = gradient_signed_message(g.node_id, g.round, g.data)
        assert pki_verify(vks[g.node_id], signed, g.signature), (
            f"gradient from {g.node_id} must verify against its registered vk"
        )
        await fog.on_gradient_received(pickle.loads(payload))
    assert len(fog._pending_gradients) == 4, (
        "adversarial gradients must pass the fog PKI gate, not be dropped"
    )
    await runner.stop()


async def test_explicit_device_ids_kept_as_is(tmp_path):
    runner, fake = make_runner(
        tmp_path, device_ids=["phone0", "phone1"], rounds=1,
        adversarial_ratio=1.0,
    )
    await runner.setup()
    assert [w.device_id for w in runner.workers] == ["phone0", "phone1"], (
        "explicit device_ids must never be renamed"
    )
    await runner.stop()


def test_edge_worker_forge_signature_flag():
    from dataset.edge_worker import EdgeWorker

    rng = np.random.default_rng(0)
    data = rng.normal(size=(5, 12000)).astype(np.float32)
    labels = np.zeros(5, dtype=np.int32)

    default = EdgeWorker("adv_0", [0, 1], data, labels, is_adversarial=True)
    assert default.forge_signature is True, (
        "default: adversarial devices forge the signature (simulator behavior)"
    )

    honest = EdgeWorker("d0", [0, 1], data, labels, is_adversarial=False)
    assert honest.forge_signature is False

    data_only = EdgeWorker(
        "adv_0", [0, 1], data, labels, is_adversarial=True,
        forge_signature=False,
    )
    assert data_only.forge_signature is False, (
        "forge_signature=False must keep real signatures even when adversarial"
    )
    g = data_only.train_round(data_only.model.get_weights(), round_num=1).gradient
    assert pki_verify(
        data_only.vk, gradient_signed_message("adv_0", 1, g.data), g.signature
    ), "forge_signature=False must sign with the real sk"


# ----------------------------------------------------------------------
# Fix 5 — services path publishes raw float32 bytes, not pickled lists
# ----------------------------------------------------------------------

async def test_published_gradient_data_is_float32_bytes(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=1, num_devices=2)
    auto_answer(fake, {1: np.zeros(TOTAL_PARAMS, np.float32)})
    await runner.setup()
    await runner.run()

    grads = fake.published_to(f"gradients/{CLUSTER}")
    assert len(grads) == 2, "one gradient per device"
    for payload in grads:
        data = pickle.loads(payload).gradient.data
        assert isinstance(data, bytes), (
            f"published gradient data must be float32 bytes, got {type(data)}"
        )
        assert len(data) == TOTAL_PARAMS * 4, (
            "bytes payload must be exactly size * 4 bytes"
        )
    await runner.stop()


# ----------------------------------------------------------------------
# Fix 6 — per-cluster deterministic diversity (partition/data/key dirs)
# ----------------------------------------------------------------------

async def test_per_cluster_diversity_and_shared_init(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KEY_DIR", raising=False)

    r1, _ = make_runner(tmp_path, cluster_id="cA", key_dir=None)
    r2, _ = make_runner(tmp_path, cluster_id="cB", key_dir=None)
    await r1.setup()
    await r2.setup()

    assert np.array_equal(
        flatten_weights(r1.global_weights),
        flatten_weights(r2.global_weights),
    ), "model init must stay globally shared (MLP seed 42)"

    assert not np.array_equal(r1.workers[0].X, r2.workers[0].X), (
        "clusters must not train on identical synthetic data"
    )
    assert r1.workers[0].indices != r2.workers[0].indices, (
        "clusters must not produce identical partitions"
    )

    assert r1.key_dir != r2.key_dir, "default key_dir must be per-cluster"
    assert r1.key_dir.endswith("cA") and r2.key_dir.endswith("cB")
    await r1.stop()
    await r2.stop()


def test_key_dir_env_set_explicitly_wins(tmp_path, monkeypatch):
    shared = str(tmp_path / "shared_keys")
    monkeypatch.setenv("KEY_DIR", shared)

    runner, _ = make_runner(tmp_path, key_dir=None)
    assert runner.key_dir == shared, (
        "an explicit KEY_DIR env must be used as-is"
    )


# ----------------------------------------------------------------------
# Extra — pi_inter fail-closed, converged break, stale-round guard,
# republish cadence, EdgeWorker keypair, entrypoint re-export
# ----------------------------------------------------------------------

async def test_invalid_pi_inter_skips_update(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=2)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    d1 = np.full(TOTAL_PARAMS, 1.0, dtype=np.float32)
    d2 = np.full(TOTAL_PARAMS, 0.5, dtype=np.float32)
    delivered: set[int] = set()

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        r = pickle.loads(payload).gradient.round
        if r in delivered:
            return
        delivered.add(r)
        if r == 1:
            fake.deliver("models/global",
                         model_payload(r, d1, pi_inter=b"not-a-proof"))
        else:
            fake.deliver("models/global", model_payload(r, d2))

    fake.on_publish = on_publish
    await runner.run()

    assert runner.rounds_completed == 2
    expected = init_flat - 0.01 * d2.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "a proof that fails verification must fail closed (skip the update)"
    )


async def test_converged_flag_stops_further_rounds(tmp_path):
    runner, fake = make_runner(tmp_path, rounds=3)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    delta = np.full(TOTAL_PARAMS, 0.5, dtype=np.float32)
    auto_answer(fake, {1: delta}, converged_round=1)
    await runner.run()

    assert runner.rounds_completed == 1
    assert gradient_rounds(fake) == [1], "no training past convergence"
    expected = init_flat - 0.01 * delta.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected)


async def test_stale_round_model_does_not_satisfy_wait(tmp_path):
    """A retained models/global for an OLD round must not unlock the wait:
    it is ignored, and only the later round-matching model is applied."""
    runner, fake = make_runner(tmp_path, rounds=1, model_timeout_s=5.0)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    stale_delta = np.full(TOTAL_PARAMS, 2.0, dtype=np.float32)
    delta = np.full(TOTAL_PARAMS, 0.5, dtype=np.float32)
    delivered = {"n": 0}

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        if delivered["n"]:
            return
        delivered["n"] = 1
        fake.deliver("models/global", model_payload(99, stale_delta))
        loop = asyncio.get_running_loop()
        loop.call_later(
            0.2, fake.deliver, "models/global", model_payload(1, delta)
        )

    fake.on_publish = on_publish
    await runner.run()

    expected = init_flat - 0.01 * delta.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "only the round-matching model may be applied"
    )


async def test_early_model_for_next_round_is_buffered(tmp_path):
    """A models/global for round r+1 arriving during round r must be
    buffered (not discarded), so round r+1's wait consumes it immediately
    instead of timing out waiting for a publish that already happened."""
    runner, fake = make_runner(tmp_path, rounds=2, model_timeout_s=5.0)
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    d1 = np.full(TOTAL_PARAMS, 1.0, dtype=np.float32)
    d2 = np.full(TOTAL_PARAMS, 2.0, dtype=np.float32)
    delivered: set[int] = set()
    seen: dict[str, bool] = {}

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        r = pickle.loads(payload).gradient.round
        if r == 1 and 1 not in delivered:
            delivered.add(1)
            # round 2's model lands while round 1 is still being served
            fake.deliver("models/global", model_payload(2, d2))
            fake.deliver("models/global", model_payload(1, d1))
        elif r == 2:
            seen["buffered"] = 2 in runner._model_buffer

    fake.on_publish = on_publish
    await runner.run()

    assert seen.get("buffered"), (
        "the early round-2 model must be buffered until round 2 waits"
    )
    assert runner.rounds_completed == 2, (
        "round 2 must complete from the buffered model, not time out"
    )
    expected = init_flat - 0.01 * (d1.astype(np.float64) + d2.astype(np.float64))
    assert np.allclose(flatten_weights(runner.global_weights), expected)
    await runner.stop()


async def test_gradients_republished_while_waiting(tmp_path):
    """While waiting, the pending gradient batch is re-published so a fog
    that restarted mid-round still receives it."""
    runner, fake = make_runner(
        tmp_path, rounds=1, model_timeout_s=5.0, republish_interval_s=0.1,
    )
    await runner.setup()
    init_flat = flatten_weights(runner.global_weights)

    delta = np.full(TOTAL_PARAMS, 0.5, dtype=np.float32)
    delivered = {"n": 0}

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        n = len(fake.published_to(f"gradients/{CLUSTER}"))
        if n >= 4 and not delivered["n"]:  # after one full re-publish cycle
            delivered["n"] = 1
            fake.deliver("models/global", model_payload(1, delta))

    fake.on_publish = on_publish
    await runner.run()

    n_grads = len(fake.published_to(f"gradients/{CLUSTER}"))
    assert n_grads >= 4, (
        f"expected a re-publish cycle while waiting, saw {n_grads} gradient publishes"
    )
    expected = init_flat - 0.01 * delta.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected)


def test_edge_worker_accepts_injected_keypair(tmp_path):
    from dataset.edge_worker import EdgeWorker
    from core.pki import generate_keypair

    rng = np.random.default_rng(0)
    data = rng.normal(size=(5, 12000)).astype(np.float32)
    labels = np.zeros(5, dtype=np.int32)

    sk, vk = generate_keypair()
    worker = EdgeWorker("d0", [0, 1], data, labels, keypair=(sk, vk))
    assert (worker.sk, worker.vk) == (sk, vk)

    fresh = EdgeWorker("d0", [0, 1], data, labels)
    assert fresh.vk != vk, "no keypair given → a fresh one is generated"


def test_entrypoint_reexports_shared_key_persistence():
    from core.pki.persistence import load_or_create_keypair as shared
    from services.entrypoint import load_or_create_keypair as entry

    assert entry is shared, (
        "services.entrypoint must re-export core.pki.persistence's helper"
    )


# ----------------------------------------------------------------------
# Fix I1 — persisted progress + resume (retained models/global replay)
# ----------------------------------------------------------------------

def _progress_path(key_dir, cluster: str = CLUSTER):
    import os
    return os.path.join(str(key_dir), f"{cluster}.progress")


def _write_progress(key_dir, last_completed: int):
    import json, os
    os.makedirs(str(key_dir), exist_ok=True)
    with open(_progress_path(key_dir), "w") as fh:
        json.dump({"last_completed": last_completed}, fh)


async def test_progress_file_written_after_completed_round(tmp_path):
    """Every round the runner moves past (apply OR skip) must be recorded so
    a restarted process resumes instead of replaying the previous run's
    retained models/global payload."""
    import json

    key_dir = tmp_path / "keys"
    runner, fake = make_runner(tmp_path, key_dir=str(key_dir), rounds=1)
    await runner.setup()
    auto_answer(fake, {1: np.zeros(TOTAL_PARAMS, np.float32)})

    await runner.run()

    path = key_dir / f"{CLUSTER}.progress"
    assert path.exists(), "run() must persist a progress file in key_dir"
    data = json.loads(path.read_text())
    assert data["last_completed"] >= 1
    await runner.stop()


async def test_skipped_round_is_still_recorded_as_completed(tmp_path):
    """A round whose update is skipped (tampered delta_hash) still moves the
    loop on, so it must count as completed for resume purposes."""
    import json

    key_dir = tmp_path / "keys"
    runner, fake = make_runner(tmp_path, key_dir=str(key_dir), rounds=1)
    await runner.setup()

    delivered = {"n": 0}

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        if delivered["n"]:
            return
        delivered["n"] = 1
        raw = np.ones(TOTAL_PARAMS, dtype=np.float32).tobytes()
        fake.deliver("models/global", pickle.dumps({
            "schema": 1, "round": 1,
            "delta_w_inter": raw,
            "delta_hash": b"\x00" * 32,  # tampered -> skipped
            "learning_rate": 0.01, "pi_inter": None,
            "n_active_clusters": 1, "converged": False,
        }))

    fake.on_publish = on_publish
    await runner.run()

    data = json.loads((key_dir / f"{CLUSTER}.progress").read_text())
    assert data["last_completed"] >= 1, (
        "a skipped round still completes the loop iteration"
    )
    await runner.stop()


async def test_runner_resumes_at_next_round(tmp_path):
    """A second runner with the same key_dir starts at last_completed + 1
    and only trains the remaining rounds."""
    key_dir = tmp_path / "keys"

    r1, fake1 = make_runner(tmp_path, key_dir=str(key_dir), rounds=1)
    await r1.setup()
    auto_answer(fake1, {1: np.zeros(TOTAL_PARAMS, np.float32)})
    await r1.run()
    await r1.stop()

    r2, fake2 = make_runner(tmp_path, key_dir=str(key_dir), rounds=3)
    auto_answer(fake2, {
        1: np.full(TOTAL_PARAMS, 1.0, np.float32),
        2: np.full(TOTAL_PARAMS, 0.5, np.float32),
        3: np.full(TOTAL_PARAMS, 0.25, np.float32),
    })
    await r2.setup()

    assert r2.start_round == 2, (
        f"expected resume at round 2, got start_round={r2.start_round}"
    )

    await r2.run()
    assert gradient_rounds(fake2) == [2, 3], (
        f"a resumed run must not retrain round 1; got {gradient_rounds(fake2)}"
    )
    assert r2.rounds_completed == 3
    await r2.stop()


async def test_finished_run_restarts_from_round_one(tmp_path, caplog):
    """Progress >= rounds means the previous run finished: the next run must
    start over at round 1 (a fresh experiment), not sit idle."""
    import logging

    _write_progress(tmp_path / "keys", 99)

    runner, fake = make_runner(
        tmp_path, key_dir=str(tmp_path / "keys"), rounds=2,
    )
    with caplog.at_level(logging.INFO):
        await runner.setup()

    assert runner.start_round == 1, (
        "a finished previous run must restart at round 1"
    )
    assert any("restart" in r.getMessage().lower() for r in caplog.records), (
        "the restart must be logged"
    )

    auto_answer(fake, {
        1: np.zeros(TOTAL_PARAMS, np.float32),
        2: np.zeros(TOTAL_PARAMS, np.float32),
    })
    await runner.run()
    assert gradient_rounds(fake) == [1, 2]
    await runner.stop()


async def test_retained_stale_round_does_not_satisfy_resumed_wait(tmp_path):
    """The retained models/global of a previous run (round < start_round)
    delivered at subscribe time must neither satisfy the resumed round's
    wait nor be applied as its update."""
    key_dir = tmp_path / "keys"
    _write_progress(key_dir, 2)

    runner, fake = make_runner(
        tmp_path, key_dir=str(key_dir), rounds=3, model_timeout_s=5.0,
    )
    await runner.setup()
    assert runner.start_round == 3

    # retained message from the previous run lands at subscribe time
    # (current_round == 0, so the buffer accepts it)
    stale = np.full(TOTAL_PARAMS, 9.0, np.float32)
    fake.deliver("models/global", model_payload(1, stale))
    assert 1 in runner._model_buffer, (
        "the retained stale payload is buffered while current_round == 0"
    )

    init_flat = flatten_weights(runner.global_weights)
    delta = np.full(TOTAL_PARAMS, 0.5, np.float32)
    seen: dict[str, bool] = {}

    def on_publish(topic, payload):
        if not topic.startswith("gradients/"):
            return
        if "checked" in seen:
            return
        seen["checked"] = True
        seen["pruned"] = 1 not in runner._model_buffer
        fake.deliver("models/global", model_payload(3, delta))

    fake.on_publish = on_publish
    await runner.run()

    assert seen.get("pruned"), (
        "rounds below the current round must be pruned at round start"
    )
    assert 1 not in runner._model_buffer
    assert runner.rounds_completed == 3
    expected = init_flat - 0.01 * delta.astype(np.float64)
    assert np.allclose(flatten_weights(runner.global_weights), expected), (
        "only the round-3 model may be applied, never the stale retained one"
    )
    await runner.stop()
