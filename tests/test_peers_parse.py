import asyncio


def test_parse_peers_host_port():
    from services.entrypoint import parse_peers
    peers, addrs = parse_peers("fog2:50051,fog3:50051", default_port=50051)
    assert peers == ["fog2", "fog3"]
    assert addrs == {"fog2": "fog2:50051", "fog3": "fog3:50051"}


def test_parse_peers_id_only_and_at_syntax():
    from services.entrypoint import parse_peers
    peers, addrs = parse_peers("fog2,fog3@10.0.0.9:60051", default_port=50051)
    assert peers == ["fog2", "fog3"]
    assert addrs["fog2"] == "fog2:50051"
    assert addrs["fog3"] == "10.0.0.9:60051"


def test_parse_peers_empty():
    from services.entrypoint import parse_peers
    assert parse_peers("", default_port=50051) == ([], {})


# ----------------------------------------------------------------------
# Persistent fog node keys (Round-2 Fix 2)
# ----------------------------------------------------------------------

def test_load_or_create_keypair_creates_then_reloads(tmp_path):
    import os

    from services.entrypoint import load_or_create_keypair

    key_path = str(tmp_path / "keys" / "fog1.key")
    assert not os.path.exists(key_path)

    sk1, vk1 = load_or_create_keypair(key_path)
    assert os.path.exists(key_path)
    assert len(sk1) == 32 and len(vk1) == 32
    assert (os.stat(key_path).st_mode & 0o777) == 0o600

    sk2, vk2 = load_or_create_keypair(key_path)
    assert sk2 == sk1, "second boot must reuse the persisted secret key"
    assert vk2 == vk1, "second boot must reuse the persisted public key"


# ----------------------------------------------------------------------
# FogService configuration plumbing
# ----------------------------------------------------------------------

def _make_fog(**kwargs):
    from core.pki import generate_keypair
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    defaults = dict(node_id="fog1", sk=sk, vk=vk, peers=[], n=3, f=1, grpc_port=0)
    defaults.update(kwargs)
    fog = FogService(**defaults)
    fog._mqtt_enabled = False
    return fog


def _make_gradient(node_id: str, round_num: int, sk: bytes):
    from core.pki import gradient_signed_message, sign as pki_sign
    from hfc_types.messages import Gradient, GradientWithProof

    data = [float(i) for i in range(10)]
    gradient = Gradient(
        node_id=node_id, round=round_num,
        data=data,
        signature=pki_sign(sk, gradient_signed_message(node_id, round_num, data)),
    )
    return GradientWithProof(gradient=gradient, snark_proof=None)


def test_fog_config_defaults():
    fog = _make_fog()
    assert fog.cluster_id == "default"
    assert fog.gradient_threshold == 2  # n - f
    assert fog.cloud_address == "cloud:50052"
    assert fog.peer_addrs == {}
    assert fog.vote_timeout_s == 5.0


def test_fog_config_overrides():
    fog = _make_fog(
        cluster_id="c1",
        gradient_threshold=20,
        cloud_address="cloud-alt:50052",
        peer_addrs={"fog2": "10.0.0.2:60051"},
        vote_timeout_s=1.5,
    )
    assert fog.cluster_id == "c1"
    assert fog.gradient_threshold == 20
    assert fog.cloud_address == "cloud-alt:50052"
    assert fog.peer_addrs == {"fog2": "10.0.0.2:60051"}
    assert fog.vote_timeout_s == 1.5


async def test_gradient_threshold_overrides_n_minus_f():
    """One gradient triggers (and passes the gate) when threshold=1, even if n-f=2."""
    fog = _make_fog(n=3, f=1, gradient_threshold=1)
    fog._running = True

    async def _noop_leader(*args, **kwargs):
        pass

    fog._run_leader_consensus = _noop_leader

    fog.set_vk("d0", fog.engine.vk)
    await fog.on_gradient_received(_make_gradient("d0", 1, fog.engine.sk))
    assert fog._last_aggregate is not None, (
        "round did not run with gradient_threshold=1 (n-f=2)"
    )


async def test_start_subscribes_cluster_topic():
    fog = _make_fog(cluster_id="c7")
    subscribed: list[str] = []

    class FakeMqtt:
        async def start(self):
            pass

        def subscribe(self, topic, callback):
            subscribed.append(topic)

        async def stop(self):
            pass

    async def _noop_serve():
        pass

    fog.mqtt = FakeMqtt()
    fog._mqtt_enabled = True
    fog._serve_grpc = _noop_serve
    await fog.start()
    assert subscribed == ["gradients/c7", "register/c7/#"]
    await fog.stop()


async def test_connect_peers_uses_peer_addrs():
    fog = _make_fog(
        peers=["fog2", "fog3"],
        grpc_port=50051,
        peer_addrs={"fog2": "10.0.0.2:60051"},
    )
    recorded: list[tuple[str, str]] = []

    async def _fake_connect(peer_id, addr, req_gen):
        recorded.append((peer_id, addr))

    fog._connect_one_peer = _fake_connect
    await fog._connect_peers()
    await asyncio.sleep(0)  # let create_task() run the fake connector
    assert ("fog2", "10.0.0.2:60051") in recorded
    assert ("fog3", "fog3:50051") in recorded


async def test_send_to_cloud_uses_cloud_address(monkeypatch):
    import grpc.aio
    from hfc_types.block import Block

    fog = _make_fog(cloud_address="cloud-alt:70052")
    captured: dict[str, str] = {}

    class FakeChannel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def unary_unary(self, *args, **kwargs):
            async def _call(request, **kw):
                return None

            return _call

    def fake_insecure_channel(addr):
        captured["addr"] = addr
        return FakeChannel()

    monkeypatch.setattr(grpc.aio, "insecure_channel", fake_insecure_channel)

    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None, stark_proof=None,
        accepted_devices=[], rejected_devices=[],
        timestamp=0.0, prev_hash=b"\x00" * 32,
    )
    await fog._send_to_cloud(block, b"delta-bytes", 4)
    assert captured["addr"] == "cloud-alt:70052"


async def test_send_to_cloud_has_deadline(monkeypatch):
    """A hung cloud RPC must not hold the _round_in_flight guard forever."""
    import grpc.aio

    from hfc_types.block import Block

    fog = _make_fog()
    rpc_kwargs: dict = {}

    class FakeChannel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def unary_unary(self, *args, **kwargs):
            async def _call(request, **kw):
                rpc_kwargs.update(kw)
                return None

            return _call

    monkeypatch.setattr(
        grpc.aio, "insecure_channel", lambda addr: FakeChannel()
    )

    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None, stark_proof=None,
        accepted_devices=[], rejected_devices=[],
        timestamp=0.0, prev_hash=b"\x00" * 32,
    )
    await fog._send_to_cloud(block, b"delta-bytes", 4)

    assert rpc_kwargs.get("timeout") == fog.vote_timeout_s + 5


async def test_send_to_cloud_swallows_deadline_exceeded(monkeypatch):
    """Deadline exceeded must degrade to the printed failure, not a raise."""
    import grpc.aio

    from hfc_types.block import Block

    fog = _make_fog()

    class FakeChannel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def unary_unary(self, *args, **kwargs):
            async def _call(request, **kw):
                raise asyncio.TimeoutError()

            return _call

    monkeypatch.setattr(
        grpc.aio, "insecure_channel", lambda addr: FakeChannel()
    )

    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None, stark_proof=None,
        accepted_devices=[], rejected_devices=[],
        timestamp=0.0, prev_hash=b"\x00" * 32,
    )
    await fog._send_to_cloud(block, b"delta-bytes", 4)  # must not raise


async def test_collect_quorum_votes_uses_vote_timeout():
    import time

    fog = _make_fog(vote_timeout_s=0.2)
    t0 = time.monotonic()
    result = await fog._collect_quorum_votes(1, b"hash", "prepare", 5)
    elapsed = time.monotonic() - t0
    assert result is None
    assert elapsed < 2.0, f"vote collection took {elapsed:.1f}s, expected ~0.2s"


async def test_peer_retry_loop_stops_when_not_running():
    """Retry loop must bail out immediately once _running is False."""
    import time

    from core.pki import generate_keypair
    from services.fog_service import FogService

    sk, vk = generate_keypair()
    fog = FogService(
        node_id="f1", sk=sk, vk=vk, peers=["f2"], n=3, f=1,
        grpc_port=0, peer_addrs={"f2": "127.0.0.1:1"},
    )
    fog._mqtt_enabled = False
    fog._running = False

    t0 = time.monotonic()
    await fog._connect_one_peer("f2", "127.0.0.1:1", None)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"retry loop took {elapsed:.2f}s with _running=False"


# ----------------------------------------------------------------------
# Entrypoint membership consistency (Task 6 Fix I1)
# ----------------------------------------------------------------------

def _env_fog(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("SERVICE", "fog")
    monkeypatch.setenv("NODE_ID", "fog1")
    monkeypatch.setenv("N", "3")
    monkeypatch.setenv("F", "1")
    monkeypatch.setenv("PEERS", "")
    monkeypatch.setenv("NODE_KEY_FILE", str(tmp_path / "fog1.key"))
    monkeypatch.setenv("GRPC_PORT", "0")
    monkeypatch.setenv("CLOUD_ADDRESS", "127.0.0.1:1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_entrypoint_membership_derives_n_from_peers(monkeypatch, tmp_path, capsys):
    from services.entrypoint import build_fog_service

    _env_fog(monkeypatch, tmp_path)  # N=3 but PEERS unset -> single node
    fog = build_fog_service()

    assert fog.n == 1, "empty PEERS means a single-node cluster (len+1)"
    assert fog.f == 0, "f must stay below the effective membership n"
    out = capsys.readouterr().out
    assert "N=3" in out and "overridden" in out, (
        f"expected an N-override warning, got:\n{out}"
    )


def test_entrypoint_membership_with_peers_matches_env(monkeypatch, tmp_path, capsys):
    from services.entrypoint import build_fog_service

    _env_fog(monkeypatch, tmp_path, PEERS="fog2:50051,fog3:50051")
    fog = build_fog_service()

    assert fog.n == 3, "N must match len(peers)+1 when they agree"
    assert fog.f == 1
    assert "overridden" not in capsys.readouterr().out, (
        "no warning when env N matches actual peer membership"
    )


async def test_entrypoint_single_node_round_commits(monkeypatch, tmp_path):
    from core.pki import generate_keypair, gradient_signed_message
    from core.pki import sign as pki_sign
    from hfc_types.messages import Gradient, GradientWithProof
    from services.entrypoint import build_fog_service

    _env_fog(monkeypatch, tmp_path)  # N=3, PEERS unset -> must still commit
    fog = build_fog_service()
    fog._mqtt_enabled = False
    fog._running = True

    async def _cloud_noop(block, delta_bytes, n_devices):
        pass

    fog._send_to_cloud = _cloud_noop

    sk, _ = generate_keypair()
    data = [1.0, 2.0, 3.0]
    sig = pki_sign(sk, gradient_signed_message("d0", 1, data))
    fog._pending_gradients = [GradientWithProof(gradient=Gradient(
        node_id="d0", round=1, data=data, signature=sig,
    ))]

    await fog._run_round()

    assert fog.ledger.get_height() == 1, (
        "a default-config (N=3, no PEERS) single-node round must commit"
    )
    qc = fog.ledger._entries[-1].block.qc_commit
    quorum = fog.quorum_certifier.quorum_size(fog.n)
    assert qc.is_valid(quorum, {fog.node_id: fog.engine.vk})


def test_entrypoint_fog_reads_mqtt_enabled_env(monkeypatch, tmp_path):
    """Fog parity: build_fog_service must honour MQTT_ENABLED like the cloud
    branch does (MQTT_ENABLED=0 disables MQTT on the fog too)."""
    from services.entrypoint import build_fog_service

    _env_fog(monkeypatch, tmp_path, MQTT_ENABLED="0")
    fog = build_fog_service()

    assert fog._mqtt_enabled is False, (
        "MQTT_ENABLED=0 must reach FogService as mqtt_enabled=False"
    )


def test_entrypoint_cloud_treats_falsey_mqtt_enabled_values_as_disabled(
    monkeypatch, capsys,
):
    """MQTT_ENABLED accepts 0/false/no (case-insensitive) as disabled."""
    from services.entrypoint import build_cloud_service

    for value in ("0", "false", "False", "no", "NO", "  0 "):
        monkeypatch.setenv("MQTT_ENABLED", value)
        cloud = build_cloud_service()
        assert cloud._mqtt_enabled is False, (
            f"MQTT_ENABLED={value!r} must disable MQTT"
        )

    for value in ("1", "true", "yes", ""):
        monkeypatch.setenv("MQTT_ENABLED", value)
        cloud = build_cloud_service()
        assert cloud._mqtt_enabled is True, (
            f"MQTT_ENABLED={value!r} must keep MQTT enabled"
        )


def test_entrypoint_build_cloud_service_env(monkeypatch, tmp_path, capsys):
    """Cloud branch reads N_EXPECTED_CLUSTERS/LEARNING_RATE/MQTT_*/WORM_DB_PATH."""
    from services.entrypoint import build_cloud_service

    monkeypatch.setenv("GRPC_PORT", "50059")
    monkeypatch.setenv("MQTT_BROKER", "127.0.0.1")
    monkeypatch.setenv("MQTT_PORT", "1883")
    monkeypatch.setenv("N_EXPECTED_CLUSTERS", "3")
    monkeypatch.setenv("LEARNING_RATE", "0.05")
    monkeypatch.setenv("WORM_DB_PATH", str(tmp_path / "worm.db"))
    monkeypatch.setenv("MQTT_ENABLED", "0")

    cloud = build_cloud_service()

    assert cloud.n_expected_clusters == 3
    assert cloud.learning_rate == 0.05
    assert cloud.grpc_port == 50059
    assert cloud._mqtt_enabled is False
    assert cloud.mqtt.broker_host == "127.0.0.1"
    assert cloud.mqtt.broker_port == 1883
    assert cloud.worm._db_path == str(tmp_path / "worm.db")
    out = capsys.readouterr().out
    assert "expected_clusters=3" in out, (
        f"expected effective config print, got:\n{out}"
    )


def test_entrypoint_cloud_service_defaults(monkeypatch, capsys):
    """Without env overrides the cloud falls back to documented defaults."""
    from services.entrypoint import build_cloud_service

    for key in ("N_EXPECTED_CLUSTERS", "LEARNING_RATE", "MQTT_ENABLED",
                "WORM_DB_PATH", "GRPC_PORT", "MQTT_BROKER", "MQTT_PORT"):
        monkeypatch.delenv(key, raising=False)

    cloud = build_cloud_service()

    assert cloud.n_expected_clusters == 1
    assert cloud.learning_rate == 0.01
    assert cloud._mqtt_enabled is True
    assert cloud.worm._db_path is None
    assert cloud.grpc_port == 50052
