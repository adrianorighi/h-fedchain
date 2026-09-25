"""Task 11 — edge_client: cluster topic, registration, persistent keys, bytes payload.

No broker anywhere: payload building / key loading / arg parsing are pure
functions, and ``main()`` takes an injected mqtt factory (FakeMqttClient).
SNARK stays off in these tests (py_ecc pairing may hang in this environment).
"""

import asyncio
import os
import pickle

import numpy as np

from core.pki import generate_keypair, gradient_signed_message, verify as pki_verify
from hfc_types.messages import GradientWithProof
import services.edge_client as ec

DEVICE = "d0"
CLUSTER = "c7"
ROUND = 3
INPUT_DIM = 4


def flat_grad(size: int = 10) -> np.ndarray:
    return np.random.randn(size)


class FakeMqttClient:
    """In-memory MqttClient stand-in recording (topic, payload, retain)."""

    def __init__(self, deliver: bool = True):
        self.published: list[tuple[str, bytes, bool]] = []
        self.deliver = deliver
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def publish(self, topic, payload, retain=False) -> bool:
        self.published.append((topic, payload, retain))
        return self.deliver


def run_main(argv, fake: FakeMqttClient):
    asyncio.run(ec.main(argv, mqtt_factory=lambda cid, broker, port: fake))


def main_argv(tmp_path, cluster=CLUSTER, device=DEVICE, key_name="d0.key"):
    return [
        "--device-id", device,
        "--cluster-id", cluster,
        "--round", "1",
        "--input-dim", str(INPUT_DIM),
        "--key-file", str(tmp_path / key_name),
    ]


# ------------------------------------------------------------------
# build_payloads
# ------------------------------------------------------------------


def test_registration_topic_and_payload_contract():
    sk, vk = generate_keypair()
    reg_topic, reg_payload, _, _ = ec.build_payloads(
        DEVICE, CLUSTER, ROUND, flat_grad(), sk, vk
    )
    assert reg_topic == f"register/{CLUSTER}/{DEVICE}"
    reg = pickle.loads(reg_payload)
    assert set(reg.keys()) == {"node_id", "vk"}
    assert reg["node_id"] == DEVICE
    assert reg["vk"] == vk


def test_gradient_topic_uses_cluster_not_default():
    sk, vk = generate_keypair()
    _, _, grad_topic, _ = ec.build_payloads(
        DEVICE, CLUSTER, ROUND, flat_grad(), sk, vk
    )
    assert grad_topic == f"gradients/{CLUSTER}"
    assert grad_topic != "gradients/default"


def test_gradient_payload_data_is_float32_bytes():
    flat = flat_grad(7)
    sk, vk = generate_keypair()
    _, _, _, grad_payload = ec.build_payloads(
        DEVICE, CLUSTER, ROUND, flat, sk, vk
    )
    gwp = pickle.loads(grad_payload)
    assert isinstance(gwp, GradientWithProof)
    assert isinstance(gwp.gradient.data, bytes)
    assert len(gwp.gradient.data) == flat.size * 4
    np.testing.assert_array_equal(
        np.frombuffer(gwp.gradient.data, dtype=np.float32),
        flat.astype(np.float32),
    )
    assert gwp.gradient.node_id == DEVICE
    assert gwp.gradient.round == ROUND


def test_signature_verifies_against_registration_vk():
    flat = flat_grad(12)
    sk, vk = generate_keypair()
    _, reg_payload, _, grad_payload = ec.build_payloads(
        DEVICE, CLUSTER, ROUND, flat, sk, vk
    )
    reg_vk = pickle.loads(reg_payload)["vk"]
    g = pickle.loads(grad_payload).gradient
    # bytes form (as published)
    assert pki_verify(
        reg_vk, gradient_signed_message(g.node_id, g.round, g.data), g.signature
    )
    # cross-form: the same signature must also verify over the list form,
    # since gradient_signed_message canonicalizes to float32 either way
    assert pki_verify(
        reg_vk,
        gradient_signed_message(g.node_id, g.round, flat.tolist()),
        g.signature,
    )


def test_default_key_path_is_per_cluster():
    assert ec.default_key_path(CLUSTER, DEVICE) == os.path.join(
        os.path.expanduser("~/.hfc/keys/edge"), CLUSTER, f"{DEVICE}.key"
    )


# ------------------------------------------------------------------
# key persistence
# ------------------------------------------------------------------


def test_load_keys_persists_and_is_stable(tmp_path):
    key = str(tmp_path / "sub" / "d0.key")
    _, vk1 = ec.load_keys(key, CLUSTER, DEVICE)
    _, vk2 = ec.load_keys(key, CLUSTER, DEVICE)
    assert vk1 == vk2
    assert os.path.exists(key)
    _, vk_other = ec.load_keys(str(tmp_path / "other.key"), CLUSTER, DEVICE)
    assert vk_other != vk1


def test_main_same_key_file_same_vk_different_file_differs(tmp_path):
    fake1 = FakeMqttClient()
    run_main(main_argv(tmp_path), fake1)
    fake2 = FakeMqttClient()
    run_main(main_argv(tmp_path), fake2)
    fake3 = FakeMqttClient()
    run_main(main_argv(tmp_path, key_name="other.key"), fake3)

    vk1 = pickle.loads(fake1.published[0][1])["vk"]
    vk2 = pickle.loads(fake2.published[0][1])["vk"]
    vk3 = pickle.loads(fake3.published[0][1])["vk"]
    assert vk1 == vk2
    assert vk3 != vk1
    # and the registered vk is the one persisted in the key file
    _, key_vk = ec.load_keys(str(tmp_path / "d0.key"), CLUSTER, DEVICE)
    assert vk1 == key_vk


# ------------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------------


def test_cluster_id_defaults_from_env(monkeypatch):
    argv = ["--device-id", DEVICE, "--round", "1"]
    monkeypatch.setenv("CLUSTER_ID", "c9")
    assert ec.parse_args(argv).cluster_id == "c9"
    monkeypatch.delenv("CLUSTER_ID")
    assert ec.parse_args(argv).cluster_id == "default"


def test_main_publishes_registration_before_gradient(tmp_path, capsys):
    fake = FakeMqttClient()
    run_main(main_argv(tmp_path), fake)

    assert fake.started and fake.stopped
    assert len(fake.published) == 2
    (reg_topic, reg_payload, reg_retain), (grad_topic, grad_payload, grad_retain) = (
        fake.published
    )

    assert reg_topic == f"register/{CLUSTER}/{DEVICE}"
    assert reg_retain is True
    reg = pickle.loads(reg_payload)
    assert set(reg.keys()) == {"node_id", "vk"}
    _, key_vk = ec.load_keys(str(tmp_path / "d0.key"), CLUSTER, DEVICE)
    assert reg["vk"] == key_vk

    assert grad_topic == f"gradients/{CLUSTER}"
    assert grad_retain is False
    gwp = pickle.loads(grad_payload)
    assert isinstance(gwp.gradient.data, bytes)

    out = capsys.readouterr().out
    assert reg_topic in out
    assert grad_topic in out


def test_main_warns_when_registration_not_delivered(tmp_path, capsys):
    fake = FakeMqttClient(deliver=False)
    run_main(main_argv(tmp_path), fake)
    out = capsys.readouterr().out
    assert f"register/{CLUSTER}/{DEVICE}" in out
    assert "not delivered" in out


def test_edge_client_module_imports():
    import services.edge_client as ec_mod
    assert ec_mod is not None
    assert hasattr(ec_mod, "main")
    assert callable(ec_mod.main)
