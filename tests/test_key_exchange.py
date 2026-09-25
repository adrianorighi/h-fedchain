import asyncio
import pickle
import pytest
from core.pki import generate_keypair, gradient_signed_message, sign as pki_sign
from hfc_types.messages import Gradient, GradientWithProof
from proto import hfedchain_pb2
from services.fog_service import FogService


def _mkfog(peers=("fog2",)):
    sk, vk = generate_keypair()
    fog = FogService(node_id="fog1", sk=sk, vk=vk, peers=list(peers), n=3, f=1, grpc_port=0)
    fog._mqtt_enabled = False
    return fog


def test_key_exchange_message_updates_vk_map_and_replies():
    fog = _mkfog()
    fog._peer_queues["fog2"] = asyncio.Queue()
    peer_vk = b"\x01" * 32
    msg = hfedchain_pb2.ConsensusMessage(
        key_exchange=hfedchain_pb2.KeyExchange(node_id="fog2", vk=peer_vk))

    async def go():
        await fog._on_consensus_message(msg, "irrelevant")
    asyncio.run(go())
    assert fog._vk_map["fog2"] == peer_vk
    assert not fog._peer_queues["fog2"].empty()   # echoed our own KX back


def test_key_exchange_not_resent_twice():
    fog = _mkfog()
    fog._peer_queues["fog2"] = asyncio.Queue()
    msg = hfedchain_pb2.ConsensusMessage(
        key_exchange=hfedchain_pb2.KeyExchange(
            node_id="fog2", vk=b"\x02" * 32))

    async def go():
        await fog._on_consensus_message(msg, "p")
        while not fog._peer_queues["fog2"].empty():
            fog._peer_queues["fog2"].get_nowait()
        await fog._on_consensus_message(msg, "p")
    asyncio.run(go())
    assert fog._peer_queues["fog2"].empty()       # no second echo


def test_registration_handler_stores_device_vk():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog._on_mqtt_registration(pickle.dumps({"node_id": "d0", "vk": vk}))
    assert fog._vk_map["d0"] == vk


def test_registration_handler_ignores_garbage():
    fog = _mkfog()
    fog._on_mqtt_registration(b"not a pickle")
    assert fog._vk_map == {}


@pytest.mark.asyncio
async def test_gradient_bad_signature_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)                      # vk known
    g = GradientWithProof(gradient=Gradient(
        node_id="d0", round=1, data=[0.0], signature=b"bad"))
    await fog.on_gradient_received(g)
    assert fog._pending_gradients == []


@pytest.mark.asyncio
async def test_gradient_unknown_device_rejected():
    fog = _mkfog()
    g = GradientWithProof(gradient=Gradient(
        node_id="stranger", round=1, data=[0.0], signature=b"x"))
    await fog.on_gradient_received(g)
    assert fog._pending_gradients == []


@pytest.mark.asyncio
async def test_gradient_good_signature_accepted():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    g = GradientWithProof(gradient=Gradient(
        node_id="d0", round=1, data=[0.0],
        signature=pki_sign(sk, gradient_signed_message("d0", 1, [0.0]))))
    await fog.on_gradient_received(g)
    assert len(fog._pending_gradients) == 1
    await fog.stop()


@pytest.mark.asyncio
async def test_gradient_wrong_round_signature_rejected():
    fog = _mkfog()
    sk, vk = generate_keypair()
    fog.set_vk("d0", vk)
    g = GradientWithProof(gradient=Gradient(
        node_id="d0", round=2, data=[0.0],
        signature=pki_sign(sk, gradient_signed_message("d0", 1, [0.0]))))  # signed for round 1
    await fog.on_gradient_received(g)
    assert fog._pending_gradients == []
