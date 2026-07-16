import pytest
from core.hotstuff.quorum import QuorumCertifier
from core.pki.ed25519 import generate_keypair, sign
from hfc_types.block import Block, QuorumCertificate
from hfc_types.messages import MessageType


def test_qc_with_real_signatures():
    qc = QuorumCertifier()
    vk_map = {}
    votes = []
    block_hash = b"test_block_hash_32_bytes!!"
    for i in range(4):
        sk, vk = generate_keypair()
        vk_map[f"node{i}"] = vk
        msg = str(1).encode() + block_hash + MessageType.PREPARE.name.encode()
        sig = sign(sk, msg)
        votes.append((f"node{i}", sig))
    qc_obj = qc.collect(1, block_hash, MessageType.PREPARE, votes, 3, vk_map)
    assert qc_obj.round == 1
    assert len(qc_obj.signatures) == 3


def test_qc_rejects_invalid_signature():
    qc = QuorumCertifier()
    vk_map = {}
    block_hash = b"test_block_hash_32_bytes!!"
    sk, vk = generate_keypair()
    vk_map["node0"] = vk
    with pytest.raises(ValueError, match="Invalid signature"):
        qc.collect(1, block_hash, MessageType.PREPARE, [("node0", b"fake_sig")], 1, vk_map)


def test_qc_rejects_insufficient_quorum():
    qc = QuorumCertifier()
    with pytest.raises(ValueError, match="Insufficient signatures"):
        qc.collect(1, b"hash", MessageType.PREPARE, [], 3, None)


@pytest.mark.asyncio
async def test_fog_node_generates_real_votes():
    from simulator.fog_node import FogNode
    from simulator.network import EmulatedNetwork

    sk0, vk0 = generate_keypair()
    sk1, vk1 = generate_keypair()
    sk2, vk2 = generate_keypair()

    net = EmulatedNetwork(latency_ms=0)
    net.add_node("n0")
    net.add_node("n1")
    net.add_node("n2")

    node = FogNode("n0", sk0, vk0, ["n1", "n2"], 3, 1, net)
    node._set_vk("n1", vk1)
    node._set_vk("n2", vk2)
    node._set_peer_sk("n1", sk1)
    node._set_peer_sk("n2", sk2)

    block = Block(round=1, gradient_hash=b"gh", qc_commit=None, stark_proof=None,
                  accepted_devices=[], rejected_devices=[], timestamp=0.0, prev_hash=b"\x00" * 32)

    result = await node.run_consensus(1, True, block)
    assert result is not None
    assert result.block.qc_commit is not None
    assert len(result.block.qc_commit.signatures) >= 2
