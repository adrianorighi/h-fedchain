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
    for nid in ["n0", "n1", "n2"]:
        net.add_node(nid)

    nodes = [
        FogNode("n0", sk0, vk0, ["n1", "n2"], 3, 1, net),
        FogNode("n1", sk1, vk1, ["n0", "n2"], 3, 1, net),
        FogNode("n2", sk2, vk2, ["n0", "n1"], 3, 1, net),
    ]
    for node in nodes:
        node._set_vk("n0", vk0)
        node._set_vk("n1", vk1)
        node._set_vk("n2", vk2)

    block = Block(round=1, gradient_hash=b"gh", qc_commit=None, stark_proof=None,
                  accepted_devices=[], rejected_devices=[], timestamp=0.0, prev_hash=b"\x00" * 32)

    round_num = 1
    for node in nodes:
        await node.hotstuff.start_round(round_num, node.node_id == "n0")

    proposal = await nodes[0].hotstuff.propose(block)
    assert proposal is not None

    vk_map = {n.node_id: n.vk for n in nodes}
    prepare_votes = []
    for node in nodes:
        vote = await node.hotstuff.on_prepare(block)
        if vote:
            prepare_votes.append((vote.node_id, vote.signature))

    qc_prepare = await nodes[0].hotstuff.collect_votes(
        round_num, block.hash, "prepare", prepare_votes, vk_map,
    )
    assert qc_prepare is not None
    assert len(qc_prepare.signatures) >= 2

    pre_commit_votes = []
    for node in nodes:
        vote = await node.hotstuff.on_pre_commit(qc_prepare)
        if vote:
            pre_commit_votes.append((vote.node_id, vote.signature))

    qc_pre_commit = await nodes[0].hotstuff.collect_votes(
        round_num, block.hash, "pre_commit", pre_commit_votes, vk_map,
    )
    assert qc_pre_commit is not None

    commit_votes = []
    for node in nodes:
        vote = await node.hotstuff.on_commit(qc_pre_commit)
        if vote:
            commit_votes.append((vote.node_id, vote.signature))

    qc_commit = await nodes[0].hotstuff.collect_votes(
        round_num, block.hash, "commit", commit_votes, vk_map,
    )
    assert qc_commit is not None
    assert len(qc_commit.signatures) >= 2

    entry = await nodes[0].finalize_commit(qc_commit, block)
    assert entry is not None
    assert entry.block.qc_commit is not None
