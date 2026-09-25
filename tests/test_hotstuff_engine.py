import pytest
from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.pki import generate_keypair, sign
from hfc_types.block import Block
from hfc_types.messages import MessageType
from core.hotstuff.messages import PrepareProposal


@pytest.fixture
def engine():
    sk, vk = generate_keypair()
    peers = ["n0", "n1", "n2", "n3", "n4"]
    return HotStuffEngine(
        node_id="n0",
        sk=sk,
        vk=vk,
        peers=peers,
        n=5,
        f=1,
    )


@pytest.mark.asyncio
async def test_start_round_sets_state(engine):
    await engine.start_round(round_num=1, is_leader=False)
    assert engine.round == 1
    assert engine.state == "READY_TO_VOTE"


@pytest.mark.asyncio
async def test_start_round_leader_state(engine):
    await engine.start_round(round_num=1, is_leader=True)
    assert engine.state == "READY_TO_PROPOSE"


@pytest.mark.asyncio
async def test_prepare_vote_emitted(engine):
    await engine.start_round(round_num=1, is_leader=False)
    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    vote = await engine.on_prepare(block)
    assert vote is not None
    assert vote.phase == "prepare"
    assert vote.block_hash == block.hash
    assert len(vote.signature) > 0


@pytest.mark.asyncio
async def test_double_vote_rejected_for_conflicting_block(engine):
    await engine.start_round(round_num=1, is_leader=False)
    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    other = Block(
        round=1, gradient_hash=b"gh2", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=200.0, prev_hash=b"\x00" * 32,
    )
    vote1 = await engine.on_prepare(block)
    vote2 = await engine.on_prepare(block)
    assert vote1 is not None
    assert vote2 is not None, "same (round, hash) must be re-sendable"
    assert await engine.on_prepare(other) is None


@pytest.mark.asyncio
async def test_leader_propose_creates_proposal(engine):
    await engine.start_round(round_num=1, is_leader=True)
    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    proposal = await engine.propose(block)
    assert proposal is not None
    assert proposal.block.round == 1
    assert proposal.leader_id == "n0"


@pytest.mark.asyncio
async def test_follower_cannot_propose(engine):
    await engine.start_round(round_num=1, is_leader=False)
    block = Block(
        round=1, gradient_hash=b"gh", qc_commit=None,
        stark_proof=None, accepted_devices=[], rejected_devices=[],
        timestamp=100.0, prev_hash=b"\x00" * 32,
    )
    proposal = await engine.propose(block)
    assert proposal is None


@pytest.mark.asyncio
async def test_on_pre_commit_emits_vote(engine):
    await engine.start_round(round_num=1, is_leader=False)
    qc = QuorumCertifier().collect(
        round=1, block_hash=b"bh", msg_type=MessageType.PREPARE,
        signatures=[("n0", b"sig")] * 4, quorum_size=4,
    )
    vote = await engine.on_pre_commit(qc)
    assert vote is not None
    assert vote.phase == "pre_commit"


@pytest.mark.asyncio
async def test_on_commit_emits_vote(engine):
    await engine.start_round(round_num=1, is_leader=False)
    qc = QuorumCertifier().collect(
        round=1, block_hash=b"bh", msg_type=MessageType.PRE_COMMIT,
        signatures=[("n0", b"sig")] * 4, quorum_size=4,
    )
    vote = await engine.on_commit(qc)
    assert vote is not None
    assert vote.phase == "commit"


@pytest.mark.asyncio
async def test_on_qc_commit_creates_entry(engine):
    qc = QuorumCertifier().collect(
        round=1, block_hash=b"bh", msg_type=MessageType.COMMIT,
        signatures=[("n0", b"sig")] * 4, quorum_size=4,
    )
    entry = await engine.on_qc_commit(qc)
    assert entry is not None
    assert entry.block.round == 1
    assert engine.state == "DECIDED"


@pytest.mark.asyncio
async def test_collect_votes_creates_qc(engine):
    await engine.start_round(round_num=1, is_leader=True)
    votes = [("n0", b"sig0"), ("n1", b"sig1"), ("n2", b"sig2"), ("n3", b"sig3"), ("n4", b"sig4")]
    qc = await engine.collect_votes(round=1, block_hash=b"bh", phase="prepare", votes=votes)
    assert qc is not None
    assert qc.is_valid(4)


@pytest.mark.asyncio
async def test_collect_votes_insufficient(engine):
    votes = [("n0", b"sig0"), ("n1", b"sig1")]  # only 2, need 4
    qc = await engine.collect_votes(round=1, block_hash=b"bh", phase="prepare", votes=votes)
    assert qc is None


@pytest.mark.asyncio
async def test_handle_timeout(engine):
    result = await engine.handle_timeout()
    assert result is False  # False when no view_change_handler configured
