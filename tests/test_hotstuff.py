import pytest
from core.hotstuff.quorum import QuorumCertifier
from hfc_types.block import QuorumCertificate as QC
from hfc_types.messages import MessageType


@pytest.fixture
def qc_certifier():
    return QuorumCertifier()


def test_collect_votes_until_quorum(qc_certifier):
    n, f = 5, 1
    quorum = n - f  # 4
    qc = qc_certifier.collect(
        round=1,
        block_hash=b"bh",
        msg_type=MessageType.PREPARE,
        signatures=[(f"n{i}", b"sig") for i in range(quorum)],
        quorum_size=quorum,
    )
    assert qc.is_valid(quorum)


def test_collect_insufficient_votes(qc_certifier):
    n, f = 5, 1
    quorum = n - f  # 4
    with pytest.raises(ValueError, match="quorum"):
        qc_certifier.collect(
            round=1,
            block_hash=b"bh",
            msg_type=MessageType.PREPARE,
            signatures=[(f"n{i}", b"sig") for i in range(quorum - 1)],
            quorum_size=quorum,
        )


def test_quorum_size_calculation(qc_certifier):
    assert qc_certifier.quorum_size(5) == 4  # ceil(2*5/3) + 1
    assert qc_certifier.quorum_size(4) == 3
    assert qc_certifier.quorum_size(7) == 5


from core.hotstuff.engine import HotStuffEngine
from core.pki import generate_keypair
from hfc_types.block import Block


class TestHotStuffEngine:
    @pytest.fixture
    def engine(self):
        sk, vk = generate_keypair()
        return HotStuffEngine(
            node_id="n0",
            sk=sk,
            vk=vk,
            peers=["n0", "n1", "n2", "n3", "n4"],
            n=5,
            f=1,
        )

    @pytest.mark.asyncio
    async def test_prepare_emits_vote(self, engine):
        await engine.start_round(round_num=1, is_leader=False)
        proposal = Block(
            round=1,
            gradient_hash=b"gh",
            qc_commit=None,
            stark_proof=None,
            accepted_devices=[],
            rejected_devices=[],
            timestamp=100.0,
            prev_hash=b"\x00" * 32,
        )
        vote = await engine.on_prepare(proposal)
        assert vote is not None
        assert vote.phase == "prepare"
        assert vote.round == 1

    @pytest.mark.asyncio
    async def test_double_prepare_returns_none(self, engine):
        await engine.start_round(round_num=1, is_leader=False)
        proposal = Block(
            round=1, gradient_hash=b"gh", qc_commit=None,
            stark_proof=None, accepted_devices=[], rejected_devices=[],
            timestamp=100.0, prev_hash=b"\x00" * 32,
        )
        await engine.on_prepare(proposal)
        vote2 = await engine.on_prepare(proposal)
        assert vote2 is None

    @pytest.mark.asyncio
    async def test_full_flow(self, engine):
        await engine.start_round(round_num=1, is_leader=False)
        proposal = Block(
            round=1, gradient_hash=b"gh", qc_commit=None,
            stark_proof=None, accepted_devices=[], rejected_devices=[],
            timestamp=100.0, prev_hash=b"\x00" * 32,
        )
        vote = await engine.on_prepare(proposal)
        assert vote is not None

        qc_prepare = QuorumCertifier().collect(
            round=1, block_hash=proposal.hash,
            msg_type=MessageType.PREPARE,
            signatures=[("n0", b"s")] * 4,
            quorum_size=4,
        )
        v2 = await engine.on_pre_commit(qc_prepare)
        assert v2 is not None

        qc_pre_commit = QuorumCertifier().collect(
            round=1, block_hash=proposal.hash,
            msg_type=MessageType.PRE_COMMIT,
            signatures=[("n0", b"s")] * 4,
            quorum_size=4,
        )
        v3 = await engine.on_commit(qc_pre_commit)
        assert v3 is not None

        qc_commit = QuorumCertifier().collect(
            round=1, block_hash=proposal.hash,
            msg_type=MessageType.COMMIT,
            signatures=[("n0", b"s")] * 4,
            quorum_size=4,
        )
        entry = await engine.on_qc_commit(qc_commit)
        assert entry is not None
        assert engine.state == "DECIDED"


class TestViewChange:
    @pytest.fixture
    def view_change(self):
        from core.hotstuff.view_change import ViewChangeHandler
        return ViewChangeHandler(n=5, f=1)

    def test_detect_leader_failure(self, view_change):
        assert view_change.should_change_view(timeout=True) is True

    def test_no_change_when_leader_ok(self, view_change):
        assert view_change.should_change_view(timeout=False) is False

    def test_next_leader_rotation(self, view_change):
        node_ids = ["n0", "n1", "n2", "n3", "n4"]
        leaders = []
        for i in range(5):
            leaders.append(view_change.next_leader(node_ids))
        assert len(set(leaders)) == 5
        assert leaders[0] == "n1"

    def test_record_highest_qc(self, view_change):
        view_change.record_highest_qc(round=5, qc=b"qc_data")
        assert view_change._highest_qc == (5, b"qc_data")
        view_change.record_highest_qc(round=3, qc=b"older")
        assert view_change._highest_qc == (5, b"qc_data")

    def test_create_view_change_message(self, view_change):
        from core.pki import generate_keypair
        sk, vk = generate_keypair()
        view_change.record_highest_qc(round=1, qc=b"some_qc")
        msg = view_change.create_view_change(node_id="n3", new_view=6, sk=sk)
        assert msg.new_view == 6
        assert msg.node_id == "n3"
        assert msg.highest_qc == (1, b"some_qc")
        assert len(msg.signature) > 0

    def test_create_new_view_message(self, view_change):
        qcs = [b"qc1", b"qc2"]
        msg = view_change.create_new_view(leader_id="n1", new_view=6, qc_set=qcs)
        assert msg.new_view == 6
        assert len(msg.qc_set) == 2
        assert msg.leader_id == "n1"


from core.pki import generate_keypair, sign as pki_sign


def test_collect_verifies_signatures(qc_certifier):
    n, f = 5, 1
    quorum = n - f
    keys = {f"n{i}": generate_keypair() for i in range(n)}
    vk_map = {nid: vk for nid, (sk, vk) in keys.items()}
    msg = str(1).encode() + b"bh" + MessageType.PREPARE.name.encode()
    sigs = [(nid, pki_sign(sk, msg)) for nid, (sk, vk) in keys.items()][:quorum]
    qc = qc_certifier.collect(
        round=1, block_hash=b"bh",
        msg_type=MessageType.PREPARE,
        signatures=sigs,
        quorum_size=quorum,
        vk_map=vk_map,
    )
    assert qc.is_valid(quorum, vk_map)


def test_collect_rejects_bad_signature(qc_certifier):
    n, f = 5, 1
    quorum = n - f
    keys = {f"n{i}": generate_keypair() for i in range(n)}
    vk_map = {nid: vk for nid, (sk, vk) in keys.items()}
    sigs = [(f"n{i}", b"bad_sig") for i in range(quorum)]
    with pytest.raises(ValueError, match="Invalid signature"):
        qc_certifier.collect(
            round=1, block_hash=b"bh",
            msg_type=MessageType.PREPARE,
            signatures=sigs,
            quorum_size=quorum,
            vk_map=vk_map,
        )
