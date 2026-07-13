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
