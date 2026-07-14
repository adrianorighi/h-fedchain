import pytest
from core.hotstuff.view_change import ViewChangeHandler
from core.pki import generate_keypair


@pytest.fixture
def vc():
    return ViewChangeHandler(n=5, f=1)


def test_should_change_view_on_timeout(vc):
    assert vc.should_change_view(timeout=True) is True


def test_no_change_without_timeout(vc):
    assert vc.should_change_view(timeout=False) is False


def test_next_leader_rotation(vc):
    node_ids = ["n0", "n1", "n2", "n3", "n4"]
    leaders = []
    for i in range(5):
        leaders.append(vc.next_leader(node_ids))
    assert len(set(leaders)) == 5
    assert leaders[0] == "n1"


def test_record_highest_qc(vc):
    vc.record_highest_qc(round=5, qc=b"qc_data")
    assert vc._highest_qc == (5, b"qc_data")


def test_record_highest_qc_ignores_older(vc):
    vc.record_highest_qc(round=5, qc=b"higher")
    vc.record_highest_qc(round=3, qc=b"lower")
    assert vc._highest_qc == (5, b"higher")


def test_create_view_change_with_qc(vc):
    sk, vk = generate_keypair()
    vc.record_highest_qc(round=1, qc=b"some_qc")
    msg = vc.create_view_change(node_id="n3", new_view=6, sk=sk)
    assert msg.new_view == 6
    assert msg.node_id == "n3"
    assert msg.highest_qc == (1, b"some_qc")
    assert len(msg.signature) > 0


def test_create_view_change_without_qc(vc):
    sk, vk = generate_keypair()
    msg = vc.create_view_change(node_id="n0", new_view=2, sk=sk)
    assert msg.highest_qc is None


def test_create_new_view(vc):
    qcs = [b"qc1", b"qc2"]
    msg = vc.create_new_view(leader_id="n1", new_view=6, qc_set=qcs)
    assert msg.new_view == 6
    assert len(msg.qc_set) == 2
    assert msg.leader_id == "n1"
