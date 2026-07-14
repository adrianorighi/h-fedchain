import pytest
from core.vrf.election import VRFLeaderElection
from core.pki.ed25519 import generate_keypair
from hfc_types.messages import VRFMessage


def make_vrf(node_id, seed, sk):
    election = VRFLeaderElection()
    y, proof = election.evaluate(sk, seed)
    return VRFMessage(node_id=node_id, round=1, y=y, proof=proof)


def test_elect_returns_lowest_y():
    seed = b"round_seed_1"
    node_ids = ["A", "B", "C"]
    keys = {n: generate_keypair() for n in node_ids}
    sk_map = {n: sk for n, (sk, vk) in keys.items()}
    vk_map = {n: vk for n, (sk, vk) in keys.items()}

    election = VRFLeaderElection()
    candidates = []
    expected_leader = None
    min_y = None
    for n in node_ids:
        y, proof = election.evaluate(sk_map[n], seed)
        candidates.append(VRFMessage(n, 1, y, proof))
        if min_y is None or y < min_y:
            min_y = y
            expected_leader = n
    leader = election.elect(candidates, seed, vk_map)
    assert leader == expected_leader


def test_elect_rejects_invalid_proof():
    seed = b"round_seed_1"
    keys = {"A": generate_keypair(), "B": generate_keypair(), "C": generate_keypair()}
    sk_map = {n: sk for n, (sk, vk) in keys.items()}
    vk_map = {n: vk for n, (sk, vk) in keys.items()}

    election = VRFLeaderElection()
    bad_y, bad_proof = election.evaluate(sk_map["A"], seed)
    bad_proof = b"tampered"
    candidates = [
        VRFMessage("A", 1, bad_y, bad_proof),
        make_vrf("B", seed, sk_map["B"]),
        make_vrf("C", seed, sk_map["C"]),
    ]
    leader = election.elect(candidates, seed, vk_map)
    assert leader in ("B", "C"), "A with bad proof must not be elected"


def test_elect_empty_raises():
    election = VRFLeaderElection()
    with pytest.raises(ValueError, match="No valid candidates"):
        election.elect([], b"seed", {})


def test_evaluate_and_verify_roundtrip():
    election = VRFLeaderElection()
    sk, vk = generate_keypair()
    seed = b"round_42"
    y, proof = election.evaluate(sk, seed)
    assert election.verify(vk, seed, y, proof) is True


def test_verify_rejects_wrong_vk():
    election = VRFLeaderElection()
    sk1, vk1 = generate_keypair()
    _, vk2 = generate_keypair()
    seed = b"round_42"
    y, proof = election.evaluate(sk1, seed)
    assert election.verify(vk2, seed, y, proof) is False


def test_evaluate_deterministic():
    election = VRFLeaderElection()
    sk, _ = generate_keypair()
    seed = b"round_42"
    y1, proof1 = election.evaluate(sk, seed)
    y2, proof2 = election.evaluate(sk, seed)
    assert y1 == y2
    assert proof1 == proof2


def test_vrf_verify_against_elect():
    seed = b"round_seed_1"
    keys = {"A": generate_keypair(), "B": generate_keypair()}
    sk_map = {n: sk for n, (sk, vk) in keys.items()}
    vk_map = {n: vk for n, (sk, vk) in keys.items()}

    election = VRFLeaderElection()
    candidates = [make_vrf(n, seed, sk_map[n]) for n in ["A", "B"]]
    leader = election.elect(candidates, seed, vk_map)
    assert leader in ("A", "B")
