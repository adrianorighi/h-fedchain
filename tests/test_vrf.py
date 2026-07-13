from hashlib import sha256
import pytest
from core.vrf.election import VRFLeaderElection
from hfc_types.messages import VRFMessage


def make_vrf(node_id, seed, sk_map):
    sk = sk_map[node_id]
    y = sha256(sk + seed).digest()
    proof = sha256(b"vrf_proof:" + sk + seed).digest()
    return VRFMessage(node_id, 1, y, proof)


def test_elect_returns_lowest_y():
    seed = b"seed_1"
    sk_map = {"A": b"sk_A", "B": b"sk_B", "C": b"sk_C"}
    vk_map = sk_map
    election = VRFLeaderElection()
    candidates = []
    expected_leader = None
    min_y = None
    for n in ["A", "B", "C"]:
        y, proof = election.evaluate(sk_map[n], seed)
        candidates.append(VRFMessage(n, 1, y, proof))
        if min_y is None or y < min_y:
            min_y = y
            expected_leader = n
    leader = election.elect(candidates, seed, vk_map)
    assert leader == expected_leader, f"Expected {expected_leader}, got {leader}"


def test_elect_rejects_invalid_proof():
    seed = b"seed_1"
    sk_map = {"A": b"sk_A", "B": b"sk_B", "C": b"sk_C"}
    vk_map = sk_map
    candidates = [
        VRFMessage("A", 1, sha256(b"sk_A" + seed).digest(), b"bad_proof"),
    ]
    for n in ["B", "C"]:
        candidates.append(make_vrf(n, seed, sk_map))
    election = VRFLeaderElection()
    leader = election.elect(candidates, seed, vk_map)
    assert leader in ("B", "C"), "A with bad proof must not be elected"


def test_elect_empty_raises():
    election = VRFLeaderElection()
    with pytest.raises(ValueError, match="No valid candidates"):
        election.elect([], b"seed", {})


def test_evaluate_and_verify():
    election = VRFLeaderElection()
    sk = b"my_secret"
    seed = b"round_42"
    y, proof = election.evaluate(sk, seed)
    assert election.verify(sk, seed, y, proof) is True
