import pytest
from simulator.interregional import InterRegionalManager


@pytest.mark.asyncio
async def test_interregional_manager_init():
    manager = InterRegionalManager(n=3, f=1)
    assert manager is not None
    assert manager.n == 3


@pytest.mark.asyncio
async def test_vrf_election_deterministic():
    manager = InterRegionalManager(n=3, f=1)
    candidates = ["c0", "c1", "c2"]
    seed = b"test_seed"
    leader1 = manager._elect_leader_vrf(candidates, seed)
    leader2 = manager._elect_leader_vrf(candidates, seed)
    assert leader1 == leader2
    assert leader1 in candidates


@pytest.mark.asyncio
async def test_vrf_election_different_seeds():
    manager = InterRegionalManager(n=3, f=1)
    candidates = ["c0", "c1", "c2"]
    leader_a = manager._elect_leader_vrf(candidates, b"seed_a")
    leader_b = manager._elect_leader_vrf(candidates, b"seed_b")
    assert leader_a in candidates
    assert leader_b in candidates


@pytest.mark.asyncio
async def test_vrf_election_fairness():
    manager = InterRegionalManager(n=3, f=1)
    candidates = ["c0", "c1", "c2"]
    leaders = set()
    for i in range(20):
        seed = f"round_{i}".encode()
        leader = manager._elect_leader_vrf(candidates, seed)
        leaders.add(leader)
    assert len(leaders) > 1, "VRF should distribute leadership across candidates"
