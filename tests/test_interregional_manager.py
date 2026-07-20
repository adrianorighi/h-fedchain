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
