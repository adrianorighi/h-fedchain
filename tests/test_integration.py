import pytest
from simulator.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_nominal_round_consensus():
    """Full round with n=5, f=1, 10 rounds, no adversaries -> SC=1.0"""
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=5,
        devices_per_cluster=20,
        f=1,
        latency_ms=5.0,
    )
    result = await orch.run_experiment(num_rounds=10, warmup=2)
    all_committed = all(
        m.get("qc_emitted", True) for m in result.round_metrics
    )
    assert all_committed, "Not all rounds committed"


@pytest.mark.asyncio
async def test_ledger_integrity():
    """After experiment, all nodes have same chain height"""
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=5,
        devices_per_cluster=10,
        f=1,
        latency_ms=5.0,
    )
    await orch.run_experiment(num_rounds=5, warmup=2)
    heights = [n.ledger.get_height() for n in orch.nodes]
    assert all(h == heights[0] for h in heights), "Ledgers diverged"


@pytest.mark.asyncio
async def test_ledger_chain_valid():
    """Ledger hash chain is valid after experiment"""
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=5,
        devices_per_cluster=10,
        f=1,
        latency_ms=5.0,
    )
    await orch.run_experiment(num_rounds=5, warmup=2)
    for node in orch.nodes:
        assert node.ledger.verify_chain(), f"Chain invalid on {node.node_id}"


@pytest.mark.asyncio
async def test_multi_node_vrf_and_consensus():
    """Multi-node run_round: VRF elects leader, consensus produces QC entry"""
    orch = Orchestrator(
        nodes_per_cluster=4,
        devices_per_cluster=4,
        f=1,
        latency_ms=1.0,
    )
    orch.setup()
    result = await orch.run_round(round_num=1)
    assert result["qc_emitted"] is True
    assert result["leader"] is not None
    assert result["latency"] > 0
    heights = [n.ledger.get_height() for n in orch.nodes]
    assert len(set(heights)) == 1, "All ledgers should have same height"


@pytest.mark.asyncio
async def test_multi_round_multi_node_consensus():
    """3 rounds, 4 nodes: each round elects a leader, consensus commits"""
    orch = Orchestrator(
        nodes_per_cluster=4,
        devices_per_cluster=8,
        f=1,
        latency_ms=2.0,
    )
    orch.setup()
    leaders = set()
    for rnd in range(1, 4):
        result = await orch.run_round(round_num=rnd)
        assert result["qc_emitted"] is True, f"Round {rnd} failed QC"
        leaders.add(result["leader"])
    assert len(leaders) >= 1
    heights = [n.ledger.get_height() for n in orch.nodes]
    assert all(h == 3 for h in heights), f"All ledgers must have height 3, got {heights}"
