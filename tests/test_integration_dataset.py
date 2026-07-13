import pytest
from simulator.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_runs_with_dataset():
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=2,
        devices_per_cluster=4,
        f=0,
        latency_ms=5.0,
        use_dataset=True,
        dataset_max_records=50,
    )
    result = await orch.run_experiment(num_rounds=3, warmup=1)
    assert len(result.round_metrics) == 3


@pytest.mark.asyncio
async def test_orchestrator_real_gradients_produce_ledger():
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=2,
        devices_per_cluster=4,
        f=0,
        latency_ms=5.0,
        use_dataset=True,
        dataset_max_records=50,
    )
    await orch.run_experiment(num_rounds=3, warmup=1)
    for node in orch.nodes:
        assert node.ledger.get_height() > 0
        assert node.ledger.verify_chain()


@pytest.mark.asyncio
async def test_orchestrator_synthetic_path_still_works():
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=2,
        devices_per_cluster=4,
        f=0,
        latency_ms=5.0,
    )
    result = await orch.run_experiment(num_rounds=3, warmup=1)
    assert len(result.round_metrics) == 3
