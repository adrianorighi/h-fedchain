import pytest
from simulator.cluster import Cluster
from simulator.cloud import CloudComponent
from simulator.experiment_runner import ExperimentRunner


@pytest.mark.asyncio
async def test_experiment_runner_initialization():
    clusters = [
        Cluster("c0", nodes_per_cluster=2, devices_per_cluster=3, f=0, latency_ms=1.0),
        Cluster("c1", nodes_per_cluster=2, devices_per_cluster=3, f=0, latency_ms=1.0),
    ]
    cloud = CloudComponent()
    runner = ExperimentRunner(clusters, cloud, f=0)
    assert runner is not None
    assert len(runner.clusters) == 2


@pytest.mark.asyncio
async def test_multi_cluster_run():
    clusters = [
        Cluster("c0", nodes_per_cluster=2, devices_per_cluster=3, f=0, latency_ms=1.0),
        Cluster("c1", nodes_per_cluster=2, devices_per_cluster=3, f=0, latency_ms=1.0),
    ]
    cloud = CloudComponent()
    runner = ExperimentRunner(clusters, cloud, f=0)
    result = await runner.run_experiment(num_rounds=3)
    assert result is not None
    assert len(result) == 3
    required = {"bytes_edge_fog", "bytes_fog_inter", "bytes_fog_cloud",
                "network_bytes", "bytes_fog_intra"}
    for r in result:
        assert "latency" in r
        assert "consensus_time_ms" in r
        assert "block_size_bytes" in r
        assert "round" in r
        assert required <= set(r), f"Missing directional byte keys: {required - set(r)}"
        assert r["bytes_edge_fog"] > 0
        assert r["bytes_fog_intra"] > 0
        assert r["bytes_fog_inter"] > 0
        assert r["bytes_fog_cloud"] > 0
        assert r["network_bytes"] == (
            r["bytes_edge_fog"] + r["bytes_fog_intra"]
            + r["bytes_fog_inter"] + r["bytes_fog_cloud"]
        )
