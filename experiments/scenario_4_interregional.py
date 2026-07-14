import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent
from simulator.experiment_runner import ExperimentRunner


async def run_multi_cluster(num_clusters: int, num_rounds: int = 10):
    clusters = [
        Cluster(
            cluster_id=f"c{i}",
            nodes_per_cluster=4,
            devices_per_cluster=20,
            f=1,
            latency_ms=100.0,
        )
        for i in range(num_clusters)
    ]
    runner = ExperimentRunner(
        clusters, InterRegionalManager(), CloudComponent(),
    )
    metrics = await runner.run_experiment(num_rounds=num_rounds)
    return metrics


async def run():
    sizes = [2, 4, 8, 12, 16]
    for s in sizes:
        metrics = await run_multi_cluster(s, num_rounds=5)
        avg_rounds = len(metrics)
        avg_active = sum(m["n_active_clusters"] for m in metrics) / len(metrics)
        print(f"  {s:2d} clusters — {avg_rounds} rounds, avg {avg_active:.1f} active clusters/round")


if __name__ == "__main__":
    asyncio.run(run())
