import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent
from simulator.experiment_runner import ExperimentRunner
from experiments.metrics import MetricsCollector

async def run_multi_cluster(num_clusters: int, num_rounds: int = 10):
    clusters = [Cluster(cluster_id=f"c{i}", nodes_per_cluster=4,
                        devices_per_cluster=20, f=1, latency_ms=100.0)
                for i in range(num_clusters)]
    runner = ExperimentRunner(clusters, InterRegionalManager(), CloudComponent())
    metrics = await runner.run_experiment(num_rounds=num_rounds)
    mc = MetricsCollector()
    for m in metrics:
        mc.add_round(m)
    mc.to_json(f"results/scenario_4_{num_clusters}_clusters.json")
    avg_active = sum(m["n_active_clusters"] for m in metrics) / len(metrics)
    print(f"  {num_clusters:2d} clusters — rounds={len(metrics)}  "
          f"avg_active={avg_active:.1f}  "
          f"latency={mc.avg_latency():.3f}s")
    return mc

async def run():
    for s in [2, 4, 8, 12, 16]:
        await run_multi_cluster(s, num_rounds=5)

if __name__ == "__main__":
    asyncio.run(run())
