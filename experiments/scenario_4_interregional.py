import asyncio
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector


async def run_clusters(num_clusters: int):
    orch = Orchestrator(
        num_clusters=num_clusters,
        nodes_per_cluster=5,
        devices_per_cluster=30,
        f=1,
        latency_ms=100.0,
    )
    result = await orch.run_experiment(num_rounds=20, warmup=5)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    mc.to_json(f"results/scenario_4_{num_clusters}_clusters.json")
    return mc.avg_latency()


async def run():
    sizes = [2, 4, 8, 12, 16]
    for s in sizes:
        lat = await run_clusters(s)
        print(f"  {s:2d} clusters — avg consensus latency: {lat:.3f}s")


if __name__ == "__main__":
    asyncio.run(run())
