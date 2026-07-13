import asyncio
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector


async def run_scale(devices_per_cluster: int):
    orch = Orchestrator(
        num_clusters=4,
        nodes_per_cluster=5,
        devices_per_cluster=devices_per_cluster,
        f=1,
        latency_ms=10.0,
    )
    result = await orch.run_experiment(num_rounds=20, warmup=5)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    mc.to_json(f"results/scenario_2_{devices_per_cluster}.json")
    return mc.avg_latency()


async def run():
    sizes = [10, 25, 50, 75, 100]
    for s in sizes:
        lat = await run_scale(s)
        total_devices = s * 4
        print(f"  {total_devices:3d} devices — avg latency: {lat:.3f}s")


if __name__ == "__main__":
    asyncio.run(run())
