import asyncio
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector


async def run_variant(variant: str):
    orch = Orchestrator(
        num_clusters=4,
        nodes_per_cluster=5,
        devices_per_cluster=50,
        f=1,
        latency_ms=10.0,
    )
    result = await orch.run_experiment(num_rounds=20, warmup=5)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    mc.to_json(f"results/scenario_5_{variant}.json")
    return mc.avg_latency()


async def run():
    variants = ["no_zkp", "snark_only", "stark_only", "both"]
    for v in variants:
        lat = await run_variant(v)
        print(f"  {v:12s} — avg latency: {lat:.3f}s")


if __name__ == "__main__":
    asyncio.run(run())
