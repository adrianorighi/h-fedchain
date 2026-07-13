import asyncio
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector


async def run():
    orch = Orchestrator(
        num_clusters=2,
        nodes_per_cluster=5,
        devices_per_cluster=20,
        f=1,
        latency_ms=10.0,
    )
    result = await orch.run_experiment(num_rounds=30, warmup=10)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    mc.to_json("results/scenario_1_nominal.json")
    print(f"Scenario 1 — Avg latency: {mc.avg_latency():.3f}s")
    print(f"Scenario 1 — SC: {mc.consensus_success_rate():.2%}")
    print(f"Scenario 1 — Throughput: {mc.throughput(120):.1f} blocks/min")


if __name__ == "__main__":
    asyncio.run(run())
