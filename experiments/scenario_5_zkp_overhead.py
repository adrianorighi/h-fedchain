import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector

async def run_variant(variant: str, num_rounds: int = 5) -> dict:
    orch = Orchestrator(num_clusters=1, nodes_per_cluster=5,
                        devices_per_cluster=10, f=1, latency_ms=5.0,
                        use_dataset=False, variant=variant)
    result = await orch.run_experiment(num_rounds=num_rounds)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    print(f"  {variant:8s} — latency={mc.avg_latency():.3f}s  "
          f"block={mc.block_size_bytes():.0f}B  "
          f"consensus_time={mc.consensus_time_ms():.1f}ms  "
          f"compliance={mc.compliance_completeness():.2%}")
    mc.to_json(f"results/scenario_5_{variant}.json")
    return mc

async def run():
    collectors = []
    for v in ["no_zkp", "snark", "stark", "full"]:
        mc = await run_variant(v)
        collectors.append(mc)
    return collectors[-1]

if __name__ == "__main__":
    asyncio.run(run())
