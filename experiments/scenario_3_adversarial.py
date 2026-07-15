import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector

async def run_adv(adversarial_ratio: float):
    orch = Orchestrator(num_clusters=4, nodes_per_cluster=5,
                        devices_per_cluster=50, f=2, latency_ms=10.0,
                        adversarial_ratio=adversarial_ratio)
    result = await orch.run_experiment(num_rounds=50, warmup=5)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    pct = int(adversarial_ratio * 100)
    print(f"  \u03c1={pct:2d}% — DR={mc.detection_rate():.2%}  "
          f"FPR={mc.false_positive_rate():.2%}  "
          f"SC={mc.consensus_success_rate():.2%}  "
          f"Compliance={mc.compliance_completeness():.2%}")
    mc.to_json(f"results/scenario_3_adv_{pct}.json")

async def run():
    for r in [0.0, 0.05, 0.10, 0.20, 0.30]:
        await run_adv(r)

if __name__ == "__main__":
    asyncio.run(run())
