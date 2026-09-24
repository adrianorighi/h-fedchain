import asyncio
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector
from monitoring.tracer import Tracer
from monitoring.anomaly import AnomalyDetector

async def run_variant(variant: str, num_rounds: int = 20, tracer=None, quick=False) -> dict:
    if quick:
        num_rounds = 2
    orch = Orchestrator(num_clusters=4, nodes_per_cluster=5,
                        devices_per_cluster=50, f=1, latency_ms=10.0,
                        use_dataset=False, variant=variant,
                        snark_prove=True,
                        tracer=tracer)
    result = await orch.run_experiment(num_rounds=num_rounds, warmup=2)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    print(f"  {variant:8s} — latency={mc.avg_latency():.3f}s  "
          f"block={mc.block_size_bytes():.0f}B  "
          f"consensus_time={mc.consensus_time_ms():.1f}ms  "
          f"compliance={mc.compliance_completeness():.2%}")
    mc.to_json(f"results/scenario_5_{variant}.json")
    return mc

async def run(tracer=None, quick=False):
    collectors = []
    for v in ["no_zkp", "snark", "stark", "full"]:
        mc = await run_variant(v, tracer=tracer, quick=quick)
        collectors.append(mc)
    return collectors[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Enable tracing and anomaly detection")
    parser.add_argument("--quick", action="store_true", help="Reduzido (2 rounds por variante)")
    args = parser.parse_args()
    tracer = Tracer() if args.monitor else None
    mc = asyncio.run(run(tracer=tracer, quick=args.quick))
    if args.monitor:
        detector = AnomalyDetector()
        for m in mc._records:
            alerts = detector.observe(m)
            for a in alerts:
                print(f"  [!] {a['severity']:8s} {a['rule']} — round {a.get('round','?')}")
        print(f"  Trace summary: {tracer.trace_summary()}")
        tracer.to_json("results/scenario_5_trace.json")
