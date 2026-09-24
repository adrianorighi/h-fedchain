import asyncio
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector
from monitoring.tracer import Tracer
from monitoring.anomaly import AnomalyDetector

async def run_scale(devices_per_cluster: int, tracer=None, quick=False):
    rounds = 2 if quick else 20
    warmup = 1 if quick else 5
    orch = Orchestrator(num_clusters=4, nodes_per_cluster=5,
                        devices_per_cluster=devices_per_cluster,
                        f=1, latency_ms=10.0,
                        variant="full",
                        snark_prove=True,
                        tracer=tracer)
    result = await orch.run_experiment(num_rounds=rounds, warmup=warmup)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    total_devices = devices_per_cluster * 4
    print(f"  {total_devices:3d} devices — latency={mc.avg_latency():.3f}s  "
          f"throughput={mc.throughput(sum(r.get('latency',0) for r in result.round_metrics)):.0f} blocks/min  "
          f"consensus_time={mc.consensus_time_ms():.1f}ms")
    mc.to_json(f"results/scenario_2_{devices_per_cluster}.json")
    return mc

async def run(tracer=None, quick=False):
    collectors = []
    sizes = [10, 25, 50, 75, 100] if not quick else [10, 100]
    for s in sizes:
        mc = await run_scale(s, tracer=tracer, quick=quick)
        collectors.append(mc)
    return collectors[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Enable tracing and anomaly detection")
    parser.add_argument("--quick", action="store_true", help="Reduzido (2 rounds, 2 tamanhos)")
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
        tracer.to_json("results/scenario_2_trace.json")
