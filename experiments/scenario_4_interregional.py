import asyncio
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent
from simulator.experiment_runner import ExperimentRunner
from experiments.metrics import MetricsCollector
from monitoring.tracer import Tracer
from monitoring.anomaly import AnomalyDetector

async def run_multi_cluster(num_clusters: int, num_rounds: int = 10, tracer=None, quick=False):
    if quick:
        num_rounds = 2
    clusters = [Cluster(cluster_id=f"c{i}", nodes_per_cluster=5,
                        devices_per_cluster=30, f=1, latency_ms=50.0,
                        variant="full", snark_prove=True)
                for i in range(num_clusters)]
    runner = ExperimentRunner(clusters, CloudComponent(tracer=tracer), f=1,
                              interregional_latency_ms=100.0, jitter_ms=20.0,
                              tracer=tracer)
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

async def run(tracer=None, quick=False):
    collectors = []
    sizes = [2, 4, 8, 12, 16] if not quick else [4, 8]
    for s in sizes:
        mc = await run_multi_cluster(s, num_rounds=10, tracer=tracer, quick=quick)
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
        tracer.to_json("results/scenario_4_trace.json")
