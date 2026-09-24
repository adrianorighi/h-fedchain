import asyncio
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector
from monitoring.tracer import Tracer
from monitoring.anomaly import AnomalyDetector

async def run(tracer=None, use_dataset=False, quick=False):
    rounds = 3 if quick else 30
    warmup = 1 if quick else 10
    orch = Orchestrator(num_clusters=2, nodes_per_cluster=5,
                         devices_per_cluster=20, f=1, latency_ms=10.0,
                        variant="full",
                        use_dataset=use_dataset,
                        snark_prove=True,
                        tracer=tracer)
    result = await orch.run_experiment(num_rounds=rounds, warmup=warmup)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    mc.to_json("results/scenario_1_nominal.json")
    print("Scenario 1 — Nominal")
    print(f"  Avg latency:       {mc.avg_latency():.4f}s")
    print(f"  Consensus success: {mc.consensus_success_rate():.2%}")
    print(f"  Block size:        {mc.block_size_bytes():.0f} bytes")
    print(f"  Consensus time:    {mc.consensus_time_ms():.1f} ms")
    print(f"  VRF uniformity:    {mc.vrf_election_uniformity():.3f}")
    print(f"  Compliance:        {mc.compliance_completeness():.2%}")
    return mc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Enable tracing and anomaly detection")
    parser.add_argument("--real", action="store_true", dest="real", help="Use PTB-XL dataset")
    parser.add_argument("--quick", action="store_true", help="Reduzido (3 rounds)")
    args = parser.parse_args()
    tracer = Tracer() if args.monitor else None
    mc = asyncio.run(run(tracer=tracer, use_dataset=args.real, quick=args.quick))
    if args.monitor:
        detector = AnomalyDetector()
        for m in mc._records:
            alerts = detector.observe(m)
            for a in alerts:
                print(f"  [!] {a['severity']:8s} {a['rule']} — round {a.get('round','?')}")
        print(f"  Trace summary: {tracer.trace_summary()}")
        tracer.to_json("results/scenario_1_trace.json")
