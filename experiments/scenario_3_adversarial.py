import asyncio
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector
from monitoring.tracer import Tracer
from monitoring.anomaly import AnomalyDetector

async def run_adv(adversarial_ratio: float, tracer=None, use_dataset=False, quick=False, rounds=None):
    rounds = 3 if quick else (rounds or 50)
    warmup = 1 if quick else 1
    orch = Orchestrator(num_clusters=4, nodes_per_cluster=5,
                        devices_per_cluster=50, f=1, latency_ms=10.0,
                        variant="full",
                        use_dataset=use_dataset,
                        adversarial_ratio=adversarial_ratio,
                        snark_prove=True,
                        tracer=tracer)
    result = await orch.run_experiment(num_rounds=rounds, warmup=warmup)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    pct = int(adversarial_ratio * 100)
    print(f"  \u03c1={pct:2d}% — DR={mc.detection_rate():.2%}  "
          f"FPR={mc.false_positive_rate():.2%}  "
          f"SC={mc.consensus_success_rate():.2%}  "
          f"Compliance={mc.compliance_completeness():.2%}")
    mc.to_json(f"results/scenario_3_adv_{pct}.json")
    return mc

async def run(tracer=None, use_dataset=False, quick=False):
    collectors = []
    ratios = [0.0, 0.10, 0.20, 0.30, 0.40] if not quick else [0.0, 0.20]
    for r in ratios:
        mc = await run_adv(r, tracer=tracer, use_dataset=use_dataset, quick=quick)
        collectors.append(mc)
    return collectors[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Enable tracing and anomaly detection")
    parser.add_argument("--real", action="store_true", dest="real", help="Use PTB-XL dataset")
    parser.add_argument("--quick", action="store_true", help="Reduzido (3 rounds, 2 ratios)")
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
        tracer.to_json("results/scenario_3_trace.json")
