import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector

async def run_scale(devices_per_cluster: int):
    orch = Orchestrator(num_clusters=4, nodes_per_cluster=5,
                        devices_per_cluster=devices_per_cluster,
                        f=1, latency_ms=10.0)
    result = await orch.run_experiment(num_rounds=20, warmup=5)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    total_devices = devices_per_cluster * 4
    print(f"  {total_devices:3d} devices — latency={mc.avg_latency():.3f}s  "
          f"throughput={mc.throughput(sum(r.get('latency',0) for r in result.round_metrics)):.0f} blocks/min  "
          f"consensus_time={mc.consensus_time_ms():.1f}ms")
    mc.to_json(f"results/scenario_2_{devices_per_cluster}.json")
    return mc

async def run():
    collectors = []
    for s in [10, 25, 50, 75, 100]:
        mc = await run_scale(s)
        collectors.append(mc)
    return collectors[-1]  # return last collector for summary

if __name__ == "__main__":
    asyncio.run(run())
