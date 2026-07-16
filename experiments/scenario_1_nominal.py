import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.orchestrator import Orchestrator
from experiments.metrics import MetricsCollector

async def run():
    orch = Orchestrator(num_clusters=2, nodes_per_cluster=5,
                        devices_per_cluster=20, f=1, latency_ms=10.0)
    result = await orch.run_experiment(num_rounds=30, warmup=10)
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
    asyncio.run(run())
