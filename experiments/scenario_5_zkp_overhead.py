import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.orchestrator import Orchestrator


async def run_variant(variant: str, num_rounds: int = 5) -> dict:
    orch = Orchestrator(
        num_clusters=1,
        nodes_per_cluster=5,
        devices_per_cluster=10,
        f=1,
        latency_ms=5.0,
        use_dataset=False,
        variant=variant,
    )
    result = await orch.run_experiment(num_rounds=num_rounds)
    return {
        "variant": variant,
        "avg_latency": sum(m["latency"] for m in result.round_metrics) / len(result.round_metrics),
        "total_time": sum(m["latency"] for m in result.round_metrics),
        "rounds": len(result.round_metrics),
    }


async def main():
    variants = ["no_zkp", "snark", "stark", "full"]
    for v in variants:
        stats = await run_variant(v, num_rounds=5)
        print(f"{v}: avg_latency={stats['avg_latency']:.3f}s, total={stats['total_time']:.3f}s, rounds={stats['rounds']}")

if __name__ == "__main__":
    asyncio.run(main())
