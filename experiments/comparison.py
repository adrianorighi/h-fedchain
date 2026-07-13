"""
Runs comparison between H-FedChain, FedSDM, and FLCoin for a given scenario.
Results are printed as a comparison table and saved to results/comparison_*.json.

Usage:
    python -m experiments.comparison --scenario nominal
"""
import asyncio
import argparse
import numpy as np
from hfc_types.messages import Gradient
from simulator.orchestrator import Orchestrator
from baselines.fed_sdm import FedSDM
from baselines.flcoin import FLCoin
from experiments.metrics import MetricsCollector


async def run_hfc(
    num_rounds=10, warmup=3, devices=20, clusters=1, nodes=5,
    f=1, latency=10.0, adv_ratio=0.0,
):
    orch = Orchestrator(
        num_clusters=clusters, nodes_per_cluster=nodes,
        devices_per_cluster=devices, f=f, latency_ms=latency,
        adversarial_ratio=adv_ratio,
    )
    return await orch.run_experiment(num_rounds, warmup)


async def run_fed_sdm(
    num_rounds=10, warmup=3, devices=20,
    nodes=5, f=1,
):
    baseline = FedSDM(num_nodes=nodes, f=f)
    grads_per_round = []
    for r in range(num_rounds + warmup):
        grads = [
            Gradient(node_id=f"d{i}", round=r, data=np.random.randn(10).tolist())
            for i in range(devices)
        ]
        grads_per_round.append(grads)
    return await baseline.run_experiment(num_rounds, grads_per_round, warmup)


async def run_flcoin(
    num_rounds=10, warmup=3, devices=20,
    nodes=5, f=1, committee=3,
):
    baseline = FLCoin(num_nodes=nodes, f=f, committee_size=committee)
    grads_per_round = []
    for r in range(num_rounds + warmup):
        grads = [
            Gradient(node_id=f"d{i}", round=r, data=np.random.randn(10).tolist())
            for i in range(devices)
        ]
        grads_per_round.append(grads)
    return await baseline.run_experiment(num_rounds, grads_per_round, warmup)


def print_table(results: dict):
    header = f"{'Metric':<25}"
    for name in results:
        header += f" {name:<15}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Avg Latency (s)", lambda mc: f"{mc.avg_latency():.4f}"),
        ("SC (%)", lambda mc: f"{mc.consensus_success_rate():.2%}"),
    ]
    for label, fn in metrics:
        row = f"{label:<25}"
        for name in results:
            row += f" {fn(results[name]):<15}"
        print(row)


async def compare(scenario: str):

    if scenario == "nominal":
        runs = {
            "H-FedChain": run_hfc(num_rounds=10, warmup=3, devices=20, clusters=2, nodes=5, f=1),
            "FedSDM": run_fed_sdm(num_rounds=10, warmup=3, devices=20, nodes=5, f=1),
            "FLCoin": run_flcoin(num_rounds=10, warmup=3, devices=20, nodes=5, f=1, committee=3),
        }
    else:
        runs = {"H-FedChain": run_hfc()}

    results = {}
    for name, coro in runs.items():
        print(f"  {name}...")
        result = await coro
        mc = MetricsCollector()
        for m in result.round_metrics:
            mc.add_round(m)
        results[name] = mc
        mc.to_json(f"results/comparison_{scenario}_{name.lower().replace('-', '_')}.json")

    print(f"\n=== Comparison: {scenario} ===\n")
    print_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="nominal", help="nominal | scalability | adversarial | interregional | zkp_overhead")
    args = parser.parse_args()
    asyncio.run(compare(args.scenario))
