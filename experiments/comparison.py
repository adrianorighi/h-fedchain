"""
Comparison: H-FedChain vs FedSDM vs FLCoin — LaTeX table output.
"""
import asyncio
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfc_types.messages import Gradient
from simulator.orchestrator import Orchestrator
from baselines.fed_sdm import FedSDM
from baselines.flcoin import FLCoin
from experiments.metrics import MetricsCollector


async def run_hfc(num_rounds=10, warmup=3, devices=20, clusters=1,
                  nodes=5, f=1, latency=10.0, adv_ratio=0.0):
    orch = Orchestrator(num_clusters=clusters, nodes_per_cluster=nodes,
                        devices_per_cluster=devices, f=f, latency_ms=latency,
                        adversarial_ratio=adv_ratio)
    return await orch.run_experiment(num_rounds, warmup)


async def run_fed_sdm(num_rounds=10, warmup=3, devices=20, nodes=5, f=1):
    baseline = FedSDM(num_nodes=nodes, f=f)
    grads_per_round = []
    for r in range(num_rounds + warmup):
        grads = [Gradient(node_id=f"d{i}", round=r, data=np.random.randn(10).tolist())
                 for i in range(devices)]
        grads_per_round.append(grads)
    return await baseline.run_experiment(num_rounds, grads_per_round, warmup)


async def run_flcoin(num_rounds=10, warmup=3, devices=20, nodes=5, f=1, committee=3):
    baseline = FLCoin(num_nodes=nodes, f=f, committee_size=committee)
    grads_per_round = []
    for r in range(num_rounds + warmup):
        grads = [Gradient(node_id=f"d{i}", round=r, data=np.random.randn(10).tolist())
                 for i in range(devices)]
        grads_per_round.append(grads)
    return await baseline.run_experiment(num_rounds, grads_per_round, warmup)


def build_latex_table(results: dict, metrics: list[tuple[str, str]]):
    lines = ["\\begin{table}[h]", "\\centering",
             "\\begin{tabular}{l" + "r" * len(results) + "}",
             "\\toprule"]
    header = "Métrica"
    for name in results:
        header += f" & \\textbf{{{name}}}"
    header += " \\\\ \\midrule"
    lines.append(header)
    for label, key in metrics:
        row = label
        for name in results:
            mc = results[name]
            all_m = mc.all_metrics()
            val = all_m.get(key, 0)
            if key in ("detection_rate", "false_positive_rate", "consensus_success",
                       "compliance", "vrf_uniformity"):
                row += f" & {val:.1%}"
            elif key in ("block_size_bytes", "comm_overhead_bytes"):
                row += f" & {val:.0f}"
            elif key in ("avg_latency",):
                row += f" & {val:.4f}"
            elif key in ("consensus_time_ms",):
                row += f" & {val:.1f}"
            else:
                row += f" & {val:.2f}"
        row += " \\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparação entre sistemas — Cenário Nominal}")
    lines.append("\\label{tab:comparison_nominal}")
    lines.append("\\end{table}")
    return "\n".join(lines)


async def compare(scenario: str):
    METRICS = [
        ("Latência média (s)", "avg_latency"),
        ("Taxa de sucesso consenso", "consensus_success"),
        ("Taxa de detecção (DR)", "detection_rate"),
        ("Taxa de falso positivo (FPR)", "false_positive_rate"),
        ("Throughput (blocos/min)", "throughput"),
        ("Tamanho do bloco (bytes)", "block_size_bytes"),
        ("Tempo de consenso (ms)", "consensus_time_ms"),
        ("Uniformidade VRF", "vrf_uniformity"),
        ("Completude", "compliance"),
    ]

    if scenario == "nominal":
        runs = {
            "H-FedChain": run_hfc(num_rounds=10, warmup=3, devices=20, clusters=2, nodes=5, f=1),
            "FedSDM": run_fed_sdm(num_rounds=10, warmup=3, devices=20, nodes=5, f=1),
            "FLCoin": run_flcoin(num_rounds=10, warmup=3, devices=20, nodes=5, f=1, committee=3),
        }
    elif scenario == "adversarial":
        runs = {
            "H-FedChain": run_hfc(num_rounds=10, warmup=3, devices=20, clusters=4,
                                  nodes=5, f=1, adv_ratio=0.2),
            "FedSDM": run_fed_sdm(num_rounds=10, warmup=3, devices=20, nodes=5, f=1),
            "FLCoin": run_flcoin(num_rounds=10, warmup=3, devices=20, nodes=5, f=1, committee=3),
        }
    else:
        runs = {"H-FedChain": run_hfc()}

    results = {}
    for name, coro in runs.items():
        print(f"  Executando {name}...")
        result = await coro
        mc = MetricsCollector()
        for m in result.round_metrics:
            mc.add_round(m)
        results[name] = mc
        mc.to_json(f"results/comparison_{scenario}_{name.lower().replace('-', '_')}.json")

    print(f"\n=== Comparação: {scenario} ===\n")
    print(build_latex_table(results, METRICS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="nominal",
                        help="nominal | scalability | adversarial | interregional | zkp_overhead")
    args = parser.parse_args()
    asyncio.run(compare(args.scenario))
