"""
Gera tabelas de resultados (CSV + LaTeX + Markdown) a partir dos JSONs de
cenário em results/<data>/ (1 execução por configuração, com as métricas
novas de CPU, memória, bytes por elo e provas) e, como referência
secundária, do batch antigo em results/runs/ (sem ρ=5%).

Uso:
    python -m experiments.generate_tables
    python -m experiments.generate_tables --dir results/24-09-2026
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.metrics import MetricsCollector
from experiments.analyze_results import (
    SCENARIO_CONFIGS, CONFIG_ROW_LABEL, TABLE_HEADERS, TABLE_METRICS,
    fmt_cell, compute_stats, core_latency, eval_criteria, _flatten,
)


def expected_file(scenario: str, lbl: str) -> str:
    if scenario == "scenario_1_nominal":
        return "scenario_1_nominal.json"
    if scenario == "scenario_2_scalability":
        return f"scenario_2_{lbl[1:]}.json"
    if scenario == "scenario_3_adversarial":
        return f"scenario_3_adv_{lbl[4:]}.json"
    if scenario == "scenario_4_interregional":
        return f"scenario_4_{lbl[1:]}_clusters.json"
    if scenario == "scenario_5_zkp_overhead":
        return f"scenario_5_{lbl}.json"
    raise KeyError(f"cenário desconhecido: {scenario}")


def _core_latency_record(r: dict) -> float:
    m = dict(r)
    m["avg_latency"] = r.get("latency", 0.0)
    return core_latency(m)


SERIES = {
    "avg_latency": lambda r: r.get("latency"),
    "core_latency": _core_latency_record,
    "consensus_time_ms": lambda r: r.get("consensus_time_ms"),
    "block_size_bytes": lambda r: r.get("block_size_bytes"),
    "comm_overhead_bytes": lambda r: r.get("comm_overhead_bytes"),
    "cpu_percent": lambda r: r.get("cpu_percent"),
    "memory_rss_bytes": lambda r: r.get("memory_rss_bytes"),
    "network_bytes": lambda r: r.get("network_bytes"),
    "mb_edge_fog": lambda r: r.get("bytes_edge_fog", 0) / 1e6,
    "mb_fog_inter": lambda r: r.get("bytes_fog_inter", 0) / 1e6,
    "mb_fog_cloud": lambda r: r.get("bytes_fog_cloud", 0) / 1e6,
    "proof_gen_cpu_ms": lambda r: r.get("proof_gen_cpu_ms"),
    "proof_verify_cpu_ms": lambda r: r.get("proof_verify_cpu_ms"),
    "proof_size_bytes": lambda r: r.get("proof_size_bytes"),
    "view_change_latency_ms": lambda r: r.get("view_change_latency_ms"),
    "snark_proofs_total": lambda r: r.get("snark_proofs_total"),
    "snark_verify_projected_ms": lambda r: r.get("snark_verify_projected_ms"),
}


def _series_for(path: str, records: list[dict]) -> list[float]:
    if path.startswith("stage_time_ms."):
        key = path.split(".", 1)[1]
        vals = [r.get("stage_time_ms", {}).get(key) for r in records]
        return [v for v in vals if isinstance(v, (int, float))]
    fn = SERIES.get(path)
    if fn is None:
        return []
    vals = []
    for r in records:
        v = fn(r)
        if isinstance(v, (int, float)):
            vals.append(v)
    return vals


def config_stats(records: list[dict]) -> tuple[dict, dict]:
    mc = MetricsCollector()
    for r in records:
        mc.add_round(r)
    am = mc.all_metrics()
    am["core_latency"] = core_latency(am)
    stats = {}
    for path, value in _flatten(am).items():
        series = _series_for(path, records)
        stats[path] = compute_stats(series) if series else compute_stats([value])
    return am, stats
