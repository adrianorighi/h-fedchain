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
    load_runs, metric_paths,
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


BYTE_KEYS = ("bytes_edge_fog", "bytes_fog_intra", "bytes_fog_inter",
             "bytes_fog_cloud")


def check_byte_invariant(fname: str, records: list[dict]) -> list[str]:
    warnings = []
    for i, r in enumerate(records):
        nb = r.get("network_bytes") or 0
        total = sum(r.get(k) or 0 for k in BYTE_KEYS)
        if nb != total:
            rnd = r.get("round", i)
            warnings.append(
                f"[AVISO] {fname} round {rnd}: network_bytes="
                f"{nb} != soma dos elos={total}")
    return warnings


def load_configs(results_dir: Path) -> tuple[dict, dict, list[str]]:
    stats, aggs, warnings = {}, {}, []
    for scenario, configs in SCENARIO_CONFIGS.items():
        stats[scenario], aggs[scenario] = {}, {}
        for c in configs:
            lbl = c["label"]
            path = results_dir / expected_file(scenario, lbl)
            if not path.exists():
                warnings.append(f"[AVISO] arquivo ausente: {path.name}")
                continue
            try:
                records = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON inválido em {path}: {e}")
            if not isinstance(records, list):
                raise SystemExit(
                    f"JSON não é uma lista de rounds em {path}")
            if not records:
                warnings.append(f"[AVISO] sem rounds: {path.name}")
                continue
            warnings += check_byte_invariant(path.name, records)
            am, st = config_stats(records)
            stats[scenario][lbl] = st
            aggs[scenario][lbl] = am
    return stats, aggs, warnings


STATS_FIELDS = ["scenario", "config", "metric", "mean", "std", "ci95",
                "cv", "min", "max", "n"]
CRITERIA_FIELDS = ["scenario", "config", "criterio", "valor", "limiar",
                   "atende", "tipo"]


def build_criteria(stats: dict):
    missing = [
        f"{s}/{c['label']}"
        for s, cfgs in SCENARIO_CONFIGS.items()
        for c in cfgs if c["label"] not in stats.get(s, {})
    ]
    if missing:
        return None, f"[AVISO] critérios pulados, configs faltando: {missing}"
    return eval_criteria(stats, {}), None


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stats_rows(stats: dict) -> list[dict]:
    rows = []
    for scenario, cfgs in stats.items():
        for lbl, metrics in cfgs.items():
            for m, st in metrics.items():
                rows.append({
                    "scenario": scenario, "config": lbl, "metric": m,
                    "mean": st["mean"], "std": st["std"], "ci95": st["ci95"],
                    "cv": st["cv"], "min": st["min"], "max": st["max"],
                    "n": st["n"],
                })
    return rows


def config_rows(stats: dict) -> list[dict]:
    rows = []
    for scenario, cfgs in stats.items():
        for lbl, metrics in cfgs.items():
            row = {"scenario": scenario, "config": lbl}
            row.update({p: st["mean"] for p, st in metrics.items()})
            rows.append(row)
    return rows


def config_fields(stats: dict) -> list[str]:
    keys = set()
    for cfgs in stats.values():
        for metrics in cfgs.values():
            keys.update(metrics.keys())
    return ["scenario", "config"] + sorted(keys)


def build_resultados_tex(stats: dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Resultados experimentais da arquitetura H-FedChain. "
        r"Média $\pm$ desvio padrão calculados sobre as rodadas de uma "
        r"única execução por configuração.}",
        r"\label{tab:resultados}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(TABLE_HEADERS) - 1) + "}",
        r"\toprule",
        " & ".join(TABLE_HEADERS) + r" \\ \midrule",
    ]
    for scenario, configs in SCENARIO_CONFIGS.items():
        lat_metric = ("core_latency" if scenario == "scenario_2_scalability"
                      else "avg_latency")
        for c in configs:
            lbl = c["label"]
            if lbl not in stats.get(scenario, {}):
                continue
            st = stats[scenario][lbl]
            row = [CONFIG_ROW_LABEL[scenario](c)]
            for m in TABLE_METRICS:
                key = lat_metric if m == "avg_latency" else m
                cell = fmt_cell(key, st.get(key, compute_stats([0.0])))
                row.append(cell.replace("%", r"\%"))
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)


OVERHEAD_COLS = [
    ("CPU (\\%)", "cpu_percent", lambda v: f"{v:.1f}"),
    ("Mem. (MB)", "memory_rss_bytes", lambda v: f"{v / 1e6:.1f}"),
    ("Rede (MB)", "network_bytes", lambda v: f"{v / 1e6:.2f}"),
    ("Edge$\\leftrightarrow$Fog (MB)", "mb_edge_fog", lambda v: f"{v:.2f}"),
    ("Fog$\\leftrightarrow$inter (MB)", "mb_fog_inter", lambda v: f"{v:.2f}"),
    ("Fog$\\leftrightarrow$Cloud (MB)", "mb_fog_cloud", lambda v: f"{v:.2f}"),
    ("Proof gen (s)", "proof_gen_cpu_ms", lambda v: f"{v / 1000:.2f}"),
    ("Proof verif. (ms)", "proof_verify_cpu_ms", lambda v: f"{v:.1f}"),
    ("Proof tam. (KB)", "proof_size_bytes", lambda v: f"{v / 1e3:.1f}"),
]


def build_overhead_tex(stats: dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Overhead de recursos por configuração. Média $\pm$ "
        r"desvio padrão sobre as rodadas da execução.}",
        r"\label{tab:overhead}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(OVERHEAD_COLS) + "}",
        r"\toprule",
        "Cenário & " + " & ".join(h for h, _, _ in OVERHEAD_COLS)
        + r" \\ \midrule",
    ]
    for scenario, configs in SCENARIO_CONFIGS.items():
        for c in configs:
            lbl = c["label"]
            if lbl not in stats.get(scenario, {}):
                continue
            st = stats[scenario][lbl]
            row = [CONFIG_ROW_LABEL[scenario](c)]
            for _, key, fn in OVERHEAD_COLS:
                mean = st.get(key, compute_stats([0.0]))["mean"]
                std = st.get(key, compute_stats([0.0]))["std"]
                row.append(f"{fn(mean)} $\\pm$ {fn(std)}")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)


COMPARISON_SYSTEMS = {"h_fedchain": "H-FedChain", "fedsdm": "FedSDM",
                      "flcoin": "FLCoin"}

COMPARISON_METRICS = [
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


def build_comparison(results_dir: Path):
    from experiments.comparison import build_latex_table
    collectors, warnings = {}, []
    for slug, display in COMPARISON_SYSTEMS.items():
        path = results_dir / f"comparison_nominal_{slug}.json"
        if not path.exists():
            warnings.append(f"[AVISO] comparação ausente: {path.name}")
            continue
        try:
            records = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"JSON inválido em {path}: {e}")
        if not isinstance(records, list):
            raise SystemExit(
                f"JSON não é uma lista de rounds em {path}")
        if not records:
            warnings.append(f"[AVISO] comparação vazia: {path.name}")
            continue
        mc = MetricsCollector()
        for r in records:
            mc.add_round(r)
        collectors[display] = mc
    if not collectors:
        return None, collectors, warnings
    tex = build_latex_table(collectors, COMPARISON_METRICS)
    return tex, collectors, warnings


def build_batch_reference() -> list[dict]:
    rows = []
    for scenario, configs in SCENARIO_CONFIGS.items():
        for c in configs:
            runs = load_runs(scenario, c["label"])
            if not runs:
                continue
            metrics = []
            for r in runs:
                m = dict(r["metrics"])
                m["core_latency"] = core_latency(m)
                metrics.append(m)
            for p in metric_paths(metrics):
                vals = [_flatten(m).get(p) for m in metrics]
                if vals and all(
                        isinstance(v, (int, float))
                        and not isinstance(v, bool) for v in vals):
                    st = compute_stats(vals)
                    rows.append({
                        "scenario": scenario, "config": c["label"],
                        "metric": p, "mean": st["mean"], "std": st["std"],
                        "n": st["n"],
                    })
    return rows
