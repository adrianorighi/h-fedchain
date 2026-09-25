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


def _pct(x):
    return f"{x:.1%}"


def _m(stats, scenario, lbl, metric, pct=False, nd=3):
    st = stats.get(scenario, {}).get(lbl, {}).get(metric)
    if st is None:
        return "—"
    v = st["mean"]
    return _pct(v) if pct else f"{v:.{nd}f}"


def build_markdown(stats, criteria, warnings, collectors,
                   results_dir: Path) -> str:
    L = []
    A = L.append
    A(f"# Análise de Resultados — H-FedChain ({results_dir.name})")
    A("")
    A("## Base dos dados")
    A("")
    A(f"`{results_dir}/` — 5 cenários, 20 configurações, 1 execução por "
      "configuração (média ± desvio padrão sobre as rodadas). Inclui as "
      "métricas novas de CPU, memória, bytes por elo e provas. Referência "
      "secundária: batch antigo `results/runs/` (5 execuções, ρ=5% "
      "excluído), em `referencia_batch_antigo.csv`.")
    A("")
    if warnings:
        A("## Avisos")
        A("")
        for w in warnings:
            A(f"- {w}")
        A("")

    A("## Resumo executivo")
    A("")
    A("| Config | Lat. (s) | SC | DR | FPR | T_BFT (ms) | I_ledger | D_rep |")
    A("|---|---|---|---|---|---|---|---|")
    A("| C1 nominal | " + " | ".join([
        _m(stats, "scenario_1_nominal", "nominal", "avg_latency"),
        _m(stats, "scenario_1_nominal", "nominal", "consensus_success", pct=True),
        "—",
        _m(stats, "scenario_1_nominal", "nominal", "false_positive_rate", pct=True),
        _m(stats, "scenario_1_nominal", "nominal", "consensus_time_ms", nd=1),
        _m(stats, "scenario_1_nominal", "nominal", "ledger_integrity", pct=True),
        _m(stats, "scenario_1_nominal", "nominal", "state_divergence_rate", pct=True),
    ]) + " |")
    for c in SCENARIO_CONFIGS["scenario_2_scalability"]:
        lbl = c["label"]
        A(f"| C2 {lbl} | " + " | ".join([
            _m(stats, "scenario_2_scalability", lbl, "core_latency"),
            _m(stats, "scenario_2_scalability", lbl, "consensus_success", pct=True),
            "—", "—",
            _m(stats, "scenario_2_scalability", lbl, "consensus_time_ms", nd=1),
            _m(stats, "scenario_2_scalability", lbl, "ledger_integrity", pct=True),
            _m(stats, "scenario_2_scalability", lbl, "state_divergence_rate", pct=True),
        ]) + " |")
    for c in SCENARIO_CONFIGS["scenario_3_adversarial"]:
        lbl = c["label"]
        A(f"| C3 {lbl} | " + " | ".join([
            _m(stats, "scenario_3_adversarial", lbl, "avg_latency"),
            _m(stats, "scenario_3_adversarial", lbl, "consensus_success", pct=True),
            _m(stats, "scenario_3_adversarial", lbl, "detection_rate", pct=True),
            _m(stats, "scenario_3_adversarial", lbl, "false_positive_rate", pct=True),
            "—",
            _m(stats, "scenario_3_adversarial", lbl, "ledger_integrity", pct=True),
            _m(stats, "scenario_3_adversarial", lbl, "state_divergence_rate", pct=True),
        ]) + " |")
    for c in SCENARIO_CONFIGS["scenario_4_interregional"]:
        lbl = c["label"]
        A(f"| C4 {lbl} | " + " | ".join([
            _m(stats, "scenario_4_interregional", lbl, "avg_latency"),
            _m(stats, "scenario_4_interregional", lbl, "consensus_success", pct=True),
            "—", "—",
            _m(stats, "scenario_4_interregional", lbl, "consensus_time_ms", nd=1),
            _m(stats, "scenario_4_interregional", lbl, "ledger_integrity", pct=True),
            _m(stats, "scenario_4_interregional", lbl, "state_divergence_rate", pct=True),
        ]) + " |")
    for v in SCENARIO_CONFIGS["scenario_5_zkp_overhead"]:
        lbl = v["label"]
        A(f"| C5 {lbl} | " + " | ".join([
            _m(stats, "scenario_5_zkp_overhead", lbl, "avg_latency"),
            _m(stats, "scenario_5_zkp_overhead", lbl, "consensus_success", pct=True),
            "—", "—",
            _m(stats, "scenario_5_zkp_overhead", lbl, "consensus_time_ms", nd=1),
            _m(stats, "scenario_5_zkp_overhead", lbl, "ledger_integrity", pct=True),
            _m(stats, "scenario_5_zkp_overhead", lbl, "state_divergence_rate", pct=True),
        ]) + " |")
    A("")
    A("> C2 reporta L_core (pipeline sem geração de provas); os demais "
      "cenários reportam L_total.")
    A("")

    A("## Critérios (Seção 5.3)")
    A("")
    if criteria:
        A("| Cenário | Config | Critério | Valor | Limiar | Atende |")
        A("|---|---|---|---|---|---|")
        for r in criteria:
            ok = "✅" if r["atende"] else "❌"
            A(f"| {r['scenario']} | {r['config']} | "
              f"{r['criterio']} | {r['valor']:g} | {r['limiar']:g} | {ok} |")
    else:
        A("_Critérios não avaliados (configurações faltando)._")
    A("")

    A("## Cenário 1 — Nominal")
    A("")
    A(f"- Latência/rodada: {_m(stats, 'scenario_1_nominal', 'nominal', 'avg_latency')}s; "
      f"consenso: {_m(stats, 'scenario_1_nominal', 'nominal', 'consensus_time_ms', nd=1)}ms; "
      f"SC: {_m(stats, 'scenario_1_nominal', 'nominal', 'consensus_success', pct=True)}; "
      f"auditoria: {_m(stats, 'scenario_1_nominal', 'nominal', 'compliance', pct=True)}.")
    A("")

    A("## Cenário 2 — Escalabilidade (40–400 dispositivos)")
    A("")
    for c in SCENARIO_CONFIGS["scenario_2_scalability"]:
        lbl = c["label"]
        A(f"- {lbl}: L_core={_m(stats, 'scenario_2_scalability', lbl, 'core_latency')}s, "
          f"L_total={_m(stats, 'scenario_2_scalability', lbl, 'avg_latency')}s, "
          f"STARK gen={_m(stats, 'scenario_2_scalability', lbl, 'stage_time_ms.stark_gen', nd=0)}ms.")
    A("")

    A("## Cenário 3 — Adversarial")
    A("")
    for c in SCENARIO_CONFIGS["scenario_3_adversarial"]:
        lbl = c["label"]
        A(f"- {lbl}: DR={_m(stats, 'scenario_3_adversarial', lbl, 'detection_rate', pct=True)}, "
          f"FPR={_m(stats, 'scenario_3_adversarial', lbl, 'false_positive_rate', pct=True)}, "
          f"SC={_m(stats, 'scenario_3_adversarial', lbl, 'consensus_success', pct=True)}.")
    A("")

    A("## Cenário 4 — Inter-regional")
    A("")
    for c in SCENARIO_CONFIGS["scenario_4_interregional"]:
        lbl = c["label"]
        A(f"- {lbl}: latência={_m(stats, 'scenario_4_interregional', lbl, 'avg_latency')}s, "
          f"VRF={_m(stats, 'scenario_4_interregional', lbl, 'stage_time_ms.vrf_elect', nd=1)}ms, "
          f"STARK inter={_m(stats, 'scenario_4_interregional', lbl, 'stage_time_ms.stark_gen_inter', nd=1)}ms.")
    A("")

    A("## Cenário 5 — Overhead ZKP")
    A("")
    for v in SCENARIO_CONFIGS["scenario_5_zkp_overhead"]:
        lbl = v["label"]
        A(f"- {lbl}: latência={_m(stats, 'scenario_5_zkp_overhead', lbl, 'avg_latency')}s, "
          f"bloco={_m(stats, 'scenario_5_zkp_overhead', lbl, 'block_size_bytes', nd=0)}B, "
          f"proof gen={_m(stats, 'scenario_5_zkp_overhead', lbl, 'proof_gen_cpu_ms', nd=0)}ms.")
    A("")

    A("## Comparação com baselines")
    A("")
    if collectors:
        names = list(collectors)
        A("| Métrica | " + " | ".join(names) + " |")
        A("|---|" + "|".join("---" for _ in names) + "|")
        for label, key in COMPARISON_METRICS:
            vals = []
            for name in names:
                v = collectors[name].all_metrics().get(key, 0)
                if key in ("detection_rate", "false_positive_rate",
                           "consensus_success", "compliance",
                           "vrf_uniformity"):
                    vals.append(f"{v:.1%}")
                elif key in ("block_size_bytes", "comm_overhead_bytes"):
                    vals.append(f"{v:.0f}")
                elif key == "avg_latency":
                    vals.append(f"{v:.4f}")
                elif key == "consensus_time_ms":
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.2f}")
            A(f"| {label} | " + " | ".join(vals) + " |")
    else:
        A("_Dados de comparação indisponíveis._")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera tabelas de resultados a partir de results/<data>/")
    ap.add_argument("--dir", default="results/24-09-2026",
                    help="diretório com os JSONs de cenário")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.is_dir():
        raise SystemExit(f"diretório não encontrado: {results_dir}")
    out_dir = results_dir / "analise"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats, _aggs, warnings = load_configs(results_dir)
    for w in warnings:
        print(w, file=sys.stderr)

    write_csv(out_dir / "estatisticas.csv", stats_rows(stats), STATS_FIELDS)
    write_csv(out_dir / "metricas_por_config.csv", config_rows(stats),
              config_fields(stats))

    criteria, cw = build_criteria(stats)
    if cw:
        print(cw, file=sys.stderr)
    if criteria is not None:
        write_csv(out_dir / "criterios.csv", criteria, CRITERIA_FIELDS)

    (out_dir / "tabela_resultados.tex").write_text(
        build_resultados_tex(stats))
    (out_dir / "tabela_overhead.tex").write_text(build_overhead_tex(stats))

    tex, collectors, cmp_warnings = build_comparison(results_dir)
    for w in cmp_warnings:
        print(w, file=sys.stderr)
    if tex is not None:
        (out_dir / "tabela_comparacao.tex").write_text(tex)

    batch_rows = build_batch_reference()
    if batch_rows:
        write_csv(out_dir / "referencia_batch_antigo.csv", batch_rows,
                  ["scenario", "config", "metric", "mean", "std", "n"])
    else:
        print("[AVISO] batch antigo (results/runs/) sem dados",
              file=sys.stderr)

    md = build_markdown(stats, criteria, warnings, collectors, results_dir)
    (out_dir / "analise_resultados.md").write_text(md)

    print(f"-> Saídas em {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
