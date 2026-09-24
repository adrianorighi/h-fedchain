"""
Batch runner — N execuções independentes por configuração de cada cenário.

Coleta as métricas agregadas (all_metrics) e por-rodada (records) de cada
execução e produz estatísticas (média, std, min, max) para análise posterior.

Uso:
    python -m experiments.batch_runner --all --runs 5
    python -m experiments.batch_runner --scenario 1 --runs 5
    python -m experiments.batch_runner --all --quick
"""
import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import scenario_1_nominal
from experiments import scenario_2_scalability
from experiments import scenario_3_adversarial
from experiments import scenario_4_interregional
from experiments import scenario_5_zkp_overhead

RUNS_DIR = Path(__file__).parent.parent / "results" / "runs"

S2_SIZES = [10, 25, 50, 75, 100]
S3_RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40]
S4_CLUSTERS = [2, 4, 8, 12, 16]
S5_VARIANTS = ["no_zkp", "snark", "stark", "full"]


async def _run_s1(config, quick=False):
    return await scenario_1_nominal.run(tracer=None, quick=quick)


async def _run_s2(config, quick=False):
    return await scenario_2_scalability.run_scale(
        config["devices"], tracer=None, quick=quick
    )


async def _run_s3(config, quick=False):
    return await scenario_3_adversarial.run_adv(
        config["ratio"], tracer=None, quick=quick,
        rounds=None if quick else 15,
    )


async def _run_s4(config, quick=False):
    return await scenario_4_interregional.run_multi_cluster(
        config["clusters"], num_rounds=5, tracer=None, quick=quick
    )


async def _run_s5(config, quick=False):
    return await scenario_5_zkp_overhead.run_variant(
        config["variant"], tracer=None, quick=quick, num_rounds=10
    )


SCENARIOS = {
    "scenario_1_nominal": {
        "configs": [{"label": "nominal"}],
        "run": _run_s1,
    },
    "scenario_2_scalability": {
        "configs": [
            {"label": f"d{d}", "devices": d} for d in S2_SIZES
        ],
        "run": _run_s2,
    },
    "scenario_3_adversarial": {
        "configs": [
            {"label": f"rho_{int(r * 100)}", "ratio": r} for r in S3_RATIOS
        ],
        "run": _run_s3,
    },
    "scenario_4_interregional": {
        "configs": [
            {"label": f"c{c}", "clusters": c} for c in S4_CLUSTERS
        ],
        "run": _run_s4,
    },
    "scenario_5_zkp_overhead": {
        "configs": [
            {"label": v, "variant": v} for v in S5_VARIANTS
        ],
        "run": _run_s5,
    },
}

KEY_MAP = {
    "1": "scenario_1_nominal",
    "2": "scenario_2_scalability",
    "3": "scenario_3_adversarial",
    "4": "scenario_4_interregional",
    "5": "scenario_5_zkp_overhead",
    "all": list(SCENARIOS.keys()),
}


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _flatten(d: dict, prefix: str = ""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, p))
        else:
            out[p] = v
    return out


def _collect_paths(metrics_list: list[dict]):
    paths = set()
    for m in metrics_list:
        paths.update(_flatten(m).keys())
    return sorted(paths)


def compute_summary(metrics_list: list[dict]) -> dict:
    paths = _collect_paths(metrics_list)
    summary = {}
    for p in paths:
        vals = [_flatten(m).get(p) for m in metrics_list]
        if not all(isinstance(v, (int, float)) for v in vals):
            summary[p] = {"values": vals}
            continue
        summary[p] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "values": vals,
        }
    return summary


def _write_summary_csv(path: Path, summary: dict):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for p in sorted(summary.keys()):
            st = summary[p]
            writer.writerow([p, st.get("mean", ""), st.get("std", ""),
                             st.get("min", ""), st.get("max", "")])


def _write_overview_csv(path: Path, rows: list[dict]):
    cols = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _save_overview(all_rows: list[dict]):
    _save_json(RUNS_DIR / "overview.json", {"runs": all_rows})
    _write_overview_csv(RUNS_DIR / "overview.csv", all_rows)


async def batch_scenario(scenario_key: str, num_runs: int, quick: bool) -> list[dict]:
    spec = SCENARIOS[scenario_key]
    base_dir = RUNS_DIR / scenario_key
    all_rows: list[dict] = []
    for config in spec["configs"]:
        label = config["label"]
        cfg_dir = base_dir / label
        cfg_dir.mkdir(parents=True, exist_ok=True)
        metrics_list: list[dict] = []
        cfg_start = time.perf_counter()
        for i in range(num_runs):
            print(f"  [{scenario_key}] {label} — run {i + 1}/{num_runs}", flush=True)
            t0 = time.perf_counter()
            mc = await spec["run"](config, quick=quick)
            elapsed = time.perf_counter() - t0
            all_m = mc.all_metrics()
            metrics_list.append(all_m)
            _save_json(cfg_dir / f"run_{i}_metrics.json", all_m)
            _save_json(cfg_dir / f"run_{i}_rounds.json", mc._records)
            row = _flatten(all_m)
            row["scenario"] = scenario_key
            row["config"] = label
            row["run"] = i
            row["wall_clock_s"] = round(elapsed, 3)
            all_rows.append(row)
            print(f"      run {i + 1} ok — {elapsed:.1f}s", flush=True)
        summary = compute_summary(metrics_list)
        _save_json(cfg_dir / "summary.json", summary)
        _write_summary_csv(cfg_dir / "summary.csv", summary)
        cfg_total = time.perf_counter() - cfg_start
        print(f"    -> {cfg_dir}  ({cfg_total:.1f}s)", flush=True)
    return all_rows


async def main(num_runs: int, quick: bool, scenario_keys: list[str]):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    all_rows: list[dict] = []
    for key in scenario_keys:
        print(f"\n=== {key} ===", flush=True)
        rows = await batch_scenario(key, num_runs, quick)
        all_rows.extend(rows)
        _save_overview(all_rows)
    _save_overview(all_rows)
    total = time.perf_counter() - total_start
    print(f"\nTotal: {total:.1f}s — {len(all_rows)} execuções")
    print(f"Resultados em {RUNS_DIR}")
    from simulator.snark_worker import shutdown_pool
    shutdown_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N execuções por cenário")
    parser.add_argument("--scenario", default="all",
                        help="1|2|3|4|5|all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="Usa rounds reduzidos dos cenários")
    args = parser.parse_args()

    keys = KEY_MAP.get(args.scenario)
    if keys is None:
        parser.error(f"cenário inválido: {args.scenario} (use 1-5 ou 'all')")
    if isinstance(keys, str):
        keys = [keys]

    asyncio.run(main(args.runs, args.quick, keys))
