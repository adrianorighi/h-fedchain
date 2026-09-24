import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "csv")
OUT_CSV = os.path.join(OUT_DIR, "stage_times_all.csv")

STAGE_KEYS = [
    "verify",
    "multikrum",
    "stark_gen",
    "vrf_elect",
    "consensus_inter",
    "fedavg_weighted",
    "stark_gen_inter",
]

STAGE_PATTERN = re.compile(r"^scenario_(\d+)_(.*)\.json$")


def iter_scenario_rounds():
    seen = set()
    for name in sorted(os.listdir(RESULTS)):
        m = STAGE_PATTERN.match(name)
        if not m:
            continue
        if name in seen:
            continue
        seen.add(name)
        scenario = int(m.group(1))
        config = m.group(2)
        path = os.path.join(RESULTS, name)
        records = json.load(open(path))
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            continue
        if "stage_time_ms" not in records[0]:
            continue
        for r in records:
            yield scenario, config, r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = [
        "scenario",
        "config",
        "round",
        "variant",
        "latency",
        "consensus_time_ms",
        "block_size_bytes",
        "wall_clock_s",
    ] + ["stage_time_ms." + s for s in STAGE_KEYS]

    n_rows = 0
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, config, r in iter_scenario_rounds():
            row = {
                "scenario": scenario,
                "config": config,
                "round": r.get("round", ""),
                "variant": r.get("variant", ""),
                "latency": r.get("latency", ""),
                "consensus_time_ms": r.get("consensus_time_ms", ""),
                "block_size_bytes": r.get("block_size_bytes", ""),
                "wall_clock_s": r.get("wall_clock_s", ""),
            }
            stage = r.get("stage_time_ms") or {}
            for s in STAGE_KEYS:
                row["stage_time_ms." + s] = stage.get(s, "")
            writer.writerow(row)
            n_rows += 1

    print(f"Written {n_rows} rows to {OUT_CSV}")
    return n_rows


if __name__ == "__main__":
    sys.exit(main())