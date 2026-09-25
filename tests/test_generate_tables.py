import json
import re
from pathlib import Path

import pytest

import experiments.generate_tables as gt
from experiments.analyze_results import SCENARIO_CONFIGS


def make_records(n=4):
    recs = []
    for i in range(n):
        recs.append({
            "round": i, "latency": 1.0 + i, "consensus_time_ms": 100.0 + 10 * i,
            "block_size_bytes": 1000, "comm_overhead_bytes": 500,
            "qc_emitted": True, "leader": "n1",
            "cpu_percent": 20.0 + i, "memory_rss_bytes": 1_000_000.0,
            "network_bytes": 100, "bytes_edge_fog": 40, "bytes_fog_intra": 30,
            "bytes_fog_inter": 20, "bytes_fog_cloud": 10,
            "proof_gen_cpu_ms": 5.0, "proof_verify_cpu_ms": 0.0,
            "proof_size_bytes": 128, "view_change_count": 0,
            "view_change_latency_ms": 0.0, "state_divergence": False,
            "ledger_integrity": True, "num_adversarial": 1, "num_honest": 4,
            "rejected_adversarial": 1, "falsely_rejected": 0,
            "stage_time_ms": {"verify": 1.0, "multikrum": 2.0, "stark_gen": 0.5},
            "snark_proofs_total": 0, "snark_verify_projected_ms": 0.0,
            "variant": "no_zkp",
        })
    return recs


def test_expected_file_covers_all_20_configs():
    files = {gt.expected_file(s, c["label"])
             for s, cfgs in SCENARIO_CONFIGS.items() for c in cfgs}
    assert len(files) == 20
    assert "scenario_2_100.json" in files
    assert "scenario_3_adv_40.json" in files
    assert "scenario_4_16_clusters.json" in files
    assert "scenario_5_full.json" in files


def test_config_stats_mean_std_over_rounds():
    am, st = gt.config_stats(make_records())
    assert st["avg_latency"]["n"] == 4
    assert st["avg_latency"]["mean"] == pytest.approx(2.5)
    assert st["avg_latency"]["std"] > 0
    assert st["cpu_percent"]["mean"] == pytest.approx(21.5)
    assert st["stage_time_ms.stark_gen"]["mean"] == pytest.approx(0.5)


def test_config_stats_aggregated_metric_single_value():
    am, st = gt.config_stats(make_records())
    assert st["detection_rate"]["mean"] == 1.0
    assert st["detection_rate"]["std"] == 0.0
    assert st["detection_rate"]["n"] == 1
    assert st["false_positive_rate"]["mean"] == 0.0
    assert st["consensus_success"]["mean"] == 1.0


def test_config_stats_core_latency_series_and_aggregate():
    am, st = gt.config_stats(make_records())
    expected = 2.5 - 0.5 / 1000.0
    assert am["core_latency"] == pytest.approx(expected)
    assert st["core_latency"]["mean"] == pytest.approx(expected)
    assert st["core_latency"]["n"] == 4


def test_config_stats_bytes_per_link_in_mb():
    am, st = gt.config_stats(make_records())
    assert st["mb_edge_fog"]["mean"] == pytest.approx(40 / 1e6)
    assert st["mb_fog_inter"]["mean"] == pytest.approx(20 / 1e6)
    assert st["mb_fog_cloud"]["mean"] == pytest.approx(10 / 1e6)


def test_byte_invariant_ok():
    assert gt.check_byte_invariant("x.json", make_records()) == []


def test_byte_invariant_warns_with_round():
    recs = make_records(2)
    recs[1]["network_bytes"] = 999
    w = gt.check_byte_invariant("x.json", recs)
    assert len(w) == 1
    assert "round 1" in w[0] and "999" in w[0]


def test_byte_invariant_none_and_missing_keys_no_crash():
    recs = make_records(2)
    recs[0]["bytes_fog_cloud"] = None
    del recs[1]["network_bytes"]
    w = gt.check_byte_invariant("x.json", recs)
    assert len(w) == 2
    assert "network_bytes=100" in w[0] and "soma dos elos=90" in w[0]
    assert "network_bytes=0" in w[1] and "soma dos elos=100" in w[1]


def test_load_configs_partial_dir_warns_and_continues(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text(json.dumps(make_records()))
    stats, aggs, warnings = gt.load_configs(tmp_path)
    assert "nominal" in stats["scenario_1_nominal"]
    assert "nominal" in aggs["scenario_1_nominal"]
    missing = [w for w in warnings if "ausente" in w]
    assert len(missing) == 19


def test_load_configs_propagates_invariant_warning(tmp_path):
    recs = make_records()
    recs[2]["network_bytes"] = 999
    (tmp_path / "scenario_1_nominal.json").write_text(json.dumps(recs))
    stats, _, warnings = gt.load_configs(tmp_path)
    assert "nominal" in stats["scenario_1_nominal"]
    missing = [w for w in warnings if "ausente" in w]
    invariant = [w for w in warnings if "network_bytes" in w]
    assert len(missing) == 19
    assert len(invariant) == 1
    assert "round 2" in invariant[0] and "999" in invariant[0]


def test_load_configs_malformed_json_exits(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text("{nope")
    with pytest.raises(SystemExit) as e:
        gt.load_configs(tmp_path)
    assert "scenario_1_nominal.json" in str(e.value)


def test_load_configs_non_list_json_exits(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text('{"round": 1}')
    with pytest.raises(SystemExit) as e:
        gt.load_configs(tmp_path)
    assert "scenario_1_nominal.json" in str(e.value)


def test_load_configs_empty_file_warns(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text("[]")
    stats, _, warnings = gt.load_configs(tmp_path)
    assert "nominal" not in stats["scenario_1_nominal"]
    assert any("sem rounds" in w for w in warnings)


def _write_all_configs(tmp_path, records=None):
    records = records or make_records()
    for scenario, configs in SCENARIO_CONFIGS.items():
        for c in configs:
            (tmp_path / gt.expected_file(scenario, c["label"])).write_text(
                json.dumps(records))


def test_build_criteria_full(tmp_path):
    _write_all_configs(tmp_path)
    stats, _, _ = gt.load_configs(tmp_path)
    criteria, warn = gt.build_criteria(stats)
    assert warn is None
    assert criteria, "lista de critérios vazia"
    for row in criteria:
        assert isinstance(row["atende"], bool)
        assert row["criterio"] and row["limiar"] is not None
    dr = [r for r in criteria if r["criterio"] == "DR >= 95%"]
    assert dr and all(r["atende"] for r in dr)


def test_build_criteria_missing_config_skips(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text(json.dumps(make_records()))
    stats, _, _ = gt.load_configs(tmp_path)
    criteria, warn = gt.build_criteria(stats)
    assert criteria is None
    assert "faltando" in warn


def test_csv_outputs(tmp_path):
    _write_all_configs(tmp_path)
    stats, aggs, _ = gt.load_configs(tmp_path)
    out = tmp_path / "analise"
    out.mkdir()
    gt.write_csv(out / "estatisticas.csv", gt.stats_rows(stats),
                 gt.STATS_FIELDS)
    gt.write_csv(out / "metricas_por_config.csv", gt.config_rows(stats),
                 gt.config_fields(stats))
    criteria, _ = gt.build_criteria(stats)
    gt.write_csv(out / "criterios.csv", criteria, gt.CRITERIA_FIELDS)
    est = (out / "estatisticas.csv").read_text().splitlines()
    assert est[0] == "scenario,config,metric,mean,std,ci95,cv,min,max,n"
    assert len(est) == 1 + sum(len(m) for cfgs in stats.values() for m in cfgs.values())
    cfg = (out / "metricas_por_config.csv").read_text().splitlines()
    assert cfg[0].startswith("scenario,config,")
    assert "proof_size_bytes" in cfg[0] and "cpu_percent" in cfg[0]
    assert len(cfg) == 21  # header + 20 configs
    crit = (out / "criterios.csv").read_text().splitlines()
    assert crit[0] == "scenario,config,criterio,valor,limiar,atende,tipo"
    assert len(crit) > 10


def test_config_rows_derive_from_stats_not_aggregate():
    recs = make_records(3)
    for r in recs:
        r["stage_time_ms"]["stark_gen"] = 0.5
    recs[2]["stage_time_ms"]["stark_gen"] = 0.0
    am, st = gt.config_stats(recs)
    assert am["stage_time_ms"]["stark_gen"] == pytest.approx(0.5)
    rows = gt.config_rows({"s": {"c": st}})
    assert rows[0]["stage_time_ms.stark_gen"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_resultados_tex_contains_all_rows(tmp_path):
    _write_all_configs(tmp_path)
    stats, _, _ = gt.load_configs(tmp_path)
    tex = gt.build_resultados_tex(stats)
    assert r"\begin{table}" in tex
    assert "tab:resultados" in tex
    assert "1 Nominal" in tex
    assert "2 d100" in tex
    assert r"3 $\rho$=40\%" in tex
    assert "4 c16" in tex
    assert "5 full" in tex
    assert "$\\pm$" in tex


def test_resultados_tex_skips_missing_config(tmp_path):
    (tmp_path / "scenario_1_nominal.json").write_text(json.dumps(make_records()))
    stats, _, _ = gt.load_configs(tmp_path)
    tex = gt.build_resultados_tex(stats)
    assert "1 Nominal" in tex
    assert "2 d10" not in tex


def test_overhead_tex_new_metrics(tmp_path):
    _write_all_configs(tmp_path)
    stats, _, _ = gt.load_configs(tmp_path)
    tex = gt.build_overhead_tex(stats)
    assert r"\begin{table}" in tex
    assert "tab:overhead" in tex
    for hdr in ("CPU (\\%)", "Mem. (MB)", "Rede (MB)", "Proof gen (s)",
                "Proof verif. (ms)", "Proof tam. (KB)",
                "Fog$\\leftrightarrow$inter (MB)"):
        assert hdr in tex
    assert "1 Nominal" in tex and "5 stark" in tex


def test_resultados_tex_escapes_percent_and_columns(tmp_path):
    _write_all_configs(tmp_path)
    stats, _, _ = gt.load_configs(tmp_path)
    tex = gt.build_resultados_tex(stats)
    lines = tex.splitlines()
    data_rows = [ln for ln in lines if ln.endswith(r" \\")]
    assert len(data_rows) == 20
    header = next(ln for ln in lines if ln.endswith(r" \\ \midrule"))
    ncols = header.count("&") + 1
    for ln in data_rows:
        assert ln.count("&") == ncols - 1
        assert re.search(r"(?<!\\)%", ln) is None, f"unescaped % em: {ln}"


def test_build_comparison_full(tmp_path):
    for slug in ("h_fedchain", "fedsdm", "flcoin"):
        (tmp_path / f"comparison_nominal_{slug}.json").write_text(
            json.dumps(make_records(3)))
    tex, collectors, warnings = gt.build_comparison(tmp_path)
    assert r"\begin{table}" in tex
    assert "tab:comparison_nominal" in tex
    assert set(collectors) == {"H-FedChain", "FedSDM", "FLCoin"}
    assert warnings == []
    assert "nan" not in tex.lower()


def test_build_comparison_missing_all(tmp_path):
    tex, collectors, warnings = gt.build_comparison(tmp_path)
    assert tex is None and collectors == {}
    assert any("comparação ausente" in w for w in warnings)


def test_build_comparison_escapes_percent(tmp_path):
    for slug in ("h_fedchain", "fedsdm", "flcoin"):
        (tmp_path / f"comparison_nominal_{slug}.json").write_text(
            json.dumps(make_records(3)))
    tex, _, _ = gt.build_comparison(tmp_path)
    assert re.search(r"(?<!\\)%", tex) is None
    assert r"100.0\%" in tex


def _make_batch(tmp_path):
    import experiments.analyze_results as ar
    for lbl in ("rho_0", "rho_5", "rho_20"):
        d = tmp_path / "scenario_3_adversarial" / lbl
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"run_{i}_metrics.json").write_text(json.dumps({
                "avg_latency": 1.0 + i, "consensus_time_ms": 100.0,
                "stage_time_ms": {"stark_gen": 10.0},
                "detection_rate": 1.0}))
            (d / f"run_{i}_rounds.json").write_text("[]")
    return ar


def test_batch_reference_excludes_rho5(tmp_path, monkeypatch):
    ar = _make_batch(tmp_path)
    monkeypatch.setattr(ar, "RUNS_DIR", tmp_path)
    rows = gt.build_batch_reference()
    labels = {r["config"] for r in rows}
    assert "rho_5" not in labels
    assert {"rho_0", "rho_20"} <= labels
    assert all(r["n"] == 2 for r in rows)
    assert any(r["metric"] == "core_latency" for r in rows)


def test_batch_reference_empty_runs_dir(tmp_path, monkeypatch):
    import experiments.analyze_results as ar
    monkeypatch.setattr(ar, "RUNS_DIR", tmp_path / "nope")
    assert gt.build_batch_reference() == []


def test_build_comparison_invalid_json(tmp_path):
    (tmp_path / "comparison_nominal_h_fedchain.json").write_text("{nope")
    with pytest.raises(SystemExit):
        gt.build_comparison(tmp_path)


def test_build_markdown_sections(tmp_path):
    _write_all_configs(tmp_path)
    for slug in ("h_fedchain", "fedsdm", "flcoin"):
        (tmp_path / f"comparison_nominal_{slug}.json").write_text(
            json.dumps(make_records(3)))
    stats, aggs, warnings = gt.load_configs(tmp_path)
    criteria, _ = gt.build_criteria(stats)
    _, collectors, _ = gt.build_comparison(tmp_path)
    md = gt.build_markdown(stats, criteria, warnings, collectors, tmp_path)
    assert str(tmp_path) in md
    for section in ("# Análise de Resultados", "## Base dos dados",
                    "## Resumo executivo", "## Critérios",
                    "## Cenário 1", "## Cenário 2", "## Cenário 3",
                    "## Cenário 4", "## Cenário 5",
                    "## Comparação com baselines"):
        assert section in md, section
    assert "| C1 nominal |" in md
    assert "| C3 rho_40 |" in md


def test_build_markdown_fallbacks():
    md = gt.build_markdown({}, None, [], {}, Path("."))
    assert "_Critérios não avaliados (configurações faltando)._" in md
    assert "_Dados de comparação indisponíveis._" in md


def test_main_end_to_end(tmp_path, monkeypatch):
    _write_all_configs(tmp_path)
    for slug in ("h_fedchain", "fedsdm", "flcoin"):
        (tmp_path / f"comparison_nominal_{slug}.json").write_text(
            json.dumps(make_records(3)))
    batch = tmp_path / "runs"
    _make_batch(batch)
    import experiments.analyze_results as ar
    monkeypatch.setattr(ar, "RUNS_DIR", batch)
    monkeypatch.setattr("sys.argv",
                        ["generate_tables", "--dir", str(tmp_path)])
    rc = gt.main()
    assert rc == 0
    out = tmp_path / "analise"
    for name in ("estatisticas.csv", "criterios.csv",
                 "metricas_por_config.csv", "tabela_resultados.tex",
                 "tabela_overhead.tex", "tabela_comparacao.tex",
                 "referencia_batch_antigo.csv", "analise_resultados.md"):
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 0
    ref = (out / "referencia_batch_antigo.csv").read_text()
    assert "rho_5" not in ref
    md_text = (out / "analise_resultados.md").read_text()
    assert tmp_path.name in md_text


def test_main_removes_stale_conditional_outputs(tmp_path, monkeypatch):
    (tmp_path / "scenario_1_nominal.json").write_text(json.dumps(make_records()))
    out = tmp_path / "analise"
    out.mkdir()
    for name in ("criterios.csv", "tabela_comparacao.tex",
                 "referencia_batch_antigo.csv"):
        (out / name).write_text("stale content")
    batch = tmp_path / "runs"
    _make_batch(batch)
    import experiments.analyze_results as ar
    monkeypatch.setattr(ar, "RUNS_DIR", batch)
    monkeypatch.setattr("sys.argv",
                        ["generate_tables", "--dir", str(tmp_path)])
    assert gt.main() == 0
    assert not (out / "criterios.csv").exists()      # criteria skipped (configs faltando)
    assert not (out / "tabela_comparacao.tex").exists()  # comparison absent
    # batch reference still written (RUNS_DIR patched to fixture):
    assert "stale" not in (out / "referencia_batch_antigo.csv").read_text()
    md = (out / "analise_resultados.md").read_text()
    assert "## Avisos" in md
    assert "comparação ausente" in md
    assert "critérios pulados" in md
