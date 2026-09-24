import json

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
