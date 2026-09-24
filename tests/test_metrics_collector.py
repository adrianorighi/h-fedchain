import pytest
from experiments.metrics import MetricsCollector


def test_empty_metrics():
    mc = MetricsCollector()
    assert mc.avg_latency() == 0.0
    assert mc.consensus_success_rate() == 0.0
    assert mc.compliance_completeness() == 0.0


def test_single_round_metrics():
    mc = MetricsCollector()
    mc.add_round({
        "latency": 100.0,
        "qc_emitted": True,
        "num_adversarial": 5,
        "rejected_adversarial": 4,
        "num_honest": 15,
        "falsely_rejected": 1,
        "block_size_bytes": 1024,
        "comm_overhead_bytes": 2048,
        "consensus_time_ms": 50.0,
        "leader": "node_1",
    })
    assert mc.avg_latency() == 100.0
    assert mc.consensus_success_rate() == 1.0
    assert mc.detection_rate() == 0.8
    assert abs(mc.false_positive_rate() - 1.0 / 15) < 1e-9
    assert mc.block_size_bytes() == 1024
    assert mc.comm_overhead_bytes() == 2048
    assert mc.consensus_time_ms() == 50.0


def test_multiple_round_average():
    mc = MetricsCollector()
    for i in range(5):
        mc.add_round({
            "latency": float(100 + i * 10),
            "qc_emitted": i % 2 == 0,
            "consensus_time_ms": 50.0,
        })
    assert mc.avg_latency() == 120.0
    assert mc.consensus_success_rate() == 0.6
    assert mc.compliance_completeness() == 0.6


def test_detection_rate_no_adversarial():
    mc = MetricsCollector()
    mc.add_round({"latency": 10.0})
    assert mc.detection_rate() == 1.0


def test_vrf_election_uniformity():
    mc = MetricsCollector()
    for i in range(10):
        mc.add_round({"latency": 1.0, "leader": f"node_{i % 3}"})
    uni = mc.vrf_election_uniformity()
    assert 0.0 <= uni <= 1.0


def test_all_metrics():
    mc = MetricsCollector()
    mc.add_round({
        "latency": 50.0,
        "qc_emitted": True,
        "num_adversarial": 2,
        "rejected_adversarial": 2,
        "num_honest": 18,
        "falsely_rejected": 0,
        "block_size_bytes": 512,
        "comm_overhead_bytes": 1024,
        "consensus_time_ms": 25.0,
        "leader": "node_a",
    })
    metrics = mc.all_metrics()
    assert "avg_latency" in metrics
    assert "consensus_success" in metrics
    assert "detection_rate" in metrics
    assert metrics["avg_latency"] == 50.0
    assert metrics["detection_rate"] == 1.0


def test_to_json(tmp_path):
    mc = MetricsCollector()
    mc.add_round({"latency": 10.0, "qc_emitted": True})
    path = tmp_path / "metrics.json"
    mc.to_json(str(path))
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["latency"] == 10.0


# ── Testes para métricas novas (Grupo A/B) ────────────

def test_view_change_frequency_zero():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "view_change_count": 0})
    assert mc.view_change_frequency() == 0.0
    assert mc.view_change_resistance() == 1.0


def test_view_change_frequency_nonzero():
    mc = MetricsCollector()
    for i in range(5):
        mc.add_round({"latency": 1.0, "view_change_count": i})
    assert mc.view_change_frequency() == 2.0  # (0+1+2+3+4)/5


def test_view_change_latency():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "view_change_latency_ms": 150.0})
    mc.add_round({"latency": 1.0, "view_change_latency_ms": 50.0})
    assert mc.view_change_latency_ms() == 100.0


def test_ledger_integrity_all_ok():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "ledger_integrity": True})
    mc.add_round({"latency": 1.0, "ledger_integrity": True})
    assert mc.ledger_integrity() == 1.0


def test_ledger_integrity_mixed():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "ledger_integrity": True})
    mc.add_round({"latency": 1.0, "ledger_integrity": False})
    assert mc.ledger_integrity() == 0.5


def test_state_divergence_none():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "state_divergence": False})
    assert mc.state_divergence_rate() == 0.0


def test_state_divergence_some():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "state_divergence": False})
    mc.add_round({"latency": 1.0, "state_divergence": True})
    assert mc.state_divergence_rate() == 0.5


def test_participation_traceability():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "num_accepted": 10, "num_rejected": 2})
    assert mc.participation_traceability() == 1.0


def test_stage_time_empty():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0})
    assert mc.stage_time_ms() == {}


def test_stage_time_single():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "stage_time_ms": {"verify": 10.0, "multikrum": 2.0}})
    result = mc.stage_time_ms()
    assert result.get("verify") == 10.0
    assert result.get("multikrum") == 2.0
    assert mc.stage_time_ms("verify") == 10.0


def test_stage_time_average():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "stage_time_ms": {"verify": 10.0}})
    mc.add_round({"latency": 1.0, "stage_time_ms": {"verify": 20.0}})
    assert mc.stage_time_ms("verify") == 15.0


def test_compliance_overhead_no_baseline():
    mc = MetricsCollector()
    mc.add_round({"latency": 2.0})
    result = mc.compliance_overhead()
    assert result["overhead_pct"] == 0.0
    assert result["actual"] == 2.0


def test_compliance_overhead_with_baseline():
    mc = MetricsCollector()
    mc.add_round({"latency": 3.0})
    result = mc.compliance_overhead(baseline_latency=2.0)
    assert result["overhead_pct"] == 50.0
    assert result["baseline"] == 2.0


def test_convergence_rounds():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0, "loss": 2.0})
    mc.add_round({"latency": 1.0, "loss": 1.5})
    mc.add_round({"latency": 1.0, "loss": 1.0})
    mc.add_round({"latency": 1.0, "loss": 0.99})
    assert mc.convergence_rounds(threshold=0.02) == 3


def test_convergence_rounds_no_loss():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0})
    assert mc.convergence_rounds() == 1


def test_all_metrics_includes_new():
    mc = MetricsCollector()
    mc.add_round({
        "latency": 50.0, "qc_emitted": True,
        "view_change_count": 0, "view_change_latency_ms": 0,
        "ledger_integrity": True, "state_divergence": False,
        "stage_time_ms": {}, "num_accepted": 10, "num_rejected": 0,
    })
    metrics = mc.all_metrics()
    new_keys = [
        "participation_traceability", "view_change_frequency",
        "view_change_latency_ms", "view_change_resistance",
        "ledger_integrity", "state_divergence_rate", "stage_time_ms",
        "cpu_percent", "memory_rss_bytes", "network_bytes",
        "mb_edge_fog", "mb_fog_inter", "mb_fog_cloud",
        "proof_gen_cpu_ms", "proof_verify_cpu_ms", "proof_size_bytes",
    ]
    for key in new_keys:
        assert key in metrics, f"Missing key: {key}"


def test_new_metric_getters():
    mc = MetricsCollector()
    mc.add_round({
        "round": 1, "latency": 1.0, "qc_emitted": True,
        "num_adversarial": 0, "rejected_adversarial": 0,
        "num_honest": 5, "falsely_rejected": 0,
        "cpu_percent": 12.5, "memory_rss_bytes": 1024,
        "network_bytes": 5000,
        "bytes_edge_fog": 2_000_000,
        "bytes_fog_inter": 1_000_000,
        "bytes_fog_cloud": 500_000,
        "proof_gen_cpu_ms": 10.0, "proof_verify_cpu_ms": 1.0,
        "proof_size_bytes": 921,
    })
    assert mc.cpu_percent() == 12.5
    assert mc.memory_rss_bytes() == 1024
    assert mc.network_bytes() == 5000
    assert mc.mb_edge_fog() == 2.0
    assert mc.mb_fog_inter() == 1.0
    assert mc.mb_fog_cloud() == 0.5
    assert mc.proof_gen_cpu_ms() == 10.0
    assert mc.proof_verify_cpu_ms() == 1.0
    assert mc.proof_size_bytes() == 921
    am = mc.all_metrics()
    for k in ("cpu_percent", "memory_rss_bytes", "network_bytes",
              "mb_edge_fog", "mb_fog_inter", "mb_fog_cloud",
              "proof_gen_cpu_ms", "proof_verify_cpu_ms", "proof_size_bytes"):
        assert k in am


def test_new_metric_getters_missing_keys():
    mc = MetricsCollector()
    mc.add_round({"latency": 1.0})
    assert mc.cpu_percent() == 0.0
    assert mc.memory_rss_bytes() == 0.0
    assert mc.network_bytes() == 0.0
    assert mc.mb_edge_fog() == 0.0
    assert mc.mb_fog_inter() == 0.0
    assert mc.mb_fog_cloud() == 0.0
    assert mc.proof_gen_cpu_ms() == 0.0
    assert mc.proof_verify_cpu_ms() == 0.0
    assert mc.proof_size_bytes() == 0.0


def test_vrf_election_uniformity_single_leader():
    mc = MetricsCollector()
    for _ in range(10):
        mc.add_round({"latency": 1.0, "leader": "n1"})
    uni = mc.vrf_election_uniformity()
    assert uni == 0.0 and uni == uni  # finite, no NaN
