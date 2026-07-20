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
