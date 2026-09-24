import time
import pytest
from monitoring.tracer import Tracer, Span
from monitoring.anomaly import AnomalyDetector, AlertRule


class TestTracer:
    def test_span_records_duration(self):
        tracer = Tracer()
        start = time.time()
        with tracer.span("op", "test"):
            pass
        elapsed = time.time() - start
        assert len(tracer._roots) == 1
        root = tracer._roots[0]
        assert root.name == "op"
        assert root.component == "test"
        assert root.duration_us > 0
        assert root.duration_us < elapsed * 1_100_000

    def test_span_metadata(self):
        tracer = Tracer()
        with tracer.span("op", "test", node_id="n0", round_num=5, extra="val"):
            pass
        root = tracer._roots[0]
        assert root.node_id == "n0"
        assert root.round_num == 5
        assert root.tags == {"extra": "val"}

    def test_nested_spans(self):
        tracer = Tracer()
        with tracer.span("parent", "test"):
            with tracer.span("child", "test"):
                pass
        root = tracer._roots[0]
        assert len(root.children) == 1
        assert root.children[0].name == "child"

    def test_multiple_roots(self):
        tracer = Tracer()
        with tracer.span("a", "test"):
            pass
        with tracer.span("b", "test"):
            pass
        assert len(tracer._roots) == 2

    def test_trace_summary(self):
        tracer = Tracer()
        with tracer.span("op", "test"):
            with tracer.span("sub", "test"):
                pass
        summary = tracer.trace_summary()
        assert "test.op" in summary
        assert "test.sub" in summary
        assert summary["test.op"]["count"] == 1

    def test_clear(self):
        tracer = Tracer()
        with tracer.span("op", "test"):
            pass
        assert len(tracer._roots) == 1
        tracer.clear()
        assert len(tracer._roots) == 0

    def test_to_json(self, tmp_path):
        tracer = Tracer()
        with tracer.span("op", "test"):
            pass
        path = tmp_path / "trace.json"
        output = tracer.to_json(str(path))
        assert path.exists()
        assert "op" in output

    @pytest.mark.asyncio
    async def test_tracer_works_with_async(self):
        tracer = Tracer()
        with tracer.span("async_op", "test"):
            await asyncio.sleep(0.01)
        root = tracer._roots[0]
        assert root.duration_us > 0
        assert root.duration_us > 5_000

    @pytest.mark.asyncio
    async def test_orchestrator_integration(self):
        from simulator.orchestrator import Orchestrator
        from simulator.fog_node import FogNode
        tracer = Tracer()
        orch = Orchestrator(num_clusters=1, nodes_per_cluster=2,
                            devices_per_cluster=3, f=0, latency_ms=2.0,
                            tracer=tracer)
        result = await orch.run_experiment(num_rounds=3, warmup=2)
        assert len(result.round_metrics) == 3
        assert len(tracer._roots) == 5
        for root in tracer._roots:
            assert root.name == "run_round"
            assert root.component == "orchestrator"

    @pytest.mark.asyncio
    async def test_fog_node_tracer_propagation(self):
        from simulator.orchestrator import Orchestrator
        tracer = Tracer()
        orch = Orchestrator(num_clusters=1, nodes_per_cluster=2,
                            devices_per_cluster=3, f=0, latency_ms=2.0,
                            tracer=tracer)
        orch.setup()
        for node in orch.nodes:
            assert node.tracer is tracer


import asyncio


class TestAnomalyDetector:
    def test_empty_detector_no_alerts(self):
        detector = AnomalyDetector()
        assert detector.alerts_summary()["total"] == 0

    def test_normal_metrics_no_alerts(self):
        detector = AnomalyDetector()
        for i in range(5):
            detector.observe({
                "round": i,
                "latency": 0.1,
                "qc_emitted": True,
                "view_change_count": 0,
                "state_divergence": False,
                "stage_time_ms": {"verify": 5.0},
            })
        summary = detector.alerts_summary()
        assert summary["total"] == 0

    def test_latency_spike_detected(self):
        detector = AnomalyDetector()
        for i in range(4):
            detector.observe({
                "round": i,
                "latency": 0.1,
                "qc_emitted": True,
                "view_change_count": 0,
                "state_divergence": False,
            })
        alerts = detector.observe({
            "round": 5,
            "latency": 0.5,
            "qc_emitted": True,
            "view_change_count": 0,
            "state_divergence": False,
        })
        lat_alerts = [a for a in alerts if a["rule"] == "LATENCY_SPIKE"]
        assert len(lat_alerts) == 1

    def test_dr_drop_detected(self):
        detector = AnomalyDetector()
        alerts = detector.observe({
            "round": 1,
            "latency": 0.1,
            "qc_emitted": True,
            "num_adversarial": 10,
            "rejected_adversarial": 2,
        })
        dr_alerts = [a for a in alerts if a["rule"] == "DR_DROP"]
        assert len(dr_alerts) == 1
        assert dr_alerts[0]["severity"] == "critical"

    def test_consensus_failure_detected(self):
        detector = AnomalyDetector()
        alerts = detector.observe({
            "round": 1,
            "latency": 0.1,
            "qc_emitted": False,
        })
        assert len([a for a in alerts if a["rule"] == "CONSENSUS_FAILURE"]) == 1

    def test_state_divergence_detected(self):
        detector = AnomalyDetector()
        alerts = detector.observe({
            "round": 1,
            "latency": 0.1,
            "state_divergence": True,
        })
        assert len([a for a in alerts if a["rule"] == "STATE_DIVERGENCE"]) == 1

    def test_vc_anomaly_detected(self):
        detector = AnomalyDetector()
        detector.observe({
            "round": 0,
            "latency": 0.1,
            "view_change_count": 0,
        })
        alerts = detector.observe({
            "round": 1,
            "latency": 0.1,
            "view_change_count": 3,
        })
        vc_alerts = [a for a in alerts if a["rule"] == "VC_ANOMALY"]
        assert len(vc_alerts) == 1

    def test_custom_rule(self):
        rule = AlertRule("CUSTOM", "info", lambda m, h: m.get("custom", False))
        detector = AnomalyDetector(rules=[rule])
        alerts = detector.observe({"round": 1, "custom": True})
        assert len(alerts) == 1
        assert alerts[0]["rule"] == "CUSTOM"

    def test_clear_resets(self):
        detector = AnomalyDetector()
        detector.observe({"round": 1, "latency": 0.5, "qc_emitted": False})
        assert detector.alerts_summary()["total"] > 0
        detector.clear()
        assert detector.alerts_summary()["total"] == 0
