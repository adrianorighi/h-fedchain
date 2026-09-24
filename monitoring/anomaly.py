import statistics
from typing import Optional


class AlertRule:
    def __init__(self, name: str, severity: str, condition_fn, message_template: str = ""):
        self.name = name
        self.severity = severity
        self.condition_fn = condition_fn
        self.message_template = message_template

    def evaluate(self, metrics: dict, history: list[dict]) -> Optional[dict]:
        result = self.condition_fn(metrics, history)
        if result:
            msg = self.message_template.format(**metrics) if self.message_template else self.name
            return {"rule": self.name, "severity": self.severity, "message": msg}
        return None


def _moving_avg(key: str, history: list[dict], window: int = 5) -> float:
    vals = [r[key] for r in history[-window:] if key in r]
    return statistics.mean(vals) if vals else 0.0


def _check_latency_spike(metrics: dict, history: list[dict]) -> bool:
    if "latency" not in metrics or len(history) < 3:
        return False
    avg = _moving_avg("latency", history[:-1], 5)
    return avg > 0 and metrics["latency"] > avg * 2


def _check_dr_drop(metrics: dict, history: list[dict]) -> bool:
    dr_num = metrics.get("rejected_adversarial", 0)
    dr_den = metrics.get("num_adversarial", 0)
    if dr_den == 0:
        return False
    dr = dr_num / dr_den
    return dr < 0.5


def _check_consensus_failure(metrics: dict, history: list[dict]) -> bool:
    return not metrics.get("qc_emitted", True)


def _check_vc_anomaly(metrics: dict, history: list[dict]) -> bool:
    baseline = _moving_avg("view_change_count", history[:-1], 5)
    current = metrics.get("view_change_count", 0)
    return current > max(1, baseline * 2)


def _check_state_divergence(metrics: dict, history: list[dict]) -> bool:
    return metrics.get("state_divergence", False)


def _check_stage_timeout(metrics: dict, history: list[dict]) -> bool:
    st = metrics.get("stage_time_ms", {})
    for stage, val in st.items():
        all_vals = [r.get("stage_time_ms", {}).get(stage, 0) for r in history if "stage_time_ms" in r]
        if len(all_vals) >= 3 and val > 0:
            mean = statistics.mean(all_vals[:-1])
            stdev = statistics.stdev(all_vals[:-1]) if len(all_vals) > 2 else 0
            if stdev > 0 and val > mean + 3 * stdev:
                return True
    return False


ANOMALY_RULES = [
    AlertRule("LATENCY_SPIKE", "warn", _check_latency_spike,
              "Latency {latency:.3f}s exceeds 2x moving average"),
    AlertRule("DR_DROP", "critical", _check_dr_drop,
              "Detection rate dropped below 0.5"),
    AlertRule("CONSENSUS_FAILURE", "critical", _check_consensus_failure,
              "Round {round} failed to reach consensus"),
    AlertRule("VC_ANOMALY", "warn", _check_vc_anomaly,
              "View change count {view_change_count} above baseline"),
    AlertRule("STATE_DIVERGENCE", "critical", _check_state_divergence,
              "State divergence detected at round {round}"),
    AlertRule("STAGE_TIMEOUT", "warn", _check_stage_timeout,
              "Stage time outlier detected"),
]


class AnomalyDetector:
    def __init__(self, rules: Optional[list[AlertRule]] = None):
        self.rules = rules or ANOMALY_RULES
        self._history: list[dict] = []
        self._alerts: list[dict] = []

    def observe(self, metrics: dict) -> list[dict]:
        self._history.append(metrics)
        alerts = []
        for rule in self.rules:
            alert = rule.evaluate(metrics, self._history)
            if alert:
                alert["round"] = metrics.get("round", 0)
                self._alerts.append(alert)
                alerts.append(alert)
        return alerts

    def alerts_summary(self) -> dict:
        by_severity = {}
        for a in self._alerts:
            sev = a["severity"]
            by_severity.setdefault(sev, []).append(a)
        return {
            "total": len(self._alerts),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "details": self._alerts,
        }

    def clear(self):
        self._history.clear()
        self._alerts.clear()
