import json
import statistics


class MetricsCollector:
    def __init__(self):
        self._records: list[dict] = []

    def add_round(self, metrics: dict):
        self._records.append(metrics)

    def latency_per_round(self) -> list[float]:
        return [r["latency"] for r in self._records]

    def avg_latency(self) -> float:
        vals = self.latency_per_round()
        return statistics.mean(vals) if vals else 0.0

    def detection_rate(self) -> float:
        total_adv = sum(r.get("num_adversarial", 0) for r in self._records)
        rejected_adv = sum(r.get("rejected_adversarial", 0) for r in self._records)
        if total_adv == 0:
            return 1.0
        return rejected_adv / total_adv

    def false_positive_rate(self) -> float:
        total_honest = sum(r.get("num_honest", 0) for r in self._records)
        falsely_rejected = sum(r.get("falsely_rejected", 0) for r in self._records)
        if total_honest == 0:
            return 0.0
        return falsely_rejected / total_honest

    def consensus_success_rate(self) -> float:
        if not self._records:
            return 0.0
        success = sum(1 for r in self._records if r.get("qc_emitted", True))
        return success / len(self._records)

    def throughput(self, total_time_seconds: float) -> float:
        if total_time_seconds <= 0:
            return 0.0
        return len(self._records) / (total_time_seconds / 60.0)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
