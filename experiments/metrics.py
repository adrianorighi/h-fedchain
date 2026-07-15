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

    def block_size_bytes(self) -> float:
        vals = [r.get("block_size_bytes", 0) for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def comm_overhead_bytes(self) -> float:
        vals = [r.get("comm_overhead_bytes", 0) for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def consensus_time_ms(self) -> float:
        vals = [r.get("consensus_time_ms", 0) for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def vrf_election_uniformity(self) -> float:
        from collections import Counter
        leaders = [r.get("leader", "") for r in self._records if r.get("leader")]
        if len(leaders) < 2:
            return 1.0
        counts = Counter(leaders)
        expected = len(leaders) / len(counts)
        chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
        return max(0.0, 1.0 - chi2 / (len(leaders) * 2))

    def compliance_completeness(self) -> float:
        if not self._records:
            return 0.0
        complete = sum(1 for r in self._records if r.get("qc_emitted", False))
        return complete / len(self._records)

    def all_metrics(self) -> dict:
        return {
            "avg_latency": self.avg_latency(),
            "consensus_success": self.consensus_success_rate(),
            "detection_rate": self.detection_rate(),
            "false_positive_rate": self.false_positive_rate(),
            "throughput": self.throughput(total_time_seconds=sum(
                r.get("latency", 0) for r in self._records
            )),
            "block_size_bytes": self.block_size_bytes(),
            "consensus_time_ms": self.consensus_time_ms(),
            "vrf_uniformity": self.vrf_election_uniformity(),
            "compliance": self.compliance_completeness(),
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
