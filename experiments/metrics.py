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
        from scipy.stats import chisquare
        leaders = [r.get("leader", "") for r in self._records if r.get("leader")]
        if len(leaders) < 2:
            return 1.0
        counts = Counter(leaders)
        observed = [counts[k] for k in sorted(counts.keys())]
        _, p_value = chisquare(observed)
        return p_value

    def vrf_diversity(self) -> float:
        """Fraction of unique leaders observed. Stable metric vs p-value."""
        leaders = [r.get("leader", "") for r in self._records if r.get("leader")]
        if len(leaders) < 2:
            return 1.0
        return len(set(leaders)) / len(leaders)

    def compliance_completeness(self) -> float:
        if not self._records:
            return 0.0
        complete = 0
        for r in self._records:
            if not r.get("qc_emitted", False):
                continue
            variant = r.get("variant", "no_zkp")
            if variant in ("snark", "full"):
                attempted = r.get("snark_attempted", 0)
                passed = r.get("snark_passed", 0)
                if attempted > 0 and passed == 0:
                    continue
            if variant in ("stark", "full") and not r.get("stark_proof_generated", False):
                continue
            complete += 1
        return complete / len(self._records)

    # ── Grupo A: dados já existem ──────────────────────

    def participation_traceability(self) -> float:
        """RP — Rastreabilidade de participação."""
        if not self._records:
            return 0.0
        traceable = sum(
            1 for r in self._records
            if r.get("num_accepted", 0) >= 0
        )
        return traceable / len(self._records)

    def view_change_resistance(self) -> float:
        """R_vc — Resistência a view-changes (1 - F_vc)."""
        return max(0.0, 1.0 - self.view_change_frequency())

    # ── Grupo B: novos campos do _build_metrics ────────

    def view_change_frequency(self) -> float:
        """F_vc — View-changes por rodada."""
        if not self._records:
            return 0.0
        total = sum(r.get("view_change_count", 0) for r in self._records)
        return total / len(self._records)

    def view_change_latency_ms(self) -> float:
        """ΔT_vc — Duração média de view-change (ms)."""
        vals = [r["view_change_latency_ms"] for r in self._records
                if r.get("view_change_latency_ms", 0) > 0]
        return statistics.mean(vals) if vals else 0.0

    def ledger_integrity(self) -> float:
        """I_ledger — Fração de rodadas com ledger íntegro."""
        if not self._records:
            return 1.0
        intact = sum(1 for r in self._records if r.get("ledger_integrity", True))
        return intact / len(self._records)

    def state_divergence_rate(self) -> float:
        """D_state — Fração de rodadas com divergência de estado."""
        if not self._records:
            return 0.0
        div = sum(1 for r in self._records if r.get("state_divergence", False))
        return div / len(self._records)

    def _round_mean(self, key: str) -> float:
        vals = [r.get(key, 0) for r in self._records
                if isinstance(r.get(key, 0), (int, float))]
        return statistics.mean(vals) if vals else 0.0

    def stage_time_ms(self, stage: str = "") -> dict | float:
        """C_cpu — Tempo médio por estágio.

        Args:
            stage: nome do estágio ("verify", "multikrum", "stark_gen").
                   Vazio retorna dict com todos.
        """
        stage_times = [r.get("stage_time_ms", {}) for r in self._records]
        if not stage_times:
            return {} if not stage else 0.0
        if stage:
            vals = [s[stage] for s in stage_times if s.get(stage, 0) > 0]
            return statistics.mean(vals) if vals else 0.0
        keys = set()
        for s in stage_times:
            keys.update(s.keys())
        return {
            k: statistics.mean(
                [s[k] for s in stage_times if s.get(k, 0) > 0]
            ) or 0.0
            for k in sorted(keys)
        }

    # ── Grupo C: dados externos ─────────────────────────

    def model_accuracy(self, X_val, y_val) -> float:
        """A_model — Acurácia do modelo final em validação."""
        from dataset.model import MLP
        model = MLP()
        if not self._records:
            return 0.0
        w = self._records[-1].get("global_weights")
        if not w:
            return 0.0
        model.set_weights(w)
        probs, _ = model.forward(X_val)
        preds = probs.argmax(axis=1)
        return float((preds == y_val).mean())

    def convergence_rounds(self, threshold: float = 0.01) -> int:
        """C_round — Primeira rodada onde loss estabiliza."""
        losses = [r.get("loss", float("inf")) for r in self._records]
        losses = [l for l in losses if l < float("inf")]
        if len(losses) < 3:
            return len(self._records)
        for i in range(2, len(losses)):
            if abs(losses[i] - losses[i-1]) < threshold:
                return i
        return len(losses)

    def compliance_overhead(self, baseline_latency: float = 0.0) -> dict:
        """O_conf — Overhead de latência vs baseline."""
        current = self.avg_latency()
        if baseline_latency <= 0:
            return {"overhead_pct": 0.0, "baseline": 0.0, "actual": current}
        overhead = ((current - baseline_latency) / baseline_latency) * 100
        return {"overhead_pct": overhead, "baseline": baseline_latency, "actual": current}

    def zkp_verification_success_rate(self) -> float:
        total_attempted = sum(r.get("snark_attempted", 0) for r in self._records)
        total_passed = sum(r.get("snark_passed", 0) for r in self._records)
        if total_attempted == 0:
            return 1.0
        return total_passed / total_attempted

    def all_metrics(self) -> dict:
        last = self._records[-1] if self._records else {}
        has_loss = "loss" in last
        loss_val = last.get("loss", 0.0) if has_loss else 0.0
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
            "comm_overhead_bytes": self.comm_overhead_bytes(),
            "vrf_uniformity": self.vrf_election_uniformity(),
            "vrf_diversity": self.vrf_diversity(),
            "zkp_success_rate": self.zkp_verification_success_rate(),
            "compliance": self.compliance_completeness(),
            "convergence_proxy": 1.0 / (1.0 + loss_val) if has_loss and loss_val > 0 else (1.0 if has_loss else 0.0),
            "convergence_rounds": self.convergence_rounds() if has_loss else 0,
            "final_loss": loss_val,
            "final_loss_normalized": loss_val / len(self._records) if has_loss and self._records else 0.0,
            "participation_traceability": self.participation_traceability(),
            "view_change_frequency": self.view_change_frequency(),
            "view_change_latency_ms": self.view_change_latency_ms(),
            "view_change_resistance": self.view_change_resistance(),
            "ledger_integrity": self.ledger_integrity(),
            "state_divergence_rate": self.state_divergence_rate(),
            "snark_proofs_total": self._round_mean("snark_proofs_total"),
            "snark_verify_projected_ms": self._round_mean("snark_verify_projected_ms"),
            "stage_time_ms": self.stage_time_ms(),
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
