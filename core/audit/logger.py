import time


class AuditLogger:
    EVENT_TYPES = [
        "REJ_PKI", "REJ_ZKP", "REJ_MULTIKRUM",
        "LEADER_ELECTION", "VIEW_CHANGE", "QC_COMMIT",
        "INSUF_CONTRIBUTIONS", "ROUND_ABORTED",
        "GLOBAL_AGGREGATION", "MODEL_UPDATE",
        "STARK_VERIFY", "WORM_APPEND",
        "MODEL_VALIDATION",
    ]

    def __init__(self):
        self._entries: list[dict] = []

    def log(self, event_type: str, node_id: str, round_num: int,
            metadata: dict | None = None):
        self._entries.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "node_id": node_id,
            "round": round_num,
            "metadata": metadata or {},
        })

    def get_entries(self) -> list[dict]:
        return list(self._entries)

    def count_by_type(self, event_type: str) -> int:
        return sum(1 for e in self._entries if e["event_type"] == event_type)

    def count_by_round(self, round_num: int) -> int:
        return sum(1 for e in self._entries if e["round"] == round_num)

    def clear(self):
        self._entries.clear()

    def to_list(self) -> list[dict]:
        return self.get_entries()
