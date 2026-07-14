from typing import Optional
from core.hotstuff.messages import ViewChangeMessage, NewViewMessage
from core.pki import sign as pki_sign


class ViewChangeHandler:
    def __init__(self, n: int, f: int):
        self.n = n
        self.f = f
        self.current_view = 1
        self._highest_qc: Optional[tuple[int, bytes]] = None

    def should_change_view(self, timeout: bool = False) -> bool:
        return timeout

    def record_highest_qc(self, round: int, qc: bytes):
        if self._highest_qc is None or round > self._highest_qc[0]:
            self._highest_qc = (round, qc)

    def next_leader(self, node_ids: list[str]) -> str:
        current_idx = (self.current_view - 1) % len(node_ids)
        next_idx = (current_idx + 1) % len(node_ids)
        self.current_view += 1
        return node_ids[next_idx]

    def create_view_change(
        self, node_id: str, new_view: int, sk: bytes
    ) -> ViewChangeMessage:
        highest_qc = self._highest_qc
        msg = str(new_view).encode()
        if highest_qc:
            msg += highest_qc[1]
        sig = pki_sign(sk, msg)
        return ViewChangeMessage(
            node_id=node_id,
            new_view=new_view,
            highest_qc=highest_qc,
            signature=sig,
        )

    def create_new_view(
        self, leader_id: str, new_view: int, qc_set: list[bytes]
    ) -> NewViewMessage:
        return NewViewMessage(
            leader_id=leader_id,
            new_view=new_view,
            qc_set=qc_set,
            signature=b"",
        )
