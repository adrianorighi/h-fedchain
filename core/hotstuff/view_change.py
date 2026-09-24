import time
from typing import Optional
from core.hotstuff.messages import ViewChangeMessage, NewViewMessage
from core.pki import sign as pki_sign


class ViewChangeHandler:
    def __init__(self, n: int, f: int):
        self.n = n
        self.f = f
        self.current_view = 1
        self._highest_qc: Optional[tuple[int, bytes]] = None
        self._vc_start_time: Optional[float] = None
        self._vc_durations: list[float] = []

    def should_change_view(self, timeout: bool = False) -> bool:
        if timeout and self._vc_start_time is None:
            self._vc_start_time = time.time()
        return timeout

    def record_highest_qc(self, round: int, qc: bytes):
        if self._highest_qc is None or round > self._highest_qc[0]:
            self._highest_qc = (round, qc)

    def next_leader(self, node_ids: list[str], vrf_candidates: Optional[list[tuple[str, bytes]]] = None) -> str:
        if vrf_candidates and self.current_view < len(vrf_candidates):
            leader = vrf_candidates[self.current_view][0]
        else:
            current_idx = (self.current_view - 1) % len(node_ids)
            leader = node_ids[(current_idx + 1) % len(node_ids)]
        self.current_view += 1
        if self._vc_start_time is not None:
            self._vc_durations.append(time.time() - self._vc_start_time)
            self._vc_start_time = None
        return leader

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
