import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InterRegionalEntry:
    round_num: int
    delta_w_hash: bytes
    pi_inter_hash: bytes
    clusters: list[str]
    prev_hash: bytes
    timestamp: float


class InterRegionalLedger:
    def __init__(self):
        self._entries: list[InterRegionalEntry] = []
        self._hash_chain: list[bytes] = [b"\x00" * 32]

    def append(self, round_num: int, delta_w: bytes, pi_inter: bytes,
               clusters: list[str]) -> None:
        entry = InterRegionalEntry(
            round_num=round_num,
            delta_w_hash=hashlib.sha256(delta_w).digest(),
            pi_inter_hash=hashlib.sha256(pi_inter).digest(),
            clusters=sorted(clusters),
            prev_hash=self._hash_chain[-1],
            timestamp=time.time(),
        )
        self._entries.append(entry)
        chain_input = entry.prev_hash + entry.delta_w_hash + entry.pi_inter_hash
        self._hash_chain.append(hashlib.sha256(chain_input).digest())

    def get_height(self) -> int:
        return len(self._entries)

    def get_entry(self, round_num: int) -> Optional[InterRegionalEntry]:
        for e in self._entries:
            if e.round_num == round_num:
                return e
        return None

    def verify_chain(self) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != self._hash_chain[i]:
                return False
            chain_input = entry.prev_hash + entry.delta_w_hash + entry.pi_inter_hash
            expected_hash = hashlib.sha256(chain_input).digest()
            if self._hash_chain[i + 1] != expected_hash:
                return False
        return True
