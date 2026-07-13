import time
from typing import Optional
from hfc_types.block import Block, LedgerEntry


class LedgerStore:
    def __init__(self):
        self._entries: list[LedgerEntry] = []

    def append(self, block: Block) -> bool:
        if self._entries:
            last_hash = self._entries[-1].block.hash
            if block.prev_hash != last_hash:
                return False
        elif block.prev_hash != b"\x00" * 32:
            return False
        entry = LedgerEntry(
            block=block,
            node_id="",
            stored_at=time.time(),
            verified=True,
        )
        self._entries.append(entry)
        return True

    def get_block(self, block_hash: bytes) -> Optional[Block]:
        for e in self._entries:
            if e.block.hash == block_hash:
                return e.block
        return None

    def get_height(self) -> int:
        return len(self._entries)

    def verify_chain(self) -> bool:
        for i in range(1, len(self._entries)):
            prev = self._entries[i - 1].block
            cur = self._entries[i].block
            if cur.prev_hash != prev.hash:
                return False
        return True
