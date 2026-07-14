from typing import Optional
from hfc_types.block import GlobalOutput


class WormStore:
    def __init__(self):
        self._entries: list[GlobalOutput] = []

    def append(self, entry: GlobalOutput) -> bool:
        for e in self._entries:
            if e.round_num == entry.round_num:
                return False
        self._entries.append(entry)
        return True

    def get_height(self) -> int:
        return len(self._entries)

    def get_entry(self, round_num: int) -> Optional[GlobalOutput]:
        for e in self._entries:
            if e.round_num == round_num:
                return e
        return None
