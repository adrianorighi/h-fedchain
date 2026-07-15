import pickle
import time
from typing import Optional
from hfc_types.block import Block, LedgerEntry

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


class LedgerStore:
    def __init__(self, db_path: Optional[str] = None):
        self._entries: list[LedgerEntry] = []
        self._db_path = db_path
        if db_path and HAS_SQLITE:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                hash BLOB PRIMARY KEY,
                round INTEGER,
                data BLOB,
                stored_at REAL
            )
        """)
        conn.commit()
        cursor = conn.execute("SELECT data FROM blocks ORDER BY round")
        for row in cursor.fetchall():
            entry = pickle.loads(row[0])
            self._entries.append(entry)
        conn.close()

    def _persist(self, block: Block):
        if not self._db_path or not HAS_SQLITE:
            return
        conn = sqlite3.connect(self._db_path)
        entry = self._entries[-1]
        conn.execute(
            "INSERT OR REPLACE INTO blocks (hash, round, data, stored_at) VALUES (?, ?, ?, ?)",
            (block.hash, block.round, pickle.dumps(entry), entry.stored_at),
        )
        conn.commit()
        conn.close()

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
        self._persist(block)
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
