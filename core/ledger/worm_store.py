import logging
import pickle
from typing import Optional
from hfc_types.block import GlobalOutput

logger = logging.getLogger(__name__)

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


class WormStore:
    def __init__(self, db_path: Optional[str] = None):
        self._entries: list[GlobalOutput] = []
        self._db_path = db_path
        if db_path and HAS_SQLITE:
            self._init_db()
        elif db_path:
            # Configured persistence with no backend must not be a silent
            # no-op: entries would vanish on restart without anyone noticing.
            logger.warning(
                "WORM db_path=%s configured but sqlite3 is unavailable; "
                "entries will not be persisted across restarts", db_path,
            )

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_outputs (
                round INTEGER PRIMARY KEY,
                data BLOB
            )
        """)
        conn.commit()
        cursor = conn.execute("SELECT data FROM global_outputs ORDER BY round")
        for row in cursor.fetchall():
            self._entries.append(pickle.loads(row[0]))
        conn.close()

    def _persist(self, entry: GlobalOutput):
        if not self._db_path or not HAS_SQLITE:
            return
        conn = sqlite3.connect(self._db_path)
        # INSERT OR IGNORE, not REPLACE: the dup check above is per-instance,
        # so a stale instance can reach this call for an already-stored round.
        # The ledger is append-only — the first record for a round wins.
        conn.execute(
            "INSERT OR IGNORE INTO global_outputs (round, data) VALUES (?, ?)",
            (entry.round_num, pickle.dumps(entry)),
        )
        conn.commit()
        conn.close()

    def append(self, entry: GlobalOutput) -> bool:
        for e in self._entries:
            if e.round_num == entry.round_num:
                return False
        self._entries.append(entry)
        self._persist(entry)
        return True

    def get_height(self) -> int:
        return len(self._entries)

    def get_entry(self, round_num: int) -> Optional[GlobalOutput]:
        for e in self._entries:
            if e.round_num == round_num:
                return e
        return None
