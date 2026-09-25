import logging
import pickle

from core.ledger.worm_store import WormStore
from hfc_types.block import GlobalOutput


def test_worm_append_and_get():
    store = WormStore()
    entry = GlobalOutput(delta_w_inter=b"data", pi_inter=None, n_active_clusters=3, round_num=1)
    assert store.append(entry) is True
    assert store.get_height() == 1
    retrieved = store.get_entry(1)
    assert retrieved is not None
    assert retrieved.delta_w_inter == b"data"
    assert retrieved.round_num == 1


def test_worm_rejects_duplicate_round():
    store = WormStore()
    entry1 = GlobalOutput(delta_w_inter=b"data1", pi_inter=None, n_active_clusters=3, round_num=1)
    entry2 = GlobalOutput(delta_w_inter=b"data2", pi_inter=None, n_active_clusters=3, round_num=1)
    assert store.append(entry1) is True
    assert store.append(entry2) is False


def test_worm_nonexistent_round():
    store = WormStore()
    assert store.get_entry(99) is None


def test_worm_multiple_entries():
    store = WormStore()
    for i in range(3):
        entry = GlobalOutput(delta_w_inter=f"data{i}".encode(), pi_inter=None, n_active_clusters=3, round_num=i)
        assert store.append(entry) is True
    assert store.get_height() == 3
    assert store.get_entry(0) is not None
    assert store.get_entry(2) is not None


# ----------------------------------------------------------------------
# GlobalOutput.accepted compliance flag (Task 7 review fix)
# ----------------------------------------------------------------------

def test_global_output_accepted_defaults_to_true():
    entry = GlobalOutput(
        delta_w_inter=b"x", pi_inter=None, n_active_clusters=1, round_num=1,
    )
    assert entry.accepted is True, "accepted must default to True"


def test_global_output_rejected_flag_round_trips_through_pickle():
    entry = GlobalOutput(
        delta_w_inter=b"x", pi_inter=None, n_active_clusters=1, round_num=1,
        accepted=False,
    )
    restored = pickle.loads(pickle.dumps(entry))
    assert restored.accepted is False


def test_global_output_pickled_before_accepted_field_unpickles():
    """Entries pickled before the field existed have no `accepted` in their
    instance __dict__; the class default must kick in (True)."""
    entry = GlobalOutput(
        delta_w_inter=b"old", pi_inter=None, n_active_clusters=2, round_num=9,
    )
    entry.__dict__.pop("accepted")  # simulate a pre-field pickle payload
    restored = pickle.loads(pickle.dumps(entry))
    assert restored.delta_w_inter == b"old"
    assert restored.accepted is True


# ----------------------------------------------------------------------
# SQLite persistence (Task 7 review fixes 4 and 5)
# ----------------------------------------------------------------------

def test_db_path_without_sqlite_warns_once(caplog, monkeypatch):
    """A configured db_path with sqlite3 unavailable must not be a silent
    no-op: warn once at init (repo style: module logger)."""
    import core.ledger.worm_store as worm_module

    monkeypatch.setattr(worm_module, "HAS_SQLITE", False)

    with caplog.at_level(logging.WARNING):
        store = WormStore(db_path="/nonexistent/worm.db")

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got: {warnings}"
    assert "sqlite" in warnings[0].lower()
    # in-memory append still works without a backend
    entry = GlobalOutput(
        delta_w_inter=b"d", pi_inter=None, n_active_clusters=1, round_num=1,
    )
    assert store.append(entry) is True


def test_persist_is_append_only_insert_or_ignore(monkeypatch):
    """_persist must never overwrite an existing round: the in-memory dup
    check is per-instance, so a stale instance can reach _persist for an
    already-stored round — INSERT OR IGNORE keeps the first record (REPLACE
    would clobber it). Uses a fake sqlite backend so it runs without the
    sqlite3 module too."""
    import core.ledger.worm_store as worm_module

    executed: list[str] = []

    class _FakeCursor:
        def fetchall(self):
            return []

    class _FakeConn:
        def execute(self, sql, params=None):
            executed.append(sql)
            return _FakeCursor()

        def commit(self):
            pass

        def close(self):
            pass

    class _FakeSqlite3:
        @staticmethod
        def connect(path):
            return _FakeConn()

    monkeypatch.setattr(worm_module, "HAS_SQLITE", True)
    monkeypatch.setattr(worm_module, "sqlite3", _FakeSqlite3, raising=False)

    store_a = WormStore(db_path="worm.db")
    store_b = WormStore(db_path="worm.db")  # stale: its dup check misses round 1

    first = GlobalOutput(
        delta_w_inter=b"first", pi_inter=None, n_active_clusters=1, round_num=1,
    )
    late = GlobalOutput(
        delta_w_inter=b"late", pi_inter=None, n_active_clusters=1, round_num=1,
    )
    assert store_a.append(first) is True
    assert store_b.append(late) is True  # passes store_b's own dup check

    inserts = [s for s in executed if s.lstrip().upper().startswith("INSERT")]
    assert inserts, "append must persist through _persist"
    assert all("INSERT OR IGNORE" in s for s in inserts), (
        f"_persist must be append-only (INSERT OR IGNORE), got: {inserts}"
    )
    assert not any("REPLACE" in s for s in inserts), (
        "INSERT OR REPLACE would clobber the first record for a round"
    )
