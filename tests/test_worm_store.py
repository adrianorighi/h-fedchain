import pytest
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
