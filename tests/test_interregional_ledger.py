from dataclasses import FrozenInstanceError

import pytest
from core.ledger.interregional_ledger import InterRegionalLedger


def test_append_and_verify():
    ledger = InterRegionalLedger()
    ledger.append(round_num=1, delta_w=b"abcd", pi_inter=b"proof123",
                  clusters=["c0", "c1"])
    assert ledger.get_height() == 1


def test_chain_integrity():
    ledger = InterRegionalLedger()
    ledger.append(1, b"dw1", b"pi1", ["c0"])
    ledger.append(2, b"dw2", b"pi2", ["c0", "c1"])
    assert ledger.verify_chain()


def test_detect_tamper():
    ledger = InterRegionalLedger()
    ledger.append(1, b"dw1", b"pi1", ["c0"])
    ledger._hash_chain[1] = b"\xff" * 32
    assert not ledger.verify_chain()


def test_get_nonexistent():
    ledger = InterRegionalLedger()
    assert ledger.get_entry(99) is None


def test_multiple_entries_chain():
    ledger = InterRegionalLedger()
    for i in range(5):
        ledger.append(i, f"dw{i}".encode(), f"pi{i}".encode(), ["c0"])
    assert ledger.get_height() == 5
    assert ledger.verify_chain()
