import time
import pytest
from core.ledger.store import LedgerStore
from hfc_types.block import Block, QuorumCertificate, LedgerEntry
from hfc_types.messages import MessageType


def make_block(round_num: int, prev_hash: bytes) -> Block:
    return Block(
        round=round_num,
        gradient_hash=b"gh" + str(round_num).encode(),
        qc_commit=QuorumCertificate(
            round=round_num,
            block_hash=b"",
            signatures=[("n1", b"sig")],
            msg_type=MessageType.COMMIT,
        ),
        stark_proof=None,
        accepted_devices=["d1"],
        rejected_devices=[],
        timestamp=time.time(),
        prev_hash=prev_hash,
    )


def test_append_and_get_block():
    store = LedgerStore()
    genesis = make_block(0, b"\x00" * 32)
    b1 = make_block(1, genesis.hash)
    assert store.append(genesis) is True
    assert store.append(b1) is True
    assert store.get_block(genesis.hash) is not None
    assert store.get_block(b"invalid") is None


def test_verify_chain():
    store = LedgerStore()
    b0 = make_block(0, b"\x00" * 32)
    b1 = make_block(1, b0.hash)
    store.append(b0)
    store.append(b1)
    assert store.verify_chain() is True


def test_verify_chain_detects_tamper():
    store = LedgerStore()
    b0 = make_block(0, b"\x00" * 32)
    b1 = make_block(1, b0.hash)
    store.append(b0)
    store.append(b1)
    tampered = make_block(1, b"tampered")
    store._entries[-1] = LedgerEntry(block=tampered, node_id="", stored_at=0, verified=True)
    assert store.verify_chain() is False


def test_get_height():
    store = LedgerStore()
    assert store.get_height() == 0
    for i in range(5):
        prev = b"\x00" * 32 if i == 0 else store._entries[-1].block.hash
        store.append(make_block(i, prev))
    assert store.get_height() == 5


def test_append_rejects_invalid_hash_chain():
    store = LedgerStore()
    b0 = make_block(0, b"\x00" * 32)
    b1 = make_block(1, b"does_not_match")
    store.append(b0)
    assert store.append(b1) is False
