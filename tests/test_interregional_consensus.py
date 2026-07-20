import pytest
from core.hotstuff.interregional import InterRegionalConsensus


def test_interregional_3_phases():
    n, f = 4, 1
    ir = InterRegionalConsensus(n_clusters=n, f=f)
    outputs = [(f"c{i}", b"gradient_data") for i in range(n)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is not None
    assert ir.qc is not None
    assert ir.qc.is_valid(ir.quorum_size())


def test_interregional_rejects_insufficient():
    ir = InterRegionalConsensus(n_clusters=4, f=1)
    outputs = [(f"c{i}", b"data") for i in range(2)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is None


def test_interregional_qc_has_signatures():
    n, f = 4, 1
    ir = InterRegionalConsensus(n_clusters=n, f=f)
    outputs = [(f"c{i}", b"gradient_data") for i in range(n)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is not None
    expected_quorum = ir.quorum_size()
    assert len(ir.qc.signatures) >= expected_quorum


def test_interregional_3_phases_n3():
    n, f = 3, 1
    ir = InterRegionalConsensus(n_clusters=n, f=f)
    outputs = [(f"c{i}", b"g{i}") for i in range(n)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is not None
    assert ir.qc is not None


def test_interregional_rejects_insufficient_n5_f2():
    ir = InterRegionalConsensus(n_clusters=5, f=2)
    outputs = [(f"c{i}", b"data") for i in range(2)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is None


def test_interregional_quorum_guard_n5_f2():
    ir = InterRegionalConsensus(n_clusters=5, f=2)
    outputs = [(f"c{i}", b"data") for i in range(3)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result is None


def test_interregional_input_integrity():
    n, f = 4, 1
    ir = InterRegionalConsensus(n_clusters=n, f=f)
    expected = [b"g0", b"g1", b"g2", b"g3"]
    outputs = [(f"c{i}", expected[i]) for i in range(n)]
    result = ir.run_round(outputs, leader_id="c0")
    assert result == expected
