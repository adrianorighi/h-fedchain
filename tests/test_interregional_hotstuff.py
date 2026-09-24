import asyncio
import pickle

from core.hotstuff.interregional import InterRegionalConsensus


async def test_interregional_quorum_size():
    ir = InterRegionalConsensus(n_clusters=4, f=1)
    assert ir.quorum_size() == 3  # ceil(2*4/3)


async def test_interregional_accepts_sufficient_inputs():
    ir = InterRegionalConsensus(n_clusters=4, f=1)
    outputs = [
        ("c0", pickle.dumps({"W1": [0.1]})),
        ("c1", pickle.dumps({"W1": [0.2]})),
        ("c2", pickle.dumps({"W1": [0.3]})),
        ("c3", pickle.dumps({"W1": [0.4]})),
    ]
    result = await ir.run_round(outputs, "c0")
    assert result is not None
    assert len(result) == 4


async def test_interregional_rejects_insufficient():
    ir = InterRegionalConsensus(n_clusters=4, f=1)
    outputs = [("c0", pickle.dumps({"W1": [0.1]}))]
    result = await ir.run_round(outputs, "c0")
    assert result is None


async def test_interregional_quorum_size_n3():
    ir = InterRegionalConsensus(n_clusters=3, f=1)
    assert ir.quorum_size() == 2  # ceil(2*3/3)
