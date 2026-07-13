import pytest
import numpy as np
from hfc_types.messages import Gradient
from baselines.fed_sdm import FedSDM
from baselines.flcoin import FLCoin


def make_gradients(n: int = 10) -> list[Gradient]:
    return [
        Gradient(node_id=f"d{i}", round=0, data=np.random.randn(10).tolist())
        for i in range(n)
    ]


class TestFedSDM:
    @pytest.mark.asyncio
    async def test_run_round_returns_metrics(self):
        baseline = FedSDM(num_nodes=5)
        grads = make_gradients(10)
        metrics = await baseline.run_round(1, grads)
        assert "latency" in metrics
        assert metrics["qc_emitted"] is True
        assert metrics["num_accepted"] == 10

    @pytest.mark.asyncio
    async def test_run_experiment(self):
        baseline = FedSDM(num_nodes=5)
        grads_per_round = [make_gradients(10) for _ in range(5)]
        result = await baseline.run_experiment(3, grads_per_round, warmup=2)
        assert len(result.round_metrics) == 3


class TestFLCoin:
    @pytest.mark.asyncio
    async def test_run_round_returns_metrics(self):
        baseline = FLCoin(num_nodes=5, committee_size=3)
        grads = make_gradients(10)
        metrics = await baseline.run_round(1, grads)
        assert "latency" in metrics
        assert metrics["qc_emitted"] is True

    @pytest.mark.asyncio
    async def test_committee_rotation(self):
        baseline = FLCoin(num_nodes=5, committee_size=3)
        c1 = baseline._elect_committee(0)
        c2 = baseline._elect_committee(1)
        assert len(c1) == 3
        assert c1 != c2, "Committee should rotate"

    @pytest.mark.asyncio
    async def test_run_experiment(self):
        baseline = FLCoin(num_nodes=5, committee_size=3)
        grads_per_round = [make_gradients(10) for _ in range(5)]
        result = await baseline.run_experiment(3, grads_per_round, warmup=2)
        assert len(result.round_metrics) == 3
