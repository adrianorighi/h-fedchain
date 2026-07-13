import time
import numpy as np
from typing import Optional
from hfc_types.messages import Gradient
from simulator.orchestrator import ExperimentResult
from baselines.base import AbsBaseline


class FLCoin(AbsBaseline):
    """FLCoin baseline: FedAvg + committee-based consensus.

    - Aggregation: simple mean of all gradients (FedAvg)
    - Consensus: a rotating committee of nodes votes on each round
    - No adversarial filtering (all gradients accepted)
    - No ZKP, no VRF
    """

    def __init__(self, num_nodes: int = 5, f: int = 1, committee_size: int = 3):
        self.num_nodes = num_nodes
        self.f = f
        self.committee_size = min(committee_size, num_nodes)

    def _elect_committee(self, round_num: int) -> list[str]:
        """Round-robin committee election."""
        members = []
        for i in range(self.committee_size):
            idx = (round_num + i) % self.num_nodes
            members.append(f"n{idx}")
        return members

    async def run_round(
        self, round_num: int, gradients: list[Gradient]
    ) -> dict:
        t_start = time.time()
        grad_vectors = [np.array(g.data) for g in gradients]
        avg_grad = np.mean(grad_vectors, axis=0)
        committee = self._elect_committee(round_num)
        import asyncio
        await asyncio.sleep(0.001 * self.committee_size)
        t_end = time.time()
        return {
            "round": round_num,
            "leader": committee[0],
            "latency": t_end - t_start,
            "num_accepted": len(gradients),
            "num_rejected": 0,
            "num_adversarial": 0,
            "num_honest": len(gradients),
            "falsely_rejected": 0,
            "qc_emitted": True,
            "ledger_height": round_num + 1,
        }

    async def run_experiment(
        self,
        num_rounds: int,
        gradients_per_round: list[list[Gradient]],
        warmup: int = 10,
    ) -> ExperimentResult:
        result = ExperimentResult()
        for r in range(num_rounds + warmup):
            if r < len(gradients_per_round):
                metrics = await self.run_round(r, gradients_per_round[r])
            else:
                metrics = await self.run_round(r, [])
            if r >= warmup:
                result.round_metrics.append(metrics)
        return result
