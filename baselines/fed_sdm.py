import time
import numpy as np
from typing import Optional
from hfc_types.messages import Gradient
from simulator.orchestrator import ExperimentResult
from baselines.base import AbsBaseline


class FedSDM(AbsBaseline):
    """FedSDM baseline: FedAvg + fixed leader (Ganache-style).

    - Aggregation: simple mean of all gradients (FedAvg)
    - Consensus: a single "leader" proposes blocks; no BFT
    - No adversarial filtering (all gradients accepted)
    - No ZKP
    """

    def __init__(self, num_nodes: int = 5, f: int = 1):
        self.num_nodes = num_nodes
        self.f = f
        self.leader_id = "n0"

    async def run_round(
        self, round_num: int, gradients: list[Gradient]
    ) -> dict:
        t_start = time.time()
        grad_vectors = [np.array(g.data) for g in gradients]
        avg_grad = np.mean(grad_vectors, axis=0)
        t_end = time.time()
        return {
            "round": round_num,
            "leader": self.leader_id,
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
