import asyncio
import pickle
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

        total_adv = sum(1 for g in gradients if g.node_id.startswith("adv_"))
        total_honest = len(gradients) - total_adv

        comm_bytes = sum(len(pickle.dumps(g)) for g in gradients)
        block_bytes = len(pickle.dumps(avg_grad.tolist() if hasattr(avg_grad, 'tolist') else avg_grad))

        # Simulate simple consensus latency (fixed leader, no BFT overhead)
        t_end = time.time()

        return {
            "round": round_num,
            "leader": self.leader_id,
            "latency": t_end - t_start,
            "consensus_time_ms": (t_end - t_start) * 1000,
            "block_size_bytes": block_bytes,
            "comm_overhead_bytes": comm_bytes,
            "num_accepted": len(gradients),
            "num_rejected": 0,
            "num_adversarial": total_adv,
            "num_honest": total_honest,
            "rejected_adversarial": 0,
            "falsely_rejected": 0,
            "qc_emitted": True,
            "ledger_height": round_num + 1,
            "variant": "no_zkp",
            "snark_required": False,
            "snark_attempted": 0,
            "snark_passed": 0,
            "stark_proof_generated": False,
            "view_change_count": 0,
            "view_change_latency_ms": 0.0,
            "state_divergence": False,
            "ledger_integrity": True,
            "stage_time_ms": {},
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
