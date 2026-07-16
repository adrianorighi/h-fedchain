import asyncio
from typing import Optional
from hfc_types.block import RegionalOutput
from simulator.orchestrator import Orchestrator


class Cluster:
    def __init__(
        self,
        cluster_id: str,
        nodes_per_cluster: int,
        devices_per_cluster: int,
        f: int,
        latency_ms: float,
        variant: str = "no_zkp",
    ):
        self.cluster_id = cluster_id
        self.orch = Orchestrator(
            num_clusters=1,
            nodes_per_cluster=nodes_per_cluster,
            devices_per_cluster=devices_per_cluster,
            f=f,
            latency_ms=latency_ms,
            variant=variant,
        )
        self.orch.setup()

    @property
    def nodes(self):
        return self.orch.nodes

    @property
    def representative_id(self) -> str:
        return self.nodes[0].node_id if self.nodes else self.cluster_id

    @property
    def n_devices(self) -> int:
        return self.orch.devices_per_cluster

    async def get_regional_output(self, round_num: int) -> RegionalOutput:
        return await self.run_round(round_num)

    async def run_round(self, round_num: int) -> RegionalOutput:
        result = await self.orch.run_round(round_num)
        if not result.get("qc_emitted", False):
            raise RuntimeError(
                f"Cluster {self.cluster_id} round {round_num} failed"
            )
        node = self.orch.nodes[0]
        block = node.ledger._entries[-1].block
        return RegionalOutput(
            delta_w=block.gradient_hash,
            stark_proof=block.stark_proof,
            qc_commit=block.qc_commit,
            n_devices=len(node.peers) + 1,
            round_num=round_num,
            cluster_id=self.cluster_id,
        )
