import asyncio
from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent


class ExperimentRunner:
    def __init__(self, clusters: list[Cluster],
                 interregional: InterRegionalManager,
                 cloud: CloudComponent):
        self.clusters = clusters
        self.interregional = interregional
        self.cloud = cloud
        self.metrics: list[dict] = []

    async def run_round(self, round_num: int) -> dict:
        cluster_outputs = await asyncio.gather(
            *[c.run_round(round_num) for c in self.clusters]
        )
        global_output = await self.interregional.process(list(cluster_outputs))
        result = await self.cloud.process(global_output)
        self.metrics.append(result)
        return result

    async def run_experiment(self, num_rounds: int) -> list[dict]:
        for rnd in range(1, num_rounds + 1):
            await self.run_round(rnd)
        return self.metrics
