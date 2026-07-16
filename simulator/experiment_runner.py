import asyncio
from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent


class ExperimentRunner:
    def __init__(self, clusters: list[Cluster],
                 cloud: CloudComponent,
                 f: int = 1):
        self.clusters = clusters
        self.interregional = InterRegionalManager(
            clusters, n=len(clusters), f=f,
        )
        self.cloud = cloud
        self.metrics: list[dict] = []

    async def run_round(self, round_num: int) -> dict:
        global_output = await self.interregional.run_round(round_num)
        result = await self.cloud.process(global_output)
        self.metrics.append(result)
        return result

    async def run_experiment(self, num_rounds: int) -> list[dict]:
        for rnd in range(1, num_rounds + 1):
            await self.run_round(rnd)
        return self.metrics
