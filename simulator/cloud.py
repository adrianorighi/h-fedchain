from hfc_types.block import GlobalOutput
from core.ledger.worm_store import WormStore


class CloudComponent:
    def __init__(self):
        self.worm = WormStore()
        self.converged = False

    async def process(self, global_output: GlobalOutput) -> dict:
        self.worm.append(global_output)
        return {
            "round": global_output.round_num,
            "n_active_clusters": global_output.n_active_clusters,
            "converged": self.converged,
        }
