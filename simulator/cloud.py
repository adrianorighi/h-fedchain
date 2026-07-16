from hfc_types.block import GlobalOutput
from core.ledger.worm_store import WormStore
from core.cloud.model_validation import ModelValidationGate


class CloudComponent:
    def __init__(self):
        self.validation_gate = ModelValidationGate()
        self.worm = WormStore()
        self.converged = False

    def validate_update(self, w_old: dict, w_new: dict, loss: float = None):
        return self.validation_gate.validate(w_old, w_new, loss)

    async def process(self, global_output: GlobalOutput) -> dict:
        self.worm.append(global_output)
        return {
            "round": global_output.round_num,
            "n_active_clusters": global_output.n_active_clusters,
            "converged": self.converged,
        }
