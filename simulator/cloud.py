from typing import Optional
from zkp.stark import StarkVerifier
from hfc_types.block import GlobalOutput
from core.ledger.worm_store import WormStore
from core.cloud.model_validation import ModelValidationGate, ValidationResult
from core.audit.logger import AuditLogger
from services.model_registry import ModelRegistry
from services.identity_service import GlobalIdentityService
from monitoring.tracer import Tracer


class CloudComponent:
    def __init__(self, tracer: Optional[Tracer] = None):
        self.validation_gate = ModelValidationGate()
        self.worm = WormStore()
        self.converged = False
        self.model_registry = ModelRegistry()
        self.identity_service = GlobalIdentityService()
        self.ca_vk = self.identity_service.ca_vk
        self.stark_verifier = StarkVerifier()
        self.audit_logger = AuditLogger()
        self.tracer = tracer

    def validate_update(self, w_old: dict, w_new: dict, loss: float = None):
        return self.validation_gate.validate(w_old, w_new, loss)

    def validate_model(self, w_old: dict, w_new: dict,
                       loss: float = None, round_num: int = 0) -> ValidationResult:
        result = self.validation_gate.validate(w_old, w_new, loss)
        self.audit_logger.log("MODEL_VALIDATION", "cloud", round_num, {
            "accepted": result.accepted,
            "loss": result.loss,
            "delta_norm": result.delta_norm,
            "reason": result.reason,
        })
        if result.accepted:
            self.converged = True
        return result

    async def process(self, global_output: GlobalOutput) -> dict:
        if self.tracer is not None:
            ctx = self.tracer.span("cloud_process", "cloud", round_num=global_output.round_num)
            ctx.__enter__()
        else:
            ctx = None
        try:
            return await self._process_impl(global_output)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    async def _process_impl(self, global_output: GlobalOutput) -> dict:
        stark_verified = True
        if global_output.pi_inter is not None:
            is_valid = await self.stark_verifier.verify(
                global_output.pi_inter,
                global_output.pi_inter.public_inputs,
            )
            self.audit_logger.log("STARK_VERIFY", "cloud", global_output.round_num,
                                  {"verified": is_valid, "n_active_clusters": global_output.n_active_clusters})
            if not is_valid:
                return {
                    "round": global_output.round_num,
                    "n_active_clusters": 0,
                    "converged": self.converged,
                    "stark_verified": False,
                }

        self.worm.append(global_output)
        self.audit_logger.log("WORM_APPEND", "cloud", global_output.round_num,
                              {"n_active_clusters": global_output.n_active_clusters})
        return {
            "round": global_output.round_num,
            "n_active_clusters": global_output.n_active_clusters,
            "converged": self.converged,
            "stark_verified": stark_verified,
        }
