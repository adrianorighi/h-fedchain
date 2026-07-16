from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ValidationResult:
    accepted: bool
    loss: float
    delta_norm: float
    reason: str = ""


class ModelValidationGate:
    """Avalia criterios de convergencia antes de registrar novo modelo."""

    def __init__(self, loss_max: float = 10.0, delta_conv: float = 100.0):
        self.loss_max = loss_max
        self.delta_conv = delta_conv

    def validate(self, w_old: Optional[dict], w_new: dict,
                 loss: Optional[float] = None) -> ValidationResult:
        delta_norm = 0.0
        if w_old is not None:
            for key in w_new:
                if key in w_old:
                    delta_norm += float(np.linalg.norm(
                        np.array(w_new[key]) - np.array(w_old[key])
                    ))

        if loss is not None and loss > self.loss_max:
            return ValidationResult(
                accepted=False, loss=loss, delta_norm=delta_norm,
                reason=f"loss {loss:.4f} > max {self.loss_max}"
            )

        if delta_norm > self.delta_conv:
            return ValidationResult(
                accepted=False, loss=loss or 0.0, delta_norm=delta_norm,
                reason=f"delta_norm {delta_norm:.4f} > conv {self.delta_conv}"
            )

        return ValidationResult(
            accepted=True, loss=loss or 0.0, delta_norm=delta_norm
        )
