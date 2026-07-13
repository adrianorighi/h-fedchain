from abc import ABC, abstractmethod
from typing import Optional
from hfc_types.messages import Gradient
from simulator.orchestrator import ExperimentResult


class AbsBaseline(ABC):
    """Abstract interface for baseline FL + consensus implementations."""

    @abstractmethod
    async def run_round(
        self, round_num: int, gradients: list[Gradient]
    ) -> dict:
        ...

    @abstractmethod
    async def run_experiment(
        self,
        num_rounds: int,
        gradients_per_round: list[list[Gradient]],
        warmup: int = 10,
    ) -> ExperimentResult:
        ...
