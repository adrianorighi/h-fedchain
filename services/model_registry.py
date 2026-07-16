from dataclasses import dataclass
from typing import Optional
import time
import pickle
import hashlib
from core.cloud.model_validation import ValidationResult


@dataclass
class ModelVersion:
    version: int
    weights: bytes
    model_hash: str
    loss: float
    delta_norm: float
    created_at: float
    parent_version: Optional[int] = None


class ModelRegistry:
    def __init__(self):
        self._versions: list[ModelVersion] = []
        self._current_version: Optional[ModelVersion] = None

    def register(self, weights: dict, validation: ValidationResult,
                 parent_version: Optional[int] = None) -> ModelVersion:
        weights_bytes = pickle.dumps(weights)
        model_hash = hashlib.sha256(weights_bytes).hexdigest()

        version = ModelVersion(
            version=len(self._versions) + 1,
            weights=weights_bytes,
            model_hash=model_hash,
            loss=validation.loss,
            delta_norm=validation.delta_norm,
            created_at=time.time(),
            parent_version=parent_version or (
                self._current_version.version if self._current_version else None
            ),
        )
        self._versions.append(version)
        self._current_version = version
        return version

    def get_latest(self) -> Optional[ModelVersion]:
        return self._current_version

    def get_version(self, v: int) -> Optional[ModelVersion]:
        if 0 < v <= len(self._versions):
            return self._versions[v - 1]
        return None

    def rollback(self, version: int) -> Optional[ModelVersion]:
        target = self.get_version(version)
        if target:
            self._current_version = target
        return target

    def list_versions(self) -> list[int]:
        return [v.version for v in self._versions]

    def count(self) -> int:
        return len(self._versions)
