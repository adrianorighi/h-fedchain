import numpy as np
from typing import Optional
from hfc_types.messages import Gradient
from dataset.model import MLP


class EdgeWorker:
    """Represents one Edge device with its local data partition and MLP model.

    Each round: receives global weights -> trains locally -> returns gradient Dw_i.
    """

    def __init__(
        self,
        device_id: str,
        indices: list[int],
        all_data: np.ndarray,
        all_labels: np.ndarray,
        is_adversarial: bool = False,
        attack_type: str = "label_flip",
        input_dim: int = 12000,
        hidden_dim: int = 64,
        num_classes: int = 5,
    ):
        self.device_id = device_id
        self.indices = indices
        self.is_adversarial = is_adversarial
        self.attack_type = attack_type
        self.num_classes = num_classes

        # Extract local data partition
        self.local_data = all_data[indices]  # (N_local, 12, 1000)
        self.local_labels = all_labels[indices].copy()

        # Flatten ECG: (N, 12, 1000) -> (N, 12000)
        self.X = self.local_data.reshape(len(indices), -1).astype(np.float64)
        self.y = self.local_labels.copy()

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    def _apply_attack(self, round_num: int) -> np.ndarray:
        """Apply adversarial modification to labels.

        Label-flip: alternate labels at specified intervals.
        Sign-flip:  negate the gradient after training (handled by caller).
        """
        if self.attack_type == "label_flip":
            # Flip labels every 5 rounds (article Scenario 3)
            if (round_num // 5) % 2 == 1:
                return (self.y + 1) % self.num_classes
        return self.y

    def train_round(self, global_weights: dict, round_num: int = 0) -> Gradient:
        """Perform one round of local training.

        Args:
            global_weights: current global model weights dict
            round_num: current round number (for adversarial schedule)

        Returns:
            Gradient with flat gradient vector
        """
        y_effective = self._apply_attack(round_num) if self.is_adversarial else self.y

        if self.is_adversarial and self.attack_type == "sign_flip":
            grad = self.model.compute_gradient(self.X, y_effective, global_weights)
            grad = -grad  # sign-flip attack
        else:
            grad = self.model.compute_gradient(self.X, y_effective, global_weights)

        return Gradient(
            node_id=self.device_id,
            round=round_num,
            data=grad.tolist(),
        )

    @property
    def num_samples(self) -> int:
        return len(self.indices)
