import numpy as np
from typing import Optional


class DirichletPartitioner:
    """Partition dataset indices across devices using Dirichlet distribution.

    Each class's samples are split among devices with proportions drawn from
    Dir(alpha). With alpha=0.5, each device sees only 2-3 classes (non-IID).
    """

    def __init__(self, alpha: float = 0.5, seed: int = 42):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def assign(
        self,
        num_devices: int,
        labels: np.ndarray,
        num_classes: int,
    ) -> list[list[int]]:
        """Assign sample indices to devices.

        Args:
            num_devices: number of devices to partition among
            labels: array of class labels for all samples (shape: N,)
            num_classes: total number of classes

        Returns:
            list of length num_devices, each entry is a list of sample indices
        """
        idx_by_class = [
            np.where(labels == c)[0].tolist()
            for c in range(num_classes)
        ]

        device_indices: list[list[int]] = [[] for _ in range(num_devices)]

        for c in range(num_classes):
            class_samples = idx_by_class[c]
            if not class_samples:
                continue

            # Draw Dirichlet proportions for this class across devices
            proportions = self.rng.dirichlet(
                [self.alpha] * num_devices
            )

            # Assign each sample to a device based on proportions
            self.rng.shuffle(class_samples)
            splits = (np.cumsum(proportions) * len(class_samples)).astype(int)
            start = 0
            for d in range(num_devices):
                end = splits[d] if d < num_devices - 1 else len(class_samples)
                device_indices[d].extend(class_samples[start:end])
                start = end

        # Shuffle each device's indices
        for d in range(num_devices):
            self.rng.shuffle(device_indices[d])

        return device_indices
