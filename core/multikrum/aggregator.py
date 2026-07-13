import numpy as np


class MultiKrum:
    def select(
        self,
        gradients: list[np.ndarray],
        f: int,
    ) -> list[int]:
        n = len(gradients)
        if n == 0:
            return []
        if f >= n / 3:
            raise ValueError(
                f"f must be < n/3, got f={f}, n={n}"
            )
        m = n - f
        if n < 2:
            return [0]

        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(gradients[i] - gradients[j]))
                dist[i, j] = d
                dist[j, i] = d

        scores = np.sum(np.partition(dist, m - 1)[:, :m - 1], axis=1)
        selected = np.argsort(scores)[:m].tolist()
        return sorted(selected)
