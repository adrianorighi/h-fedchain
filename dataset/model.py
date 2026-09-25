import numpy as np
from typing import Optional


_WEIGHT_KEYS = ("W1", "b1", "W2", "b2")


def flatten_weights(weights: dict) -> np.ndarray:
    """Flatten an MLP weight dict into a 1D vector.

    Uses the canonical order W1, b1, W2, b2 — identical to the
    concatenation order in ``MLP.compute_gradient``.

    Args:
        weights: dict with keys W1, b1, W2, b2 (numpy arrays)

    Returns:
        1D numpy array (dtype follows the input arrays)
    """
    return np.concatenate([weights[key].ravel() for key in _WEIGHT_KEYS])


def unflatten_weights(flat: np.ndarray, like: dict) -> dict:
    """Split a flat vector back into an MLP weight dict.

    Shapes are taken from the reference dict ``like``; order is the
    canonical W1, b1, W2, b2 (same as ``MLP.compute_gradient``).

    Args:
        flat: 1D numpy array of concatenated parameters
        like: reference dict with keys W1, b1, W2, b2 (shapes/dtypes)

    Returns:
        dict with keys W1, b1, W2, b2, each reshaped to the
        corresponding shape in ``like`` (dtype preserved from ``flat``)

    Raises:
        ValueError: if ``flat.size`` differs from the total parameter
            size of ``like``
    """
    expected = sum(np.prod(like[key].shape) for key in _WEIGHT_KEYS)
    if flat.size != expected:
        raise ValueError(
            f"flat size {flat.size} does not match reference weight "
            f"size {int(expected)}"
        )
    result = {}
    offset = 0
    for key in _WEIGHT_KEYS:
        shape = like[key].shape
        size = int(np.prod(shape))
        result[key] = flat[offset:offset + size].reshape(shape)
        offset += size
    return result


class MLP:
    """2-layer MLP: input(12000) → hidden(64, ReLU) → output(5, softmax).

    Trainable via pure numpy forward/backward. Provides compute_gradient()
    for FL-style gradient transmission.
    """

    def __init__(
        self,
        input_dim: int = 12000,
        hidden_dim: int = 64,
        num_classes: int = 5,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, num_classes)) * 0.01
        self.b2 = np.zeros(num_classes)

    def get_weights(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_weights(self, weights: dict):
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass. X shape: (batch, input_dim). Returns (probs, cache)."""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
        return probs, cache

    def _cross_entropy_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss. y: (batch,) with integer class labels."""
        batch = probs.shape[0]
        log_probs = -np.log(probs[np.arange(batch), y] + 1e-15)
        return float(np.mean(log_probs))

    def backward(self, X: np.ndarray, y: np.ndarray, cache: dict) -> dict:
        """Backward pass. Returns gradients for all parameters."""
        batch = X.shape[0]
        probs = cache["probs"]
        a1 = cache["a1"]
        z1 = cache["z1"]

        dz2 = probs.copy()
        dz2[np.arange(batch), y] -= 1
        dz2 /= batch

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).astype(float)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray, global_weights: dict
    ) -> np.ndarray:
        """Compute gradient Δw_i = ∇L(w; D_i) for one device.

        Sets model to global_weights, runs forward+backward on the
        device's local data (X, y), and returns the gradient as a
        flat 1D vector (for FL transmission).

        Args:
            X: local data, shape (batch, 12000)
            y: local labels, shape (batch,)
            global_weights: dict with keys W1, b1, W2, b2

        Returns:
            flat_grad: 1D numpy array of length = total params
        """
        self.set_weights(global_weights)
        probs, cache = self.forward(X)
        grads = self.backward(X, y, cache)
        flat_grad = np.concatenate([
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel(),
        ])
        return flat_grad

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss on given data."""
        probs, _ = self.forward(X)
        return self._cross_entropy_loss(probs, y)
