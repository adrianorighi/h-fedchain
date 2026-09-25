import numpy as np
import pytest

from dataset.model import MLP, flatten_weights, unflatten_weights

PARAM_KEYS = ("W1", "b1", "W2", "b2")


class TestFlattenWeights:
    def test_roundtrip_preserves_values_shapes_dtype(self):
        model = MLP(input_dim=120, hidden_dim=8, num_classes=3, seed=42)
        weights = model.get_weights()
        flat = flatten_weights(weights)
        restored = unflatten_weights(flat, like=weights)
        assert set(restored) == set(PARAM_KEYS)
        for key in PARAM_KEYS:
            assert restored[key].shape == weights[key].shape, key
            assert restored[key].dtype == weights[key].dtype, key
            np.testing.assert_allclose(restored[key], weights[key])

    def test_flatten_order_matches_compute_gradient_order(self):
        weights = {
            "W1": np.full((3, 2), 1.0),
            "b1": np.full(2, 2.0),
            "W2": np.full((2, 2), 3.0),
            "b2": np.full(2, 4.0),
        }
        expected = np.concatenate([
            weights["W1"].ravel(),
            weights["b1"].ravel(),
            weights["W2"].ravel(),
            weights["b2"].ravel(),
        ])
        flat = flatten_weights(weights)
        assert flat.dtype == expected.dtype
        np.testing.assert_array_equal(flat, expected)
        assert (flat == expected).all()

    def test_default_mlp_flat_length(self):
        weights = MLP(seed=42).get_weights()
        flat = flatten_weights(weights)
        assert flat.shape == (768389,)
        assert flat.size == 12000 * 64 + 64 + 64 * 5 + 5

    def test_unflatten_rejects_size_mismatch(self):
        weights = MLP(input_dim=12, hidden_dim=4, num_classes=2, seed=42).get_weights()
        expected = sum(v.size for v in weights.values())
        with pytest.raises(ValueError) as exc:
            unflatten_weights(np.zeros(expected + 1), like=weights)
        msg = str(exc.value)
        assert str(expected + 1) in msg
        assert str(expected) in msg

    def test_unflatten_preserves_float32_dtype(self):
        weights = MLP(seed=42).get_weights()
        flat = np.zeros(768389, dtype=np.float32)
        restored = unflatten_weights(flat, like=weights)
        for key in PARAM_KEYS:
            assert restored[key].dtype == np.float32, key
            assert restored[key].shape == weights[key].shape, key
            np.testing.assert_array_equal(
                restored[key], np.zeros(weights[key].shape, dtype=np.float32)
            )

    def test_unflatten_matches_compute_gradient_split(self):
        model = MLP(input_dim=120, hidden_dim=8, num_classes=3, seed=7)
        weights = model.get_weights()
        flat = flatten_weights(weights)
        restored = unflatten_weights(flat, like=weights)
        model.set_weights(restored)
        for key in PARAM_KEYS:
            np.testing.assert_array_equal(getattr(model, key), weights[key])
