import numpy as np
import pytest
from dataset.loader import PTBXLLoader
from dataset.model import MLP
from dataset.partitioner import DirichletPartitioner


class TestPTBXLLoader:
    def test_loader_initializes(self):
        loader = PTBXLLoader(max_records=10)
        assert loader is not None

    def test_generate_synthetic_returns_correct_shape(self):
        loader = PTBXLLoader()
        data, labels = loader.generate_synthetic(50)
        assert data.shape == (50, 12, 1000), f"Expected (50,12,1000), got {data.shape}"
        assert labels.shape == (50,), f"Expected (50,), got {labels.shape}"

    def test_generate_synthetic_label_range(self):
        loader = PTBXLLoader()
        _, labels = loader.generate_synthetic(200)
        assert labels.min() >= 0
        assert labels.max() <= 4

    def test_load_raises_if_not_downloaded(self):
        loader = PTBXLLoader(cache_dir="/tmp/nonexistent_ptbxl_xyz")
        loader.max_records = 5
        with pytest.raises(FileNotFoundError, match="PTB-XL not found"):
            loader.load()


class TestDirichletPartitioner:
    def test_partition_returns_correct_number_of_devices(self):
        partitioner = DirichletPartitioner(alpha=0.5, seed=42)
        labels = np.random.randint(0, 5, size=1000)
        result = partitioner.assign(num_devices=10, labels=labels, num_classes=5)
        assert len(result) == 10

    def test_partition_covers_all_samples(self):
        partitioner = DirichletPartitioner(alpha=0.5, seed=42)
        labels = np.random.randint(0, 5, size=1000)
        result = partitioner.assign(num_devices=10, labels=labels, num_classes=5)
        total = sum(len(d) for d in result)
        assert total == 1000

    def test_each_device_gets_at_least_one_sample(self):
        partitioner = DirichletPartitioner(alpha=1.0, seed=42)
        labels = np.random.randint(0, 3, size=500)
        result = partitioner.assign(num_devices=5, labels=labels, num_classes=3)
        for d in result:
            assert len(d) > 0, f"Device has no samples"

    def test_partition_is_non_iid(self):
        """With alpha=0.1, devices should see very few classes."""
        partitioner = DirichletPartitioner(alpha=0.1, seed=42)
        labels = np.random.randint(0, 10, size=5000)
        result = partitioner.assign(num_devices=20, labels=labels, num_classes=10)
        classes_per_device = [
            len(set(labels[idx] for idx in device))
            for device in result
        ]
        avg_classes = np.mean(classes_per_device)
        # With alpha=0.1, each device should see much fewer than 10 classes
        assert avg_classes < 6, f"Expected highly non-IID, got avg {avg_classes:.1f} classes/device"

    def test_alpha_must_be_positive(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            DirichletPartitioner(alpha=0.0)


class TestMLP:
    def test_forward_shape(self):
        model = MLP(input_dim=120, hidden_dim=8, num_classes=3, seed=42)
        X = np.random.randn(10, 120)
        probs, cache = model.forward(X)
        assert probs.shape == (10, 3)
        np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(10))

    def test_backward_shape(self):
        model = MLP(input_dim=120, hidden_dim=8, num_classes=3, seed=42)
        X = np.random.randn(10, 120)
        y = np.random.randint(0, 3, size=10)
        probs, cache = model.forward(X)
        grads = model.backward(X, y, cache)
        assert grads["W1"].shape == model.W1.shape
        assert grads["b1"].shape == model.b1.shape
        assert grads["W2"].shape == model.W2.shape
        assert grads["b2"].shape == model.b2.shape

    def test_compute_gradient_flat_shape(self):
        model = MLP(input_dim=120, hidden_dim=8, num_classes=3, seed=42)
        X = np.random.randn(5, 120)
        y = np.random.randint(0, 3, size=5)
        global_weights = model.get_weights()
        flat_grad = model.compute_gradient(X, y, global_weights)
        expected = 120 * 8 + 8 + 8 * 3 + 3
        assert flat_grad.shape == (expected,)

    def test_get_set_weights_roundtrip(self):
        model = MLP(input_dim=12, hidden_dim=4, num_classes=2, seed=42)
        w = model.get_weights()
        model2 = MLP(input_dim=12, hidden_dim=4, num_classes=2, seed=99)
        model2.set_weights(w)
        for key in w:
            np.testing.assert_array_equal(model2.get_weights()[key], w[key])

    def test_loss_decreases_with_training(self):
        model = MLP(input_dim=12, hidden_dim=4, num_classes=2, seed=42)
        X = np.random.randn(20, 12)
        y = np.random.randint(0, 2, size=20)
        global_w = model.get_weights()
        loss_before = model.loss(X, y)
        grad = model.compute_gradient(X, y, global_w)
        lr = 0.1
        flat_w = np.concatenate([v.ravel() for v in global_w.values()])
        flat_w -= lr * grad
        new_w = {}
        start = 0
        shapes = [(12, 4), (4,), (4, 2), (2,)]
        keys = ["W1", "b1", "W2", "b2"]
        for key, shape in zip(keys, shapes):
            size = np.prod(shape)
            new_w[key] = flat_w[start:start + size].reshape(shape)
            start += size
        model.set_weights(new_w)
        loss_after = model.loss(X, y)
        assert loss_after < loss_before, f"Loss increased: {loss_before:.4f} -> {loss_after:.4f}"
