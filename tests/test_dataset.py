import numpy as np
import pytest
from dataset.loader import PTBXLLoader
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
