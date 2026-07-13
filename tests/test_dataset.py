import pytest
from dataset.loader import PTBXLLoader


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
