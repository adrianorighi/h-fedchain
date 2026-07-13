import numpy as np
import pytest
from core.multikrum.aggregator import MultiKrum


class TestMultiKrum:
    def test_select_returns_m_indices(self):
        n, f = 10, 3
        m = n - f
        np.random.seed(42)
        gradients = [np.random.randn(5) for _ in range(n)]
        krum = MultiKrum()
        selected = krum.select(gradients, f)
        assert len(selected) == m
        assert all(0 <= i < n for i in selected)

    def test_select_excludes_outlier(self):
        n, f = 6, 1
        gradients = [np.zeros(2) for _ in range(n - 1)]
        gradients.append(np.array([100.0, 100.0]))
        krum = MultiKrum()
        selected = krum.select(gradients, f)
        assert 5 not in selected, "Outlier at index 5 was selected"

    def test_select_raises_on_invalid_f(self):
        n, f = 5, 3
        gradients = [np.random.randn(3) for _ in range(n)]
        krum = MultiKrum()
        with pytest.raises(ValueError, match="f must be < n/3"):
            krum.select(gradients, f)

    def test_select_empty_returns_empty(self):
        krum = MultiKrum()
        selected = krum.select([], 0)
        assert selected == []

    def test_m_equals_one(self):
        n, f = 5, 4  # m = 1
        np.random.seed(42)
        gradients = [np.random.randn(3) for _ in range(n)]
        krum = MultiKrum()
        selected = krum.select(gradients, f)
        assert len(selected) == n - f
        assert len(selected) == 1
