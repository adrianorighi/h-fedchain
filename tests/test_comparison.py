"""Test the comparison runner runs end-to-end with minimal config."""
import pytest
from experiments.comparison import compare


@pytest.mark.asyncio
async def test_comparison_nominal_runs():
    """Comparison with tiny config should not crash."""
    result = await compare("nominal")
    assert result is None
