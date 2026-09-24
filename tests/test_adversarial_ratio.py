from experiments.analyze_results import S3_RATIOS as ANALYZE_S3
from experiments.batch_runner import S3_RATIOS as BATCH_S3
from experiments.scenario_3_adversarial import QUICK_RATIOS, RATIOS

EXPECTED = [0.0, 0.10, 0.20, 0.30, 0.40]


def test_s3_ratios_batch():
    assert BATCH_S3 == EXPECTED


def test_s3_ratios_analyze():
    assert ANALYZE_S3 == EXPECTED


def test_s3_ratios_scenario():
    assert RATIOS == EXPECTED
    assert 0.05 not in RATIOS
    assert QUICK_RATIOS == [0.0, 0.20]


import asyncio

from simulator.orchestrator import Orchestrator, select_adversarial_ids


def test_select_adversarial_ids():
    a = select_adversarial_ids(200, 0.20, 42)
    b = select_adversarial_ids(200, 0.20, 42)
    c = select_adversarial_ids(200, 0.20, 7)
    assert a == b
    assert a != c
    assert len(a) == 40
    assert select_adversarial_ids(200, 0.0, 42) == set()
    assert select_adversarial_ids(200, 1.0, 42) == set(range(200))


async def _run(ratio: float):
    orch = Orchestrator(
        num_clusters=4, nodes_per_cluster=5,
        devices_per_cluster=50, f=1, latency_ms=10.0,
        variant="no_zkp", adversarial_ratio=ratio,
        snark_prove=False,
    )
    return await orch.run_experiment(num_rounds=1)


def test_adv_count_is_ratio_of_total():
    for ratio in (0.0, 0.10, 0.20, 0.30, 0.40):
        result = asyncio.run(_run(ratio))
        m = result.round_metrics[0]
        expected = int(200 * ratio)
        assert m["num_adversarial"] == expected
        assert m["num_honest"] == 200 - expected
        # rejected cannot exceed ground truth
        assert m["rejected_adversarial"] <= expected or expected == 0
