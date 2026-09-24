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
