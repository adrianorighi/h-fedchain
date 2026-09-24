from monitoring.system_metrics import SystemMetrics


def test_snapshot_keys_and_range():
    sm = SystemMetrics()
    snap = sm.snapshot()
    assert set(snap) == {"cpu_percent", "memory_rss_bytes"}
    assert snap["cpu_percent"] >= 0.0
    assert snap["memory_rss_bytes"] > 0


def test_cpu_percent_returns_nonnegative_float():
    sm = SystemMetrics()
    first = sm.cpu_percent()
    second = sm.cpu_percent()
    for value in (first, second):
        assert isinstance(value, float)
        assert value >= 0.0
