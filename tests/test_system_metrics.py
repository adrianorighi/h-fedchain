from monitoring.system_metrics import SystemMetrics


def test_snapshot_keys_and_range():
    sm = SystemMetrics()
    sm.cpu_percent()  # arm first psutil sample
    snap = sm.snapshot()
    assert set(snap) == {"cpu_percent", "memory_rss_bytes"}
    assert 0.0 <= snap["cpu_percent"] <= 100.0
    assert snap["memory_rss_bytes"] > 0


def test_cpu_percent_callable_twice():
    sm = SystemMetrics()
    sm.cpu_percent()
    first = sm.cpu_percent()
    assert isinstance(first, float)
    assert 0.0 <= first <= 100.0
