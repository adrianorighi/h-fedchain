import psutil


class SystemMetrics:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._proc.cpu_percent(interval=None)

    def cpu_percent(self) -> float:
        return self._proc.cpu_percent(interval=None)

    def memory_rss_bytes(self) -> int:
        return int(self._proc.memory_info().rss)

    def snapshot(self) -> dict:
        return {
            "cpu_percent": self.cpu_percent(),
            "memory_rss_bytes": self.memory_rss_bytes(),
        }
