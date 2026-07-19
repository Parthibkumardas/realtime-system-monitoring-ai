"""
A simple in-memory rolling buffer for metric samples. Swap this out for
InfluxDB/TimescaleDB/Prometheus remote-write in a production deployment --
the interface (add_sample / recent / as_series) is intentionally small so
that swap doesn't ripple through the rest of the codebase.
"""
from collections import deque
from threading import Lock


class MetricsStore:
    def __init__(self, max_length: int = 720):
        self.max_length = max_length
        self._buffer: deque[dict] = deque(maxlen=max_length)
        self._lock = Lock()

    def add_sample(self, sample: dict) -> None:
        with self._lock:
            self._buffer.append(sample)

    def recent(self, n: int | None = None) -> list[dict]:
        with self._lock:
            data = list(self._buffer)
        return data[-n:] if n else data

    def as_series(self, field: str, n: int | None = None) -> list[float]:
        """Return just the numeric values for one field, e.g. 'cpu_percent'."""
        samples = self.recent(n)
        return [s[field] for s in samples if field in s]

    def latest(self) -> dict | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
