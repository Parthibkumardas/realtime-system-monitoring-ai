"""
Samples live system metrics using psutil. Runs as a background task
started at app startup (see main.py) and writes each sample into the
shared MetricsStore.
"""
import asyncio
import time
from datetime import datetime, timezone

import psutil

from monitoring.metrics_store import MetricsStore


class MetricCollector:
    def __init__(self, store: MetricsStore, interval_seconds: int = 5):
        self.store = store
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    def sample_once(self) -> dict:
        """Take a single snapshot of system metrics."""
        net = psutil.net_io_counters()
        sample = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "net_sent_mb": round(net.bytes_sent / (1024 * 1024), 2),
            "net_recv_mb": round(net.bytes_recv / (1024 * 1024), 2),
        }
        return sample

    async def _run(self):
        self._running = True
        while self._running:
            sample = self.sample_once()
            self.store.add_sample(sample)
            await asyncio.sleep(self.interval_seconds)

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
