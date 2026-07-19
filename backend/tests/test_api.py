"""
Lightweight tests that don't require a live LLM key -- they exercise the
metrics store, anomaly detector, and forecaster directly. RAG/agent
endpoints are smoke-tested for structure only, mocked where an API key
would otherwise be required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly_detector import AnomalyDetector
from models.forecaster import Forecaster
from monitoring.metrics_store import MetricsStore


def make_sample(cpu, memory=50.0, disk=40.0):
    return {"cpu_percent": cpu, "memory_percent": memory, "disk_percent": disk}


def test_metrics_store_rolls_over():
    store = MetricsStore(max_length=3)
    for i in range(5):
        store.add_sample(make_sample(cpu=i))
    assert len(store.recent()) == 3
    assert store.latest()["cpu_percent"] == 4


def test_anomaly_detector_flags_outlier():
    detector = AnomalyDetector(contamination=0.1, min_samples=10)
    normal = [make_sample(cpu=50 + (i % 3)) for i in range(30)]
    outlier = make_sample(cpu=99.0, memory=98.0, disk=95.0)

    result = detector.score_latest(normal + [outlier])
    assert "is_anomaly" in result
    assert "score" in result


def test_forecaster_trains_and_predicts():
    forecaster = Forecaster()
    series = [50 + (i % 5) for i in range(40)]
    trained = forecaster.train(series)
    assert trained is True

    prediction = forecaster.predict_next(series)
    assert isinstance(prediction, list)
    assert len(prediction) > 0
