"""
Anomaly detection over recent system metrics using Isolation Forest.
Retrains periodically on the rolling window kept by MetricsStore, then
scores the most recent sample against that model.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = ["cpu_percent", "memory_percent", "disk_percent"]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, min_samples: int = 30):
        self.contamination = contamination
        self.min_samples = min_samples
        self.model: IsolationForest | None = None

    def _to_matrix(self, samples: list[dict]) -> np.ndarray:
        return np.array([[s[f] for f in FEATURES] for s in samples], dtype=float)

    def fit(self, samples: list[dict]) -> bool:
        """Fit (or refit) the model. Returns False if there isn't enough data yet."""
        if len(samples) < self.min_samples:
            return False
        X = self._to_matrix(samples)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        self.model.fit(X)
        return True

    def score_latest(self, samples: list[dict]) -> dict:
        """
        Fit on history (excluding the latest point) and score the latest
        point against it. Returns a verdict plus a normalized anomaly score.
        """
        if len(samples) < self.min_samples + 1:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "reason": "not enough history yet to evaluate",
            }

        history, latest = samples[:-1], samples[-1]
        fitted = self.fit(history)
        if not fitted or self.model is None:
            return {"is_anomaly": False, "score": 0.0, "reason": "model not fitted"}

        x_latest = self._to_matrix([latest])
        raw_score = self.model.score_samples(x_latest)[0]  # higher = more normal
        prediction = self.model.predict(x_latest)[0]  # -1 = anomaly, 1 = normal

        # Normalize raw_score (~[-0.5, 0.5]) into a 0-1 "how anomalous" score
        normalized = float(np.clip(0.5 - raw_score, 0, 1))

        return {
            "is_anomaly": bool(prediction == -1),
            "score": round(normalized, 3),
            "metrics": {f: latest[f] for f in FEATURES},
            "reason": "isolation forest flagged this point as an outlier"
            if prediction == -1
            else "within normal range",
        }
