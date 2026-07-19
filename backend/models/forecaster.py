"""
A small LSTM that forecasts a metric N steps into the future given the
last WINDOW samples. Trains on-the-fly from whatever history is currently
in the MetricsStore -- fine for a demo/portfolio project; for production
you'd train offline on a much larger history and load fixed weights here.
"""
import numpy as np
import torch
from torch import nn

WINDOW = 12       # look back this many samples
HORIZON = 6        # predict this many samples ahead
HIDDEN_SIZE = 32
EPOCHS = 60


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        # x: (batch, WINDOW, 1)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)  # (batch, HORIZON)


def _make_windows(series: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.array(X), np.array(y)


class Forecaster:
    """Wraps training + inference for one metric's forecast."""

    def __init__(self):
        self.model = LSTMForecaster()
        self.mean = 0.0
        self.std = 1.0
        self.trained = False

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std + 1e-8)

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def train(self, series: list[float]) -> bool:
        series = np.array(series, dtype=float)
        if len(series) < WINDOW + HORIZON + 5:
            return False

        self.mean, self.std = series.mean(), series.std()
        norm = self._normalize(series)

        X, y = _make_windows(norm, WINDOW, HORIZON)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, WINDOW, 1)
        y = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        self.trained = True
        return True

    def predict_next(self, series: list[float]) -> list[float]:
        """Given the most recent WINDOW points, predict the next HORIZON points."""
        if not self.trained or len(series) < WINDOW:
            return []

        recent = np.array(series[-WINDOW:], dtype=float)
        norm = self._normalize(recent)
        x = torch.tensor(norm, dtype=torch.float32).view(1, WINDOW, 1)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).numpy().flatten()

        pred = self._denormalize(pred_norm)
        return [round(float(v), 2) for v in pred]
