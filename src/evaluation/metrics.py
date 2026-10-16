"""Metrics extracted from LSTM-code.txt; original formulas retained."""
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-8):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0

def rolling_mape(y_true_series: np.ndarray, y_pred_series: np.ndarray, window: int):
    assert len(y_true_series) == len(y_pred_series)
    rolling = []
    for i in range(len(y_true_series)):
        start = max(0, i - window + 1)
        mape = mean_absolute_percentage_error(
            y_true_series[start:i+1], y_pred_series[start:i+1]
        )
        rolling.append(mape)
    return np.array(rolling)
