"""Baseline functions extracted from LSTM-code.txt."""
import numpy as np

def naive_last_value_forecast(test_series: np.ndarray):
    preds = test_series[:-1]
    true = test_series[1:]
    return true, preds

def seasonal_naive_forecast(test_series: np.ndarray, season_length: int):
    preds, true = [], []
    for i in range(season_length, len(test_series)):
        preds.append(test_series[i - season_length])
        true.append(test_series[i])
    return np.array(true), np.array(preds)

def mean_forecast(train, horizon):
    """Training-mean baseline used in the original Untitled notebook."""
    return np.full(horizon, np.mean(train), dtype=float)
