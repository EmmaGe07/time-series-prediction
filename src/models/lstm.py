"""Original LSTM implementation, reorganized only; training needs review (see README)."""
from src.evaluation.metrics import mean_absolute_percentage_error, rolling_mape
from src.models.baseline import naive_last_value_forecast, seasonal_naive_forecast

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader

DEFAULT_PAST_STEPS = 24

DEFAULT_HORIZON = 1

DEFAULT_BATCH_SIZE = 32

DEFAULT_HIDDEN_SIZE = 16

DEFAULT_NUM_LAYERS = 1

DEFAULT_DROPOUT = 0.2

DEFAULT_LR = 0.0001

DEFAULT_EPOCHS = 300

DEFAULT_ROLLING_WINDOW = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """
    Builds sliding windows from multivariate time series data.
    features: np.ndarray of shape (T, num_features)
    targets: np.ndarray of shape (T,)
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, input_len: int, horizon: int):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.x, self.y = self._build_windows()

    def _build_windows(self):
        x_list, y_list = [], []
        total_len = len(self.targets)
        for end in range(self.input_len, total_len - self.horizon + 1):
            start = end - self.input_len
            x = self.features[start:end]         # (seq_len, num_features)
            y = self.targets[end:end + self.horizon]  # (horizon,)
            x_list.append(x)
            y_list.append(y)
        if not x_list:
            raise ValueError(
                f"Not enough data to build windows: series length {total_len}, input_len {self.input_len}"
            )
        
        return np.stack(x_list), np.stack(y_list)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        output_size: int = DEFAULT_HORIZON,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x: (batch, seq_len, num_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden)
        return self.fc(last)

class TwinTowerLSTM(nn.Module):
    """
    Twin-tower: time-domain LSTM + frequency-domain LSTM, fused before final prediction.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        output_size: int = DEFAULT_HORIZON,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.time_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.freq_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):  # x: (batch, seq_len, num_features)
        # time-domain tower
        out_time, _ = self.time_lstm(x)
        last_time = out_time[:, -1, :]

        # frequency-domain tower: apply FFT to the first feature (assumes target)
        x_first = x[:, :, 0]                 # (batch, seq_len)
        fft = torch.fft.rfft(x_first, dim=1) # (batch, freq_bins)
        mag = torch.abs(fft)
        mag = torch.log1p(mag)
        mag_seq = mag.unsqueeze(-1)          # (batch, freq_bins, 1)
        out_freq, _ = self.freq_lstm(mag_seq)
        last_freq = out_freq[:, -1, :]

        combined = torch.cat([last_time, last_freq], dim=1)
        return self.fc(combined)

def prepare_series(
    data: pd.Series | pd.DataFrame,
    test_size: int,
    past_steps: int
) -> tuple[TimeSeriesDataset, TimeSeriesDataset, MinMaxScaler, np.ndarray]:
    # Normalize data container
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "target")
    else:
        df = data.copy()

    if df.shape[1] < 1:
        raise ValueError("Input data must have at least one column")

    target_col = df.columns[0]
    # Split into train/test with context for test
    train_df = df.iloc[: -test_size]
    test_df = df.iloc[-(test_size + past_steps):]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    # Features vs. target arrays
    X_train = train_scaled
    y_train = train_scaled[:, df.columns.get_loc(target_col)]
    X_test = test_scaled
    y_test = test_scaled[past_steps:, df.columns.get_loc(target_col)]

    train_ds = TimeSeriesDataset(X_train, y_train, past_steps, DEFAULT_HORIZON)
    test_ds = TimeSeriesDataset(X_test, y_test, past_steps, DEFAULT_HORIZON)

    raw_test = df[target_col].iloc[-test_size:].values.astype(np.float32)
    return train_ds, test_ds, scaler, raw_test

def train_model(model: nn.Module, dataloader: DataLoader, epochs: int, lr: float):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred.squeeze(1), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if epoch == 1 or epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:03d}/{epochs} - TRAIN MSE: {avg_loss:.6f}")
    return model

def evaluate(
    model: nn.Module,
    test_ds: TimeSeriesDataset,
    scaler: MinMaxScaler,
    raw_test_series: np.ndarray,
    rolling_window: int
):
    model.eval()
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy().flatten()
            preds.append(out[0])
            trues.append(yb.numpy().flatten()[0])
    trues_unscaled = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    preds_unscaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    mse = np.mean((trues_unscaled - preds_unscaled) ** 2)
    mean_true = np.mean(trues_unscaled)
    mape = mean_absolute_percentage_error(trues_unscaled, preds_unscaled)
    rolling = rolling_mape(trues_unscaled, preds_unscaled, rolling_window)
    true_nv, pred_nv = naive_last_value_forecast(raw_test_series)
    mae = np.mean(np.abs(trues_unscaled - preds_unscaled))
    cvrmse = mse / mean_true if mean_true != 0 else float('nan')
    nmae = mae/mean_true
    results = {
        "mse": mse,
        "mape": mape,
        "rolling_mape_last": rolling[-1],
        "rolling_mape_mean": float(np.mean(rolling)),
        "naive_mape": mean_absolute_percentage_error(true_nv, pred_nv),
        "nmae": nmae,
        "mae": mae,
        "cvrmse":cvrmse
    }
    if len(raw_test_series) > 12:
        true_sn, pred_sn = seasonal_naive_forecast(raw_test_series, season_length=12)
        results["seasonal_naive_mape"] = mean_absolute_percentage_error(true_sn, pred_sn)
    return results, trues_unscaled, preds_unscaled, rolling

def run_forecast_on_series(
    data: pd.Series | pd.DataFrame,
    test_size: int = 36,
    past_steps: int = DEFAULT_PAST_STEPS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    model_type: str = 'vanilla'
):
    """
    Full forecasting pipeline: build datasets, train, and evaluate.
    """
    train_ds, test_ds, scaler, raw_test = prepare_series(data, test_size, past_steps)
    # infer feature dimension
    n_features = train_ds.x.shape[-1]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    if model_type == 'vanilla':
        model = SimpleLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=DEFAULT_HORIZON,
            dropout=dropout
        )
    elif model_type == 'twin':
        model = TwinTowerLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=DEFAULT_HORIZON,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}, expected 'vanilla' or 'twin'")

    model = train_model(model, train_loader, epochs, lr)
    return evaluate(model, test_ds, scaler, raw_test, rolling_window)
