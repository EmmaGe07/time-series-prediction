from pathlib import Path

import pandas as pd


DATE_COLUMN_HINTS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "ds",
    "month",
    "year",
    "day",
    "week",
    "quarter",
    "period",
    "observation_date",
)
MISSING_METHODS = {"drop", "ffill", "bfill", "interpolate"}
DUPLICATE_METHODS = {"mean", "first", "last", "drop", "keep"}


def _guess_date_col(df: pd.DataFrame) -> str:
    """Guess the date column from names first, then from parse success rate."""
    normalized_columns = {str(col).strip().lower(): col for col in df.columns}

    # Prefer exact common names such as date, ds, month, and timestamp.
    for hint in DATE_COLUMN_HINTS:
        if hint in normalized_columns:
            return normalized_columns[hint]

    # Then allow labels like sales_date, invoice_time, or report_month.
    for normalized_col, original_col in normalized_columns.items():
        if any(hint in normalized_col for hint in DATE_COLUMN_HINTS):
            return original_col

    # Finally, inspect the actual values and pick the column that parses best.
    best_col = df.columns[0]
    best_score = 0.0
    sample_size = min(len(df), 1000)

    for col in df.columns:
        score = _date_parse_score(df[col], sample_size)
        if score > best_score:
            best_col = col
            best_score = score

    return best_col


def _date_parse_score(series: pd.Series, sample_size: int) -> float:
    """Estimate how likely a column is to contain dates."""
    sample = series.head(sample_size).dropna()
    if sample.empty:
        return 0.0

    # Plain numeric target columns can be misread as nanosecond timestamps.
    # Only numeric values that look like yyyymmdd are considered date-like here.
    if pd.api.types.is_numeric_dtype(sample):
        as_text = sample.astype("Int64", errors="ignore").astype(str).str.strip()
        looks_like_yyyymmdd = as_text.str.fullmatch(r"\d{8}").mean()
        if looks_like_yyyymmdd < 0.8:
            return 0.0

    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean()


def _guess_value_col(df: pd.DataFrame, date_col: str) -> str:
    """Use the first non-date column as the target value column."""
    for col in df.columns:
        if col != date_col:
            return col

    raise ValueError("Could not find a value column: CSV only contains a date column.")


def clean_time_series(
    path: str,
    date_col: str | None = None,
    value_col: str | None = None,
    freq: str | None = None,
    missing_method: str = "interpolate",
    target_name: str = "target",
    duplicate_method: str = "mean",
) -> pd.DataFrame:
    """
    Clean a raw time-series CSV into a standard forecasting-ready DataFrame.

    The returned DataFrame has a DatetimeIndex sorted in ascending order and a
    single numeric target column named by ``target_name``.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    if missing_method not in MISSING_METHODS:
        allowed = ", ".join(sorted(MISSING_METHODS))
        raise ValueError(f"missing_method must be one of: {allowed}")

    if duplicate_method not in DUPLICATE_METHODS:
        allowed = ", ".join(sorted(DUPLICATE_METHODS))
        raise ValueError(f"duplicate_method must be one of: {allowed}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    selected_date_col = date_col if date_col is not None else _guess_date_col(df)
    if selected_date_col not in df.columns:
        raise ValueError(f"Date column not found: {selected_date_col}")

    # Convert dates early and remove rows that cannot be placed on the timeline.
    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors="coerce")
    df = df.dropna(subset=[selected_date_col]).sort_values(selected_date_col)

    selected_value_col = value_col if value_col is not None else _guess_value_col(df, selected_date_col)
    if selected_value_col not in df.columns:
        raise ValueError(f"Value column not found: {selected_value_col}")

    # Keep only the time axis and selected value, then coerce the value to numeric.
    df = df[[selected_date_col, selected_value_col]].copy()
    df[selected_value_col] = pd.to_numeric(df[selected_value_col], errors="coerce")
    df = df.set_index(selected_date_col)

    # Duplicate timestamps are configurable. Some models need one row per date,
    # while exploratory workflows may prefer to keep repeated timestamps.
    if duplicate_method == "mean":
        df = df.groupby(df.index).mean()
    elif duplicate_method == "first":
        df = df.groupby(df.index).first()
    elif duplicate_method == "last":
        df = df.groupby(df.index).last()
    elif duplicate_method == "drop":
        df = df[~df.index.duplicated(keep=False)]
    elif duplicate_method == "keep":
        df = df.copy()

    # Optionally enforce a regular frequency, which may introduce missing rows.
    if freq is not None:
        if df.index.has_duplicates:
            raise ValueError(
                "Cannot apply freq when duplicate timestamps are kept. "
                "Use duplicate_method='mean', 'first', 'last', or 'drop'."
            )
        df = df.asfreq(freq)

    if missing_method == "drop":
        df = df.dropna()
    elif missing_method == "ffill":
        df = df.ffill().dropna()
    elif missing_method == "bfill":
        df = df.bfill().dropna()
    elif missing_method == "interpolate":
        df = df.interpolate(method="time").dropna()

    df = df.rename(columns={selected_value_col: target_name})

    if df.empty:
        raise ValueError("Cleaned DataFrame is empty after cleaning.")

    return df
