from pathlib import Path
import pandas as pd
from src.preprocessing.cleaning_pipeline import clean_time_series

ROOT = Path(__file__).resolve().parents[1]


def test_project_datasets():
    for name in ("inflation.csv", "australia_beer.csv"):
        result = clean_time_series(str(ROOT / "data" / name), date_col="ds", value_col="y", missing_method="drop")
        assert not result.empty
        assert result.index.is_monotonic_increasing
        assert not result.index.has_duplicates
        assert result.target.notna().all()


def test_cleaning_unsorted_duplicates_and_missing(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame({"ds": ["2020-01-03", "2020-01-01", "2020-01-01", "2020-01-02"],
                  "y": [5, 1, 3, None]}).to_csv(path, index=False)
    result = clean_time_series(str(path), date_col="ds", value_col="y")
    assert result.target.tolist() == [2.0, 3.5, 5.0]
