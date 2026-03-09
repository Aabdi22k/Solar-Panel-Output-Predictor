"""Build forecast-time features used for future predictions.

This module fetches forward-looking weather forecasts from Open-Meteo,
applies feature engineering and cleaning, and caches the resulting daily
feature table to disk for reuse.

"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from main.data_sources.open_meteo import fetch_forecast_weather_daily
from main.features.cleaning import drop_na_rows
from main.features.engineering import engineer_features


def build_forecast_features(
    *,
    latitude: float,
    longitude: float,
    days: int,
    forecasts_dir: Path,
) -> pd.DataFrame:
    """Build and cache daily forecast features for the next N days.

    Forecast dates start tomorrow and span `days` consecutive days. If a cached
    file exists for the same location and date range, it is loaded instead of
    fetching new data.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        days: Number of days to forecast (must be >= 1).
        forecasts_dir: Directory to store and load cached forecast CSVs.

    Returns:
        A DataFrame containing engineered daily forecast features.

    Raises:
        RuntimeError: If Open-Meteo forecast retrieval fails.
    """
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=days - 1)

    filename = f"forecast_{latitude}_{longitude}_{start}_to_{end}.csv"
    path = forecasts_dir / filename

    if path.exists():
        return pd.read_csv(path)

    df = fetch_forecast_weather_daily(
        latitude=latitude,
        longitude=longitude,
        start_date=start,
        end_date=end,
    )
    df = engineer_features(df)
    df = drop_na_rows(df)

    df.to_csv(path, index=False)
    return df
