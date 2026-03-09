"""Build the model training dataset.

This module combines solar irradiance targets from NREL NSRDB with daily
aggregated weather features from Open-Meteo, then applies feature engineering
and cleaning to produce the final training dataset.

"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from main.data_sources.nrel_nsrdb import fetch_nsrdb_half_hourly
from main.data_sources.open_meteo import fetch_historical_weather_daily
from main.features.cleaning import drop_na_rows
from main.features.engineering import engineer_features


def _aggregate_nsrdb_to_daily(nsrdb_half_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-hourly NSRDB records into a daily GHI target.

    Args:
        nsrdb_half_hourly: Half-hourly NSRDB DataFrame containing Year/Month/Day
            columns and a GHI column.

    Returns:
        A DataFrame with columns:
          - date: Date string (YYYY-MM-DD).
          - GHI: Daily summed GHI value.
    """

    df = nsrdb_half_hourly.copy()
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    daily = df.groupby("date").agg(GHI=("GHI", "sum")).reset_index()
    daily["date"] = daily["date"].astype(str)
    return daily


def build_training_dataset(
    *,
    latitude: float,
    longitude: float,
    years: list[int],
    nrel_api_key: str,
    nrel_email: str,
    open_meteo_start: date,
    open_meteo_end: date,
    cache_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Build a feature-engineered training dataset with target column "GHI".

    This function optionally caches the raw NSRDB half-hourly CSV response,
    aggregates solar data to daily targets, fetches daily weather aggregates,
    merges solar and weather by date, and applies feature engineering and
    row-level cleaning.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        years: List of NSRDB years to fetch.
        nrel_api_key: NREL developer API key.
        nrel_email: Email address required by the NSRDB API.
        open_meteo_start: Start date (inclusive) for historical weather
            features.
        open_meteo_end: End date (inclusive) for historical weather features.
        cache_csv_path: Optional path to cache the NSRDB half-hourly CSV data.

    Returns:
        A DataFrame containing engineered features and the target column "GHI".

    Raises:
        RuntimeError: If the underlying data sources fail.
    """

    if cache_csv_path and cache_csv_path.exists():
        nsrdb_half_hourly = pd.read_csv(cache_csv_path)
    else:
        nsrdb_half_hourly = fetch_nsrdb_half_hourly(
            latitude=latitude,
            longitude=longitude,
            years=years,
            api_key=nrel_api_key,
            email=nrel_email,
        )
        if cache_csv_path:
            nsrdb_half_hourly.to_csv(cache_csv_path, index=False)

    solar_daily = _aggregate_nsrdb_to_daily(nsrdb_half_hourly)

    weather_daily = fetch_historical_weather_daily(
        latitude=latitude,
        longitude=longitude,
        start_date=open_meteo_start,
        end_date=open_meteo_end,
    )

    merged = solar_daily.merge(weather_daily, on="date", how="inner")
    merged = engineer_features(merged)
    merged = drop_na_rows(merged)
    return merged
