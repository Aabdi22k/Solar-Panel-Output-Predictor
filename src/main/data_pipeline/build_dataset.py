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

    The NSRDB half-hourly GHI values should be converted into daily energy
    totals. For 30-minute intervals, each row contributes 0.5 hours.

    Returns:
        DataFrame with:
          - date: YYYY-MM-DD string
          - GHI: daily total in Wh/m²
    """
    df = nsrdb_half_hourly.copy()

    required_cols = {"Year", "Month", "Day", "GHI"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"NSRDB data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    df = df.dropna(subset=["date", "GHI"]).copy()

    # 30-minute interval => convert W/m²-style interval values into Wh/m²
    df["ghi_wh_m2"] = pd.to_numeric(df["GHI"], errors="coerce") * 0.5
    df = df.dropna(subset=["ghi_wh_m2"]).copy()

    daily = (
        df.groupby("date", as_index=False)
        .agg(GHI=("ghi_wh_m2", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
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
    timezone: str = "auto",
) -> pd.DataFrame:
    """Build a feature-engineered training dataset with target column "GHI"."""

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
        timezone=timezone,
    )

    merged = solar_daily.merge(weather_daily, on="date", how="inner")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    merged = engineer_features(merged)
    merged = drop_na_rows(merged)

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return merged
