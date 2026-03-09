"""Feature engineering utilities.

This module builds derived features used by the model. It adds time-based
seasonality features and a set of interaction and ratio features from weather
columns when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add seasonality features derived from a date column.

    Adds:
      - day_of_year
      - sin_day_of_year
      - cos_day_of_year

    Args:
        df: Input DataFrame containing a date column.
        date_col: Name of the date column to parse.

    Returns:
        A copy of the input DataFrame with seasonality features added.
    """
    out = df.copy()
    dt_series = pd.to_datetime(out[date_col], errors="coerce")
    out[date_col] = dt_series

    dt_index = pd.DatetimeIndex(dt_series)
    day_of_year = dt_index.dayofyear.astype(float)

    out["day_of_year"] = day_of_year
    out["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.0)
    out["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.0)
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and ratio features from existing weather columns.

    Features are only added when their required columns exist. This keeps the
    pipeline resilient to missing fields in upstream data sources.

    Possible added columns include:
      - effective_sunshine
      - sunshine_ratio
      - temp_range
      - cloud_adjusted_temp_range
      - tavg_diff
      - tavg_ewm_7
      - wdir_x_wspd

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the input DataFrame with interaction features added.
    """
    out = df.copy()

    if "sunshine_duration" in out.columns and "cloud_cover" in out.columns:
        out["effective_sunshine"] = out["sunshine_duration"] * (
            1 - out["cloud_cover"]
        )

    if (
        "sunshine_duration" in out.columns
        and "daylight_duration" in out.columns
    ):
        out["sunshine_ratio"] = out["sunshine_duration"] / (
            out["daylight_duration"] + 1e-5
        )

    if "tmax" in out.columns and "tmin" in out.columns:
        out["temp_range"] = out["tmax"] - out["tmin"]

    if "temp_range" in out.columns and "cloud_cover" in out.columns:
        out["cloud_adjusted_temp_range"] = out["temp_range"] * (
            1 - out["cloud_cover"]
        )

    if "tavg" in out.columns:
        out["tavg_diff"] = out["tavg"].diff().fillna(0)
        out["tavg_ewm_7"] = (
            out["tavg"]
            .ewm(
                span=7,
                adjust=False,
            )
            .mean()
        )

    if "wdir" in out.columns and "wspd" in out.columns:
        out["wdir_x_wspd"] = out["wdir"] * out["wspd"]

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with time and interaction features added.
    """
    out = add_time_features(df)
    out = add_interaction_features(out)
    return out
