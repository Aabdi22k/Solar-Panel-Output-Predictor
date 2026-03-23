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
    """Add seasonality features derived from a date column."""
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
    """Add interaction and ratio features from existing weather columns."""
    out = df.copy()

    if "cloud_cover" in out.columns:
        out["cloud_cover"] = pd.to_numeric(out["cloud_cover"], errors="coerce")
        out["cloud_cover_frac"] = out["cloud_cover"] / 100.0
        out["cloud_cover_frac"] = out["cloud_cover_frac"].clip(lower=0.0, upper=1.0)

    if "sunshine_duration" in out.columns:
        out["sunshine_duration"] = pd.to_numeric(
            out["sunshine_duration"], errors="coerce"
        )

    if "daylight_duration" in out.columns:
        out["daylight_duration"] = pd.to_numeric(
            out["daylight_duration"], errors="coerce"
        )

    if "tavg" in out.columns:
        out["tavg"] = pd.to_numeric(out["tavg"], errors="coerce")

    if "tmin" in out.columns:
        out["tmin"] = pd.to_numeric(out["tmin"], errors="coerce")

    if "tmax" in out.columns:
        out["tmax"] = pd.to_numeric(out["tmax"], errors="coerce")

    if "wdir" in out.columns:
        out["wdir"] = pd.to_numeric(out["wdir"], errors="coerce")

    if "wspd" in out.columns:
        out["wspd"] = pd.to_numeric(out["wspd"], errors="coerce")

    if "sunshine_duration" in out.columns and "cloud_cover_frac" in out.columns:
        out["effective_sunshine"] = out["sunshine_duration"] * (
            1.0 - out["cloud_cover_frac"]
        )

    if "sunshine_duration" in out.columns and "daylight_duration" in out.columns:
        out["sunshine_ratio"] = out["sunshine_duration"] / (
            out["daylight_duration"] + 1e-5
        )

    if "tmax" in out.columns and "tmin" in out.columns:
        out["temp_range"] = out["tmax"] - out["tmin"]

    if "temp_range" in out.columns and "cloud_cover_frac" in out.columns:
        out["cloud_adjusted_temp_range"] = out["temp_range"] * (
            1.0 - out["cloud_cover_frac"]
        )

    if "tavg" in out.columns:
        out["tavg_diff"] = out["tavg"].diff().fillna(0.0)
        out["tavg_ewm_7"] = out["tavg"].ewm(span=7, adjust=False).mean()

    if "wdir" in out.columns:
        radians = np.deg2rad(out["wdir"])
        out["wdir_sin"] = np.sin(radians)
        out["wdir_cos"] = np.cos(radians)

    if "wdir" in out.columns and "wspd" in out.columns:
        out["wdir_x_wspd"] = out["wdir"] * out["wspd"]

    return out


def engineer_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    out = add_time_features(out, date_col=date_col)
    out = out.sort_values(date_col).reset_index(drop=True)
    out = add_interaction_features(out)

    return out