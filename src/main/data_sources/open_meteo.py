"""Open-Meteo weather data source client.

This module fetches historical and forecast weather data from Open-Meteo and
converts hourly weather fields into daily aggregates used by the pipeline.

"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = (
    "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,"
    "pressure_msl,cloudcover,precipitation"
)
DAILY_FIELDS = "sunshine_duration,daylight_duration"


def _aggregate_hourly_to_daily(hourly: dict) -> pd.DataFrame:
    """Aggregate Open-Meteo hourly payload into daily features.

    Args:
        hourly: Open-Meteo "hourly" payload (dict of lists) containing the
            fields listed in HOURLY_FIELDS and the "time" list.

    Returns:
        A daily DataFrame with one row per date and aggregated columns such as
        mean temperature, precipitation sum, and average cloud cover.
    """

    df_hourly = pd.DataFrame(hourly)
    # Ensure datetime conversion
    dt_series = pd.to_datetime(df_hourly["time"], errors="coerce")
    df_hourly["time"] = dt_series

    # Use DatetimeIndex to avoid Pylance .dt warnings
    dt_index = pd.DatetimeIndex(dt_series)
    df_hourly["date"] = dt_index.date

    df_daily = (
        df_hourly.groupby("date")
        .agg(
            tavg=("temperature_2m", "mean"),
            tmin=("temperature_2m", "min"),
            tmax=("temperature_2m", "max"),
            prcp=("precipitation", "sum"),
            humidity=("relativehumidity_2m", "mean"),
            wspd=("windspeed_10m", "mean"),
            wdir=("winddirection_10m", "mean"),
            pres=("pressure_msl", "mean"),
            cloud_cover=("cloudcover", "mean"),
        )
        .reset_index()
    )
    df_daily["date"] = df_daily["date"].astype(str)
    return df_daily


def _merge_daily_fields(df_daily: pd.DataFrame, daily: dict) -> pd.DataFrame:
    """Merge Open-Meteo daily payload fields into an existing daily DataFrame.

    Args:
        df_daily: Daily DataFrame created from aggregating hourly data. Must
            contain a "date" column as a string.
        daily: Open-Meteo "daily" payload (dict of lists) containing at least
            "time" and the fields listed in DAILY_FIELDS.

    Returns:
        A DataFrame with sunshine_duration and daylight_duration columns merged
        onto the daily aggregates.
    """

    df_extra = pd.DataFrame(daily)
    df_extra["time"] = pd.to_datetime(df_extra["time"]).dt.date.astype(str)

    out = df_daily.merge(
        df_extra[["time", "sunshine_duration", "daylight_duration"]],
        left_on="date",
        right_on="time",
        how="left",
    ).drop(columns=["time"])
    return out


def fetch_historical_weather_daily(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "auto",
) -> pd.DataFrame:
    """Fetch historical weather and return daily aggregates.

    Pulls hourly data from the Open-Meteo archive endpoint, aggregates it into
    daily features, and merges any available daily fields.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        start_date: First date (inclusive) to request.
        end_date: Last date (inclusive) to request.
        timezone: Timezone used by Open-Meteo for timestamp alignment.

    Returns:
        A daily DataFrame with one row per date and aggregated weather features.

    Raises:
        RuntimeError: If the Open-Meteo request fails or returns no hourly data.
        requests.RequestException: If a network-level error occurs.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": HOURLY_FIELDS,
        "daily": DAILY_FIELDS,
        "timezone": timezone,
    }

    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"""Open-Meteo archive failed:
                           {resp.status_code} {resp.text[:200]}""")

    payload = resp.json()
    hourly = payload.get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo archive returned no hourly data.")

    df_daily = _aggregate_hourly_to_daily(hourly)
    daily = payload.get("daily", {})
    if daily:
        df_daily = _merge_daily_fields(df_daily, daily)
    else:
        df_daily["sunshine_duration"] = None
        df_daily["daylight_duration"] = None

    return df_daily


def fetch_forecast_weather_daily(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "auto",
) -> pd.DataFrame:
    """Fetch forecast weather and return daily aggregates.

    Pulls hourly data from the Open-Meteo forecast endpoint, aggregates it into
    daily features, and merges any available daily fields.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        start_date: First date (inclusive) to request.
        end_date: Last date (inclusive) to request.
        timezone: Timezone used by Open-Meteo for timestamp alignment.

    Returns:
        A daily DataFrame with one row per date and aggregated weather features.

    Raises:
        RuntimeError: If the Open-Meteo request fails or returns no hourly data.
        requests.RequestException: If a network-level error occurs.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": HOURLY_FIELDS,
        "daily": DAILY_FIELDS,
        "timezone": timezone,
    }

    resp = requests.get(FORECAST_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"""Open-Meteo forecast failed:
                           {resp.status_code} {resp.text[:200]}""")

    payload = resp.json()
    hourly = payload.get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo forecast returned no hourly data.")

    df_daily = _aggregate_hourly_to_daily(hourly)
    daily = payload.get("daily", {})
    if daily:
        df_daily = _merge_daily_fields(df_daily, daily)
    else:
        df_daily["sunshine_duration"] = None
        df_daily["daylight_duration"] = None

    return df_daily


def fetch_actual_ghi_today(
    *,
    latitude: float,
    longitude: float,
    target_date: date,
    timezone: str = "auto",
) -> float | None:
    """
    Fetch actual daily GHI (Global Horizontal Irradiance) for a given date.

    Uses Open-Meteo archive API and aggregates hourly shortwave_radiation
    into daily kWh/m².

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        target_date: Date to fetch actual GHI for.
        timezone: Timezone for alignment.

    Returns:
        Daily GHI in kWh/m², or None if unavailable.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": target_date.strftime("%Y-%m-%d"),
        "end_date": target_date.strftime("%Y-%m-%d"),
        "hourly": "shortwave_radiation",
        "timezone": timezone,
    }

    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo GHI archive failed: {resp.status_code} {resp.text[:200]}"
        )

    payload = resp.json()
    hourly = payload.get("hourly", {})

    if not hourly or "shortwave_radiation" not in hourly:
        return None

    df = pd.DataFrame({
        "time": hourly.get("time"),
        "ghi": hourly.get("shortwave_radiation"),
    })

    if df.empty:
        return None

    # Convert time
    dt_series = pd.to_datetime(df["time"], errors="coerce")
    df["date"] = pd.DatetimeIndex(dt_series).date

    # Filter to target date only (safety)
    df = df[df["date"] == target_date]

    if df.empty:
        return None

    # shortwave_radiation is in W/m² (hourly average)
    # Convert to energy:
    # Wh/m² = sum(W/m² over each hour)
    # kWh/m² = Wh/m² / 1000
    ghi_wh_m2 = df["ghi"].sum()
    ghi_kwh_m2 = ghi_wh_m2 / 1000.0

    return round(float(ghi_kwh_m2), 2)