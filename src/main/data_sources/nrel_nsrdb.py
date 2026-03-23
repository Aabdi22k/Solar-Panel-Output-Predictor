"""NREL NSRDB data source client.

This module fetches half-hourly solar and meteorological variables from the
NREL NSRDB GOES Aggregated PSM v4 API and returns them as pandas DataFrames.
"""

from __future__ import annotations

import io
import os
import time
from typing import Iterable

import pandas as pd
import requests


NSRDB_GOES_AGG_URL = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-aggregated-v4-0-0-download.csv"
)


def fetch_nsrdb_half_hourly(
    *,
    latitude: float,
    longitude: float,
    years: Iterable[int],
    api_key: str,
    email: str,
    sleep_seconds: int = 5,
    utc: bool = False,
) -> pd.DataFrame:
    """Fetch half-hourly NSRDB GOES Aggregated PSM v4 data for multiple years."""

    frames: list[pd.DataFrame] = []

    for year in years:
        params = {
            "api_key": api_key,
            "wkt": f"POINT({longitude} {latitude})",
            "names": str(year),
            "leap_day": "false",
            "interval": "30",
            "utc": "true" if utc else "false",
            "email": email,
            "attributes": "ghi,dhi,dni,air_temperature,wind_speed",
        }

        resp = requests.get(NSRDB_GOES_AGG_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"NREL NSRDB request failed for {year}: "
                f"{resp.status_code} {resp.text[:200]}"
            )

        df_year = pd.read_csv(io.StringIO(resp.text), skiprows=2)
        if df_year.empty:
            raise RuntimeError(f"NREL NSRDB returned empty data for year {year}.")

        frames.append(df_year)
        time.sleep(sleep_seconds)

    out = pd.concat(frames, ignore_index=True)

    # Normalize numeric columns where possible
    for col in ["Year", "Month", "Day", "Hour", "Minute", "GHI"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
