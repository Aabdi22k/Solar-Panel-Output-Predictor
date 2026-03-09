"""NREL NSRDB data source client.

This module fetches half-hourly solar and meteorological variables from the
NREL NSRDB GOES Aggregated PSM v4 API and returns them as pandas DataFrames.

"""

from __future__ import annotations

import io
import time
from typing import Iterable

import pandas as pd
import requests

NSRDB_GOES_AGG_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/"
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
) -> pd.DataFrame:
    """Fetch half-hourly NSRDB GOES Aggregated PSM v4 data for multiple years.

    This function downloads CSV responses from the NSRDB API for each requested
    year and concatenates them into a single DataFrame. The API returns extra
    header rows, which are skipped when parsing.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        years: Years to request
            (each value is passed to the NSRDB `names` param).
        api_key: NREL developer API key.
        email: Email address required by the NSRDB API.
        sleep_seconds: Seconds to sleep between year requests to be polite
            to the API.

    Returns:
        A DataFrame containing concatenated half-hourly NSRDB records across all
        requested years.

    Raises:
        RuntimeError: If an NSRDB request returns a non-200 response.
        requests.RequestException: If a network-level error occurs.
    """
    frames: list[pd.DataFrame] = []

    for year in years:
        params = {
            "api_key": api_key,
            "wkt": f"POINT({longitude} {latitude})",
            "names": str(year),
            "leap_day": "false",
            "interval": "30",
            "utc": "true",
            "email": email,
            "attributes": "ghi,dhi,dni,air_temperature,wind_speed",
        }

        resp = requests.get(NSRDB_GOES_AGG_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"""NREL NSRDB request failed for
                    {year}: {resp.status_code} {resp.text[:200]}""")

        # NSRDB CSV has extra header rows; your original used skiprows=2
        df_year = pd.read_csv(io.StringIO(resp.text), skiprows=2)
        frames.append(df_year)

        time.sleep(sleep_seconds)

    return pd.concat(frames, ignore_index=True)
