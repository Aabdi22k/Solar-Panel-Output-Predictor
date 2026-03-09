"""Configuration objects for the solar prediction pipeline.

This module defines immutable configuration containers used across
training, forecasting, and the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """Represents a geographic location.

    Attributes:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        name: Human-readable location name.
    """

    latitude: float
    longitude: float
    name: str = "Unknown"


@dataclass(frozen=True)
class TrainingConfig:
    """Stores parameters for dataset building and model training.

    Attributes:
        start_year: First year (inclusive) of historical data to use.
        end_year: Last year (inclusive) of historical data to use.
        interval_minutes: Sampling interval in minutes for time-series data.
        test_size: Fraction of samples reserved for the test split.
        random_state: Random seed for reproducibility.
    """

    start_year: int = 1998
    end_year: int = 2023
    interval_minutes: int = 30
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ForecastConfig:
    """Stores parameters for generating forward forecasts.

    Attributes:
        days: Number of days to forecast.
        timezone: Timezone used to align forecast timestamps.
    """

    days: int = 5
    timezone: str = "auto"


@dataclass(frozen=True)
class AppDefaults:
    """Defines default inputs used by the app UI and pipeline.

    Attributes:
        location: Default geographic location used for predictions.
        array_area_m2: Default solar array area in square meters.
        panel_efficiency: Default panel efficiency as a fraction (0-1).
    """

    location: Location = Location(33.448376, -112.074036, "Phoenix, AZ")
    array_area_m2: float = 10.0
    panel_efficiency: float = 0.15
