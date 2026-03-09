"""Streamlit app for solar output forecasting.

This module defines the Streamlit UI for the Solar Output Predictor. It loads
(or trains) a location-specific model, fetches forecast features, predicts
GHI, converts predictions to energy output (kWh), and displays uncertainty
bands.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from main.config import AppDefaults, TrainingConfig
from main.data_pipeline.build_dataset import build_training_dataset
from main.data_pipeline.forecast import build_forecast_features
from main.math.output_bands import output_bands, output_bands_std
from main.models.predict import predict_ghi
from main.models.train import (load_artifacts, save_artifacts,
                               train_random_forest)
from main.paths import ProjectPaths

DEFAULTS = AppDefaults()
TRAINING = TrainingConfig()


def _tag_for_location(lat: float, lon: float) -> str:
    """Create a filesystem-safe tag string for a latitude/longitude pair.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        A stable string tag suitable for filenames.
    """
    return f"{lat}_{lon}".replace(".", "p")


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret value from environment variables or Streamlit secrets.

    Environment variables take priority. If Streamlit secrets are unavailable
    (for example, running locally without secrets configured), default is
    returned.

    Args:
        key: Secret key name.
        default: Value to return if the secret cannot be found.

    Returns:
        The secret value if present, otherwise default.
    """
    value = os.getenv(key)
    if value:
        return value

    try:
        return st.secrets.get(key, default)
    except st.StreamlitSecretNotFoundError:
        return default


@st.cache_resource
def load_or_train(lat: float, lon: float, repo_root: Path):
    """Load a cached model for a location or train a new one.

    This function checks for existing model, scaler, and metadata files in the
    models directory. If found, it loads them. Otherwise, it builds the
    training dataset, trains the model, saves artifacts, and returns them.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        repo_root: Repository root path used to resolve data and artifact
            directories.

    Returns:
        A tuple (artifacts, paths), where artifacts is a ModelArtifacts
        instance and paths is a ProjectPaths instance.
    """
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure_dirs()

    tag = _tag_for_location(lat, lon)
    model_path = paths.models_dir / f"model_{tag}.joblib"
    meta_path = paths.models_dir / f"meta_{tag}.json"
    scaler_path = paths.models_dir / f"scaler_{tag}.joblib"

    if model_path.exists() and meta_path.exists() and scaler_path.exists():
        return (
            load_artifacts(
                models_dir=paths.models_dir,
                tag=tag,
            ),
            paths,
        )

    years = list(range(TRAINING.start_year, TRAINING.end_year + 1))
    nrel_api_key = _get_secret("NREL_API_KEY")
    nrel_email = _get_secret("NREL_EMAIL")

    if not nrel_api_key or not nrel_email:
        raise RuntimeError(
            "Missing NREL_API_KEY / NREL_EMAIL in Streamlit secrets."
        )

    dataset_cache = paths.raw_data_dir / (
        f"nsrdb_{tag}_{TRAINING.start_year}_" f"{TRAINING.end_year}.csv"
    )

    df = build_training_dataset(
        latitude=lat,
        longitude=lon,
        years=years,
        nrel_api_key=nrel_api_key,
        nrel_email=nrel_email,
        open_meteo_start=date(TRAINING.start_year, 1, 1),
        open_meteo_end=date(TRAINING.end_year, 12, 31),
        cache_csv_path=dataset_cache,
    )

    artifacts = train_random_forest(
        df,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
    )
    save_artifacts(artifacts, models_dir=paths.models_dir, tag=tag)
    return artifacts, paths


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Solar Output Predictor",
        layout="wide",
    )
    st.title("☀️ Solar Output Predictor")

    repo_root = Path(__file__).resolve().parents[3]
    lat = DEFAULTS.location.latitude
    lon = DEFAULTS.location.longitude

    with st.form("inputs"):
        st.subheader("System Inputs")
        st.number_input("Latitude", value=lat, disabled=True)
        st.number_input("Longitude", value=lon, disabled=True)

        array_area = st.number_input(
            "Array area (m²)",
            min_value=1.0,
            value=DEFAULTS.array_area_m2,
            step=0.5,
        )
        efficiency = st.number_input(
            "Panel efficiency (e.g. 0.15 = 15%)",
            min_value=0.01,
            max_value=1.0,
            value=DEFAULTS.panel_efficiency,
            step=0.01,
        )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    with st.spinner("Loading model + forecast…"):
        artifacts, paths = load_or_train(lat, lon, repo_root)
        forecast_df = build_forecast_features(
            latitude=lat,
            longitude=lon,
            days=5,
            forecasts_dir=paths.forecasts_dir,
        )
        ghi_pred = predict_ghi(
            artifacts.model,
            artifacts.scaler,
            forecast_df,
        )

        bands_mae = output_bands(
            array_area_m2=array_area,
            efficiency=efficiency,
            ghi_pred=ghi_pred,
            band=artifacts.mae,
        )
        std_bands = output_bands_std(
            array_area_m2=array_area,
            efficiency=efficiency,
            ghi_pred=ghi_pred,
            std=artifacts.error_std,
        )

    st.subheader("📈 Model Accuracy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("±MAE", f"{artifacts.accuracy_bands['MAE']:.2f}%")
    c2.metric("±1 STD", f"{artifacts.accuracy_bands['1std']:.2f}%")
    c3.metric("±2 STD", f"{artifacts.accuracy_bands['2std']:.2f}%")
    c4.metric("±3 STD", f"{artifacts.accuracy_bands['3std']:.2f}%")

    st.subheader("🔮 5-Day Forecast (Starting Tomorrow)")
    start = date.today() + timedelta(days=1)
    dates = [start + timedelta(days=i) for i in range(len(ghi_pred))]

    cols = st.columns(3)
    for i, forecast_date in enumerate(dates):
        col = cols[i % 3]
        with col:
            st.markdown(f"### {forecast_date.strftime("%b %d, %Y")}")
            st.write(f"**GHI:** {(ghi_pred[i] / 1000):.2f} kW/m²")
            st.write(
                f"**Output (±MAE):** "
                f"{bands_mae[i][0]:.2f} – {bands_mae[i][1]:.2f} kWh"
            )
            st.write(
                f"**Output (±1 STD):** "
                f"{std_bands["1std"][i][0]:.2f} – "
                f"{std_bands["1std"][i][1]:.2f} kWh"
            )
            st.write(
                f"**Output (±2 STD):** "
                f"{std_bands["2std"][i][0]:.2f} – "
                f"{std_bands["2std"][i][1]:.2f} kWh"
            )
            st.write(
                f"**Output (±3 STD):** "
                f"{std_bands["3std"][i][0]:.2f} – "
                f"{std_bands["3std"][i][1]:.2f} kWh"
            )


if __name__ == "__main__":
    main()
