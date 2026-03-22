from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import DEFAULT

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main.config import AppDefaults, Location, TrainingConfig
from main.data_pipeline.build_dataset import build_training_dataset
from main.data_pipeline.forecast import build_forecast_features
from main.models.history import load_history_file, calculate_accuracy_bands_percent
from main.models.predict import predict_ghi
from main.models.train import (
    load_artifacts,
    save_artifacts,
    train_random_forest,
)
from main.paths import ProjectPaths


app = FastAPI(title="Solar Output Predictor API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULTS = AppDefaults()
TRAINING = TrainingConfig()

REPO_ROOT = Path(__file__).resolve().parents[3]

FRONTEND_DIR = REPO_ROOT / "src" / "main" / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

def _get_location_by_key(location_key: str) -> Location:
    for location in DEFAULTS.LOCATIONS:
        if location.key == location_key:
            return location
    raise HTTPException(status_code=400, detail=f"Invalid location_key: {location_key}")


def _tag_for_location(lat: float, lon: float) -> str:
    return f"{lat}_{lon}".replace(".", "p")


def _get_secret(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _compute_output_kwh(
    predicted_ghi_kwh_m2: float,
    array_area_m2: float,
    panel_efficiency: float,
) -> float:
    return predicted_ghi_kwh_m2 * array_area_m2 * panel_efficiency


def _ghi_band(
    predicted_ghi_kwh_m2: float,
    half_width_kwh_m2: float,
) -> dict[str, float]:
    low = max(0.0, predicted_ghi_kwh_m2 - half_width_kwh_m2)
    high = predicted_ghi_kwh_m2 + half_width_kwh_m2
    return {
        "low": round(low, 2),
        "high": round(high, 2),
    }


def _build_ghi_bands(
    predicted_ghi_kwh_m2: float,
    mae_kwh_m2: float,
    std_kwh_m2: float,
) -> dict[str, dict[str, float]]:
    return {
        "mae": _ghi_band(predicted_ghi_kwh_m2, mae_kwh_m2),
        "1std": _ghi_band(predicted_ghi_kwh_m2, std_kwh_m2),
        "2std": _ghi_band(predicted_ghi_kwh_m2, 2 * std_kwh_m2),
        "3std": _ghi_band(predicted_ghi_kwh_m2, 3 * std_kwh_m2),
    }


def _label_for_future_date(forecast_date: date) -> str:
    return forecast_date.strftime("%a")


def load_or_train(lat: float, lon: float, repo_root: Path):
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure_dirs()

    tag = _tag_for_location(lat, lon)
    model_path = paths.models_dir / f"model_{tag}.joblib"
    meta_path = paths.models_dir / f"meta_{tag}.json"
    scaler_path = paths.models_dir / f"scaler_{tag}.joblib"

    if model_path.exists() and meta_path.exists() and scaler_path.exists():
        artifacts = load_artifacts(models_dir=paths.models_dir, tag=tag)
        return artifacts, paths

    years = list(range(TRAINING.start_year, TRAINING.end_year + 1))
    nrel_api_key = _get_secret("NREL_API_KEY")
    nrel_email = _get_secret("NREL_EMAIL")

    if not nrel_api_key or not nrel_email:
        raise RuntimeError(
            "Missing NREL_API_KEY / NREL_EMAIL in environment variables."
        )

    dataset_cache = paths.raw_data_dir / (
        f"nsrdb_{tag}_{TRAINING.start_year}_{TRAINING.end_year}.csv"
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


class BandRange(BaseModel):
    low: float
    high: float


class AccuracyBand(BaseModel):
    model: float
    actual: float
    delta: float


class ForecastDayResponse(BaseModel):
    date: str
    label: str
    predicted_ghi_kwh_m2: float
    actual_ghi_kwh_m2: float | None = None
    ghi_delta_kwh_m2: float | None = None
    estimated_output_kwh: float
    ghi_bands_kwh_m2: dict[str, BandRange]


class HistoryModelInfo(BaseModel):
    tag: str
    generated_at: str
    loaded_from_artifacts: bool


class HistoryEntry(BaseModel):
    date: str
    predicted_ghi_kwh_m2: float | None
    actual_ghi_kwh_m2: float | None
    delta_kwh_m2: float | None
    model_info: HistoryModelInfo


class ForecastMeta(BaseModel):
    location_name: str
    latitude: float
    longitude: float
    generated_at: str
    forecast_horizon_days: int
    array_area_m2: float
    panel_efficiency: float
    model_loaded_from_artifacts: bool


class ForecastResponse(BaseModel):
    meta: ForecastMeta
    forecast_days: list[ForecastDayResponse]
    history: list[HistoryEntry]
    accuracy_bands_percent: dict[str, AccuracyBand]

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/locations")
def get_locations():
    return DEFAULTS.LOCATIONS


@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(
    location_key: str = Query(
        default="phoenix",
        description="Location key from /api/locations",
    ),
    array_area_m2: float = Query(
        default=DEFAULTS.array_area_m2,
        gt=0,
        description="Solar panel array area in square meters.",
    ),
    panel_efficiency: float = Query(
        default=DEFAULTS.panel_efficiency,
        gt=0,
        le=1,
        description="Panel efficiency as a decimal (e.g. 0.20 = 20%).",
    ),
):
    try:
        location = _get_location_by_key(location_key)
        lat = location.latitude
        lon = location.longitude

        tag = _tag_for_location(lat, lon)
        paths = ProjectPaths.from_repo_root(REPO_ROOT)
        paths.ensure_dirs()

        model_path = paths.models_dir / f"model_{tag}.joblib"
        meta_path = paths.models_dir / f"meta_{tag}.json"
        scaler_path = paths.models_dir / f"scaler_{tag}.joblib"
        already_trained = (
            model_path.exists() and meta_path.exists() and scaler_path.exists()
        )

        artifacts, paths = load_or_train(lat, lon, REPO_ROOT)

        history_file = paths.history_file(tag)

        forecast_df = build_forecast_features(
            latitude=lat,
            longitude=lon,
            days=7,
            forecasts_dir=paths.forecasts_dir,
        )

        ghi_pred_wh_m2 = predict_ghi(
            artifacts.model,
            artifacts.scaler,
            forecast_df,
        )

        ghi_pred_kwh_m2 = [round(float(val) / 1000.0, 2) for val in ghi_pred_wh_m2]
        mae_kwh_m2 = float(artifacts.mae) / 1000.0
        std_kwh_m2 = float(artifacts.error_std) / 1000.0

        today_date = date.today()
        history = load_history_file(history_file)

        actual_accuracy_bands = calculate_accuracy_bands_percent(
                                    history=history,
                                    mae_kwh_m2=mae_kwh_m2,
                                    std_kwh_m2=std_kwh_m2,
                                )

        forecast_days: list[ForecastDayResponse] = []
        for i, predicted_ghi in enumerate(ghi_pred_kwh_m2):
            forecast_date = today_date + timedelta(days=i)
            estimated_output = round(
                _compute_output_kwh(predicted_ghi, array_area_m2, panel_efficiency),
                2,
            )

            bands_raw = _build_ghi_bands(
                predicted_ghi_kwh_m2=predicted_ghi,
                mae_kwh_m2=mae_kwh_m2,
                std_kwh_m2=std_kwh_m2,
            )

            forecast_days.append(
                ForecastDayResponse(
                    date=forecast_date.isoformat(),
                    label="Today" if i == 0 else _label_for_future_date(forecast_date),
                    predicted_ghi_kwh_m2=predicted_ghi,
                    actual_ghi_kwh_m2=None,
                    ghi_delta_kwh_m2=None,
                    estimated_output_kwh=estimated_output,
                    ghi_bands_kwh_m2={
                        key: BandRange(**value)
                        for key, value in bands_raw.items()
                    },
                )
            )

        history_response = [
            HistoryEntry(
                date=entry.get("date"),
                predicted_ghi_kwh_m2=entry.get("predicted_ghi_kwh_m2"),
                actual_ghi_kwh_m2=entry.get("actual_ghi_kwh_m2"),
                delta_kwh_m2=entry.get("delta_kwh_m2"),
                model_info=HistoryModelInfo(**entry.get("model_info", {})),
            )
            for entry in history
        ]

        accuracy_bands_percent = {}
        for key in ("MAE", "1std", "2std", "3std"):
            model_val = round(float(artifacts.accuracy_bands[key]), 2)
            actual_val = round(float(actual_accuracy_bands[key]), 2)
            accuracy_bands_percent[key] = AccuracyBand(
                model=model_val,
                actual=actual_val,
                delta=round(actual_val - model_val, 2),
            )

        return ForecastResponse(
            meta=ForecastMeta(
                location_name=location.name,
                latitude=lat,
                longitude=lon,
                generated_at=datetime.now(UTC).isoformat(),
                forecast_horizon_days=7,
                array_area_m2=array_area_m2,
                panel_efficiency=panel_efficiency,
                model_loaded_from_artifacts=already_trained,
            ),
            forecast_days=forecast_days,
            history=history_response,
            accuracy_bands_percent=accuracy_bands_percent,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc