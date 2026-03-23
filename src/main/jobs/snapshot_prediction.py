from __future__ import annotations

import argparse
import os
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from main.config import AppDefaults, Location, TrainingConfig
from main.data_pipeline.build_dataset import build_training_dataset
from main.data_pipeline.forecast import build_forecast_features
from main.models.history import upsert_prediction_file
from main.models.predict import predict_ghi
from main.models.train import load_artifacts, save_artifacts, train_random_forest
from main.paths import ProjectPaths


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = AppDefaults()
TRAINING = TrainingConfig()


def _tag_for_location(lat: float, lon: float) -> str:
    return f"{lat}_{lon}".replace(".", "p")


def _get_location_by_key(location_key: str) -> Location:
    for location in DEFAULTS.LOCATIONS:
        if location.key == location_key:
            return location
    valid_keys = ", ".join(location.key for location in DEFAULTS.LOCATIONS)
    raise ValueError(f"Invalid location_key '{location_key}'. Valid keys: {valid_keys}")


def _get_secret(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _artifact_paths(paths: ProjectPaths, tag: str) -> tuple[Path, Path, Path]:
    model_path = paths.models_dir / f"model_{tag}.joblib"
    meta_path = paths.models_dir / f"meta_{tag}.json"
    scaler_path = paths.models_dir / f"scaler_{tag}.joblib"
    return model_path, meta_path, scaler_path


def _artifacts_exist(paths: ProjectPaths, tag: str) -> bool:
    model_path, meta_path, scaler_path = _artifact_paths(paths, tag)
    return model_path.exists() and meta_path.exists() and scaler_path.exists()


def load_or_train(
    *,
    location: Location,
    repo_root: Path,
):
    paths = ProjectPaths.from_repo_root(repo_root)
    paths.ensure_dirs()

    tag = _tag_for_location(location.latitude, location.longitude)

    if _artifacts_exist(paths, tag):
        artifacts = load_artifacts(models_dir=paths.models_dir, tag=tag)
        return artifacts, paths, True

    years = list(range(TRAINING.start_year, TRAINING.end_year + 1))
    nrel_api_key = _get_secret("NREL_API_KEY")
    nrel_email = _get_secret("NREL_EMAIL")

    if not nrel_api_key or not nrel_email:
        raise RuntimeError("Missing NREL_API_KEY / NREL_EMAIL in environment variables.")

    dataset_cache = paths.raw_data_dir / (
        f"nsrdb_{tag}_{TRAINING.start_year}_{TRAINING.end_year}.csv"
    )

    df = build_training_dataset(
        latitude=location.latitude,
        longitude=location.longitude,
        years=years,
        nrel_api_key=nrel_api_key,
        nrel_email=nrel_email,
        open_meteo_start=date(TRAINING.start_year, 1, 1),
        open_meteo_end=date(TRAINING.end_year, 12, 31),
        cache_csv_path=dataset_cache,
        timezone=location.timezone,
    )

    artifacts = train_random_forest(
        df,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
    )
    save_artifacts(artifacts, models_dir=paths.models_dir, tag=tag)

    return artifacts, paths, False


def _default_target_date_for_location(location: Location) -> date:
    local_now = datetime.now(ZoneInfo(location.timezone))
    return local_now.date()


def _parse_target_date(date_str: str | None, location: Location) -> date:
    if not date_str:
        return _default_target_date_for_location(location)

    try:
        return date.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid --date '{date_str}'. Use YYYY-MM-DD.") from exc


def snapshot_prediction_for_location(
    *,
    location: Location,
    target_date: date,
) -> dict[str, object]:
    artifacts, paths, loaded_from_artifacts = load_or_train(
        location=location,
        repo_root=REPO_ROOT,
    )

    tag = _tag_for_location(location.latitude, location.longitude)
    history_file = paths.history_file(tag)

    forecast_df = build_forecast_features(
        latitude=location.latitude,
        longitude=location.longitude,
        days=7,
        forecasts_dir=paths.forecasts_dir,
    )

    if forecast_df.empty:
        raise RuntimeError(f"No forecast features generated for location {location.key}.")

    preds_wh_m2 = predict_ghi(
        artifacts.model,
        artifacts.scaler,
        forecast_df,
    )

    if len(preds_wh_m2) == 0:
        raise RuntimeError(f"No predictions generated for location {location.key}.")

    predicted_ghi_kwh_m2 = round(float(preds_wh_m2[0]) / 1000.0, 2)

    updated_history = upsert_prediction_file(
        history_file,
        target_date=target_date.isoformat(),
        predicted_ghi_kwh_m2=predicted_ghi_kwh_m2,
        model_tag=tag,
        loaded_from_artifacts=loaded_from_artifacts,
    )

    return {
        "location_key": location.key,
        "location_name": location.name,
        "target_date": target_date.isoformat(),
        "predicted_ghi_kwh_m2": predicted_ghi_kwh_m2,
        "history_file": str(history_file),
        "history_count": len(updated_history),
        "generated_at": datetime.now(UTC).isoformat(),
        "loaded_from_artifacts": loaded_from_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot today's predicted GHI into history."
    )
    parser.add_argument(
        "--location-key",
        help="Optional single location key to snapshot. If omitted, snapshots all locations.",
    )
    parser.add_argument(
        "--date",
        help="Optional target date in YYYY-MM-DD. If omitted, uses today in each location timezone.",
    )

    args = parser.parse_args()

    if args.location_key:
        locations = [_get_location_by_key(args.location_key)]
    else:
        locations = list(DEFAULTS.LOCATIONS)

    results: list[dict[str, object]] = []

    for location in locations:
        target_date = _parse_target_date(args.date, location)
        result = snapshot_prediction_for_location(
            location=location,
            target_date=target_date,
        )
        results.append(result)

    for result in results:
        print(
            "[snapshot_prediction]"
            f" location={result['location_key']}"
            f" date={result['target_date']}"
            f" predicted_ghi_kwh_m2={result['predicted_ghi_kwh_m2']}"
            f" loaded_from_artifacts={result['loaded_from_artifacts']}"
            f" history_file={result['history_file']}"
            f" history_count={result['history_count']}"
        )


if __name__ == "__main__":
    main()