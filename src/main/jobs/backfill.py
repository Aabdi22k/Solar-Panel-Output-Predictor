from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from main.config import AppDefaults, Location
from main.data_sources.open_meteo import (
    fetch_actual_ghi_today,
    fetch_historical_weather_daily,
)
from main.features.cleaning import drop_na_rows
from main.features.engineering import engineer_features
from main.models.history import (
    upsert_prediction_file,
    update_actual_ghi_file,
)
from main.models.predict import predict_ghi
from main.models.train import load_artifacts
from main.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = AppDefaults()


def _tag_for_location(lat: float, lon: float) -> str:
    return f"{lat}_{lon}".replace(".", "p")


def _get_location_by_key(location_key: str) -> Location:
    for location in DEFAULTS.LOCATIONS:
        if location.key == location_key:
            return location
    valid_keys = ", ".join(location.key for location in DEFAULTS.LOCATIONS)
    raise ValueError(f"Invalid location_key '{location_key}'. Valid keys: {valid_keys}")


def _default_date_range(location: Location, days: int) -> tuple[date, date]:
    local_now = datetime.now(ZoneInfo(location.timezone))
    end_date = local_now.date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def _normalize_to_date_obj(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()

    if isinstance(value, datetime):
        return value.date()

    return datetime.fromisoformat(str(value)).date()


def _build_historical_features(
    *,
    location: Location,
    start_date: date,
    end_date: date,
):
    df = fetch_historical_weather_daily(
        latitude=location.latitude,
        longitude=location.longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=location.timezone,
    )

    df = engineer_features(df)
    df = drop_na_rows(df)

    if "date" not in df.columns:
        raise RuntimeError("Historical feature frame must contain a 'date' column.")

    return df


def backfill_history_for_location(
    *,
    location: Location,
    start_date: date,
    end_date: date,
) -> None:
    paths = ProjectPaths.from_repo_root(REPO_ROOT)
    paths.ensure_dirs()

    tag = _tag_for_location(location.latitude, location.longitude)
    history_file = paths.history_file(tag)

    artifacts = load_artifacts(models_dir=paths.models_dir, tag=tag)

    feature_df = _build_historical_features(
        location=location,
        start_date=start_date,
        end_date=end_date,
    ).reset_index(drop=True)

    preds_wh_m2 = predict_ghi(
        artifacts.model,
        artifacts.scaler,
        feature_df,
    )
    preds_kwh_m2 = [round(float(v) / 1000.0, 2) for v in preds_wh_m2]

    for idx, row in feature_df.iterrows():
        target_date_obj = _normalize_to_date_obj(row["date"])
        target_date_str = target_date_obj.isoformat()
        predicted_ghi_kwh_m2 = preds_kwh_m2[idx]

        upsert_prediction_file(
            history_file,
            target_date=target_date_str,
            predicted_ghi_kwh_m2=predicted_ghi_kwh_m2,
            model_tag=tag,
            loaded_from_artifacts=True,
        )

        actual_ghi_kwh_m2 = fetch_actual_ghi_today(
            latitude=location.latitude,
            longitude=location.longitude,
            target_date=target_date_obj,
            timezone=location.timezone,
        )

        update_actual_ghi_file(
            history_file,
            target_date=target_date_str,
            actual_ghi_kwh_m2=actual_ghi_kwh_m2,
        )

    print(
        "[backfill_history]"
        f" location={location.key}"
        f" start={start_date.isoformat()}"
        f" end={end_date.isoformat()}"
        f" rows={len(feature_df)}"
        f" history_file={history_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill prediction history for a past date range."
    )
    parser.add_argument(
        "--location-key",
        required=True,
        help="Location key to backfill.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill ending yesterday. Ignored if --start-date and --end-date are both provided.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date YYYY-MM-DD.",
    )

    args = parser.parse_args()
    location = _get_location_by_key(args.location_key)

    if args.start_date and args.end_date:
        start_date = _parse_date(args.start_date)
        end_date = _parse_date(args.end_date)
    elif args.start_date or args.end_date:
        raise ValueError("Provide both --start-date and --end-date, or neither.")
    else:
        start_date, end_date = _default_date_range(location, args.days)

    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    backfill_history_for_location(
        location=location,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()