from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.main.config import Location, AppDefaults
from src.main.data_sources.open_meteo import fetch_actual_ghi_today
from src.main.models.history import update_actual_ghi_file
from src.main.paths import ProjectPaths


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


def _default_target_date_for_location(location: Location) -> date:
    local_now = datetime.now(ZoneInfo(location.timezone))
    return local_now.date() - timedelta(days=1)


def _parse_target_date(date_str: str | None, location: Location) -> date:
    if not date_str:
        return _default_target_date_for_location(location)

    try:
        return date.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid --date '{date_str}'. Use YYYY-MM-DD.") from exc


def update_history_for_location(
    *,
    location: Location,
    target_date: date,
) -> dict[str, object]:
    paths = ProjectPaths.from_repo_root(REPO_ROOT)
    paths.ensure_dirs()

    tag = _tag_for_location(location.latitude, location.longitude)
    history_file = paths.history_file(tag)

    actual_ghi_kwh_m2 = fetch_actual_ghi_today(
        latitude=location.latitude,
        longitude=location.longitude,
        target_date=target_date,
        timezone=location.timezone,
    )

    updated_history = update_actual_ghi_file(
        history_file,
        target_date=target_date.isoformat(),
        actual_ghi_kwh_m2=actual_ghi_kwh_m2,
    )

    return {
        "location_key": location.key,
        "location_name": location.name,
        "target_date": target_date.isoformat(),
        "history_file": str(history_file),
        "actual_ghi_kwh_m2": actual_ghi_kwh_m2,
        "history_count": len(updated_history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update history JSON with actual GHI for completed days."
    )
    parser.add_argument(
        "--location-key",
        help="Optional single location key to update. If omitted, updates all locations.",
    )
    parser.add_argument(
        "--date",
        help="Optional target date in YYYY-MM-DD. If omitted, uses yesterday in each location timezone.",
    )

    args = parser.parse_args()

    if args.location_key:
        locations = [_get_location_by_key(args.location_key)]
    else:
        locations = list(DEFAULTS.LOCATIONS)

    results: list[dict[str, object]] = []

    for location in locations:
        target_date = _parse_target_date(args.date, location)
        result = update_history_for_location(
            location=location,
            target_date=target_date,
        )
        results.append(result)

    for result in results:
        print(
            "[update_history]"
            f" location={result['location_key']}"
            f" date={result['target_date']}"
            f" actual_ghi_kwh_m2={result['actual_ghi_kwh_m2']}"
            f" history_file={result['history_file']}"
            f" history_count={result['history_count']}"
        )


if __name__ == "__main__":
    main()