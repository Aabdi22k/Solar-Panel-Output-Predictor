"""Prediction history persistence utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """Metadata describing which model generated a prediction."""

    tag: str
    generated_at: str
    loaded_from_artifacts: bool


@dataclass
class HistoryEntry:
    """Single daily prediction/actual record."""

    date: str
    predicted_ghi_kwh_m2: float | None
    actual_ghi_kwh_m2: float | None
    delta_kwh_m2: float | None
    model_info: ModelInfo


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    """Round numeric values safely, preserving None."""
    if value is None:
        return None
    return round(float(value), digits)


def compute_delta_kwh_m2(
    predicted_ghi_kwh_m2: float | None,
    actual_ghi_kwh_m2: float | None,
) -> float | None:
    """Compute actual minus predicted GHI delta."""
    if predicted_ghi_kwh_m2 is None or actual_ghi_kwh_m2 is None:
        return None
    return round(float(actual_ghi_kwh_m2) - float(predicted_ghi_kwh_m2), 2)


def load_history(history_file: Path) -> list[dict[str, Any]]:
    """Load history records from a JSON file.

    Args:
        history_file: Path to a history JSON file.

    Returns:
        List of history record dicts. Returns [] if file is missing or invalid.
    """
    if not history_file.exists():
        return []

    try:
        with history_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    return payload if isinstance(payload, list) else []


def save_history(history_file: Path, history: list[dict[str, Any]]) -> None:
    """Persist history records to disk.

    Args:
        history_file: Destination JSON file.
        history: History records to write.
    """
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def get_history_entry(
    history: list[dict[str, Any]],
    target_date: str,
) -> dict[str, Any] | None:
    """Find a history record by date."""
    for entry in history:
        if entry.get("date") == target_date:
            return entry
    return None


def upsert_prediction(
    history: list[dict[str, Any]],
    *,
    target_date: str,
    predicted_ghi_kwh_m2: float,
    model_tag: str,
    loaded_from_artifacts: bool,
) -> list[dict[str, Any]]:
    """Insert or update a prediction record for a date.

    If the date already exists, prediction/model_info are updated.
    If it does not exist, a new record is appended.
    """
    predicted_ghi_kwh_m2 = _round_or_none(predicted_ghi_kwh_m2)
    entry = get_history_entry(history, target_date)

    model_info = ModelInfo(
        tag=model_tag,
        generated_at=_utc_now_iso(),
        loaded_from_artifacts=loaded_from_artifacts,
    )

    if entry is None:
        new_entry = HistoryEntry(
            date=target_date,
            predicted_ghi_kwh_m2=predicted_ghi_kwh_m2,
            actual_ghi_kwh_m2=None,
            delta_kwh_m2=None,
            model_info=model_info,
        )
        history.append(asdict(new_entry))
    else:
        entry["predicted_ghi_kwh_m2"] = predicted_ghi_kwh_m2
        entry["model_info"] = asdict(model_info)

        actual_val = entry.get("actual_ghi_kwh_m2")
        entry["delta_kwh_m2"] = compute_delta_kwh_m2(
            predicted_ghi_kwh_m2,
            actual_val,
        )

    history.sort(key=lambda item: item.get("date", ""))
    return history


def update_actual_ghi(
    history: list[dict[str, Any]],
    *,
    target_date: str,
    actual_ghi_kwh_m2: float | None,
) -> list[dict[str, Any]]:
    """Update actual GHI for a date and recompute delta.

    If the date does not yet exist, creates a placeholder entry with no prediction.
    """
    actual_ghi_kwh_m2 = _round_or_none(actual_ghi_kwh_m2)
    entry = get_history_entry(history, target_date)

    if entry is None:
        entry = {
            "date": target_date,
            "predicted_ghi_kwh_m2": None,
            "actual_ghi_kwh_m2": actual_ghi_kwh_m2,
            "delta_kwh_m2": None,
            "model_info": {
                "tag": "",
                "generated_at": _utc_now_iso(),
                "loaded_from_artifacts": False,
            },
        }
        history.append(entry)
    else:
        entry["actual_ghi_kwh_m2"] = actual_ghi_kwh_m2

    entry["delta_kwh_m2"] = compute_delta_kwh_m2(
        entry.get("predicted_ghi_kwh_m2"),
        entry.get("actual_ghi_kwh_m2"),
    )

    history.sort(key=lambda item: item.get("date", ""))
    return history


def load_history_file(history_file: Path) -> list[dict[str, Any]]:
    """Convenience wrapper for reading history."""
    return load_history(history_file)


def save_history_file(history_file: Path, history: list[dict[str, Any]]) -> None:
    """Convenience wrapper for writing history."""
    save_history(history_file, history)


def upsert_prediction_file(
    history_file: Path,
    *,
    target_date: str,
    predicted_ghi_kwh_m2: float,
    model_tag: str,
    loaded_from_artifacts: bool,
) -> list[dict[str, Any]]:
    """Load, upsert a prediction, save, and return updated history."""
    history = load_history(history_file)
    history = upsert_prediction(
        history,
        target_date=target_date,
        predicted_ghi_kwh_m2=predicted_ghi_kwh_m2,
        model_tag=model_tag,
        loaded_from_artifacts=loaded_from_artifacts,
    )
    save_history(history_file, history)
    return history


def update_actual_ghi_file(
    history_file: Path,
    *,
    target_date: str,
    actual_ghi_kwh_m2: float | None,
) -> list[dict[str, Any]]:
    """Load, update actual GHI, save, and return updated history."""
    history = load_history(history_file)
    history = update_actual_ghi(
        history,
        target_date=target_date,
        actual_ghi_kwh_m2=actual_ghi_kwh_m2,
    )
    save_history(history_file, history)
    return history

def calculate_accuracy_bands_percent(
    history: list[dict],
    mae_kwh_m2: float,
    std_kwh_m2: float,
) -> dict[str, float]:
    valid_rows = [
        entry
        for entry in history
        if entry.get("predicted_ghi_kwh_m2") is not None
        and entry.get("actual_ghi_kwh_m2") is not None
    ]

    if not valid_rows:
        return {
            "MAE": 0.0,
            "1std": 0.0,
            "2std": 0.0,
            "3std": 0.0,
        }

    counts = {
        "MAE": 0,
        "1std": 0,
        "2std": 0,
        "3std": 0,
    }

    for entry in valid_rows:
        predicted = float(entry["predicted_ghi_kwh_m2"])
        actual = float(entry["actual_ghi_kwh_m2"])
        abs_error = abs(actual - predicted)

        if abs_error <= mae_kwh_m2:
            counts["MAE"] += 1
        if abs_error <= std_kwh_m2:
            counts["1std"] += 1
        if abs_error <= 2 * std_kwh_m2:
            counts["2std"] += 1
        if abs_error <= 3 * std_kwh_m2:
            counts["3std"] += 1

    total = len(valid_rows)

    return {
        key: round((value / total) * 100, 2)
        for key, value in counts.items()
    }