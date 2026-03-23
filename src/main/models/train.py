"""Model training and artifact persistence.

This module trains the project's baseline regression model and provides helpers
to save/load trained artifacts (model, scaler, and evaluation metadata).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from main.models.metrics import compute_accuracy_bands, compute_error_stats
from main.schemas import ModelArtifacts


def train_random_forest(
    df: pd.DataFrame,
    *,
    target_col: str = "GHI",
    date_col: str = "date",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelArtifacts:
    """Train a RandomForestRegressor on the engineered dataset.

    Uses a chronological split instead of a random split so evaluation is closer
    to real forward prediction behavior.
    """

    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col, target_col]).sort_values(date_col)
    work_df = work_df.reset_index(drop=True)

    if len(work_df) < 10:
        raise RuntimeError("Not enough training rows after cleaning.")

    split_idx = max(1, int(len(work_df) * (1.0 - test_size)))
    split_idx = min(split_idx, len(work_df) - 1)

    train_df = work_df.iloc[:split_idx].copy()
    test_df = work_df.iloc[split_idx:].copy()

    features_train = train_df.drop(columns=[date_col, target_col])
    features_test = test_df.drop(columns=[date_col, target_col])

    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        min_samples_split=6,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features_train_scaled, y_train)

    y_pred = model.predict(features_test_scaled)

    mae, err_std = compute_error_stats(y_test, y_pred)
    acc = compute_accuracy_bands(y_test, y_pred, mae, err_std)

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        mae=mae,
        error_std=err_std,
        accuracy_bands=acc,
    )


def save_artifacts(
    artifacts: ModelArtifacts, *, models_dir: Path, tag: str
) -> None:
    """Save trained model artifacts to disk."""
    model_path = models_dir / f"model_{tag}.joblib"
    scaler_path = models_dir / f"scaler_{tag}.joblib"
    meta_path = models_dir / f"meta_{tag}.json"

    joblib.dump(artifacts.model, model_path)
    joblib.dump(artifacts.scaler, scaler_path)

    meta = {
        "mae": artifacts.mae,
        "error_std": artifacts.error_std,
        "accuracy_bands": artifacts.accuracy_bands,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def load_artifacts(*, models_dir: Path, tag: str) -> ModelArtifacts:
    """Load trained model artifacts from disk."""
    model_path = models_dir / f"model_{tag}.joblib"
    scaler_path = models_dir / f"scaler_{tag}.joblib"
    meta_path = models_dir / f"meta_{tag}.json"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        mae=float(meta["mae"]),
        error_std=float(meta["error_std"]),
        accuracy_bands=dict(meta["accuracy_bands"]),
    )