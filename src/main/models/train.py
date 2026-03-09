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
from sklearn.model_selection import train_test_split
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

    The function splits the dataset into train/test sets, standardizes features,
    trains a random forest, evaluates it on the test set, and packages the
    results into a ModelArtifacts object.

    Args:
        df: Feature-engineered dataset containing date_col and target_col.
        target_col: Name of the regression target column.
        date_col: Name of the date column to exclude from features.
        test_size: Fraction of samples reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A ModelArtifacts object containing the trained model, scaler, MAE,
        error standard deviation, and accuracy-band percentages.
    """

    features = df.drop(columns=[date_col, target_col])
    y = df[target_col].to_numpy()

    features_train, features_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    model = RandomForestRegressor(random_state=random_state)
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
    """Save trained model artifacts to disk.

    Writes three files:
      - model_{tag}.joblib: Trained sklearn model.
      - scaler_{tag}.joblib: Trained scaler used for feature transforms.
      - meta_{tag}.json: Evaluation metadata (MAE, std, accuracy bands).

    Args:
        artifacts: Trained model artifacts to persist.
        models_dir: Directory where artifacts will be written.
        tag: Identifier used to name the saved files (e.g., location-based tag).

    Returns:
        None
    """
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
    """Load trained model artifacts from disk.

    Args:
        models_dir: Directory containing previously saved artifacts.
        tag: Identifier used to name the saved files (e.g., location-based tag).

    Returns:
        A ModelArtifacts object reconstructed from saved files.

    Raises:
        FileNotFoundError: If any required artifact files are missing.
    """
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
