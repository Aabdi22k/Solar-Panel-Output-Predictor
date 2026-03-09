"""Shared data containers used across training and prediction.

These lightweight dataclasses define the shapes of artifacts and outputs
passed between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelArtifacts:
    """Bundles trained artifacts and evaluation stats.

    Attributes:
        model: Trained regression estimator.
        scaler: Feature transformer used during training.
        mae: Mean Absolute Error on the evaluation set.
        error_std: Standard deviation of prediction errors.
        accuracy_bands: Precomputed coverage/accuracy bands.
    """

    model: object
    scaler: object
    mae: float
    error_std: float
    accuracy_bands: Dict[str, float]


@dataclass(frozen=True)
class PredictionResult:
    """Stores predictions and uncertainty intervals.

    Attributes:
        dates: Dates aligned to each prediction.
        ghi_pred: Predicted GHI values.
        bands_mae: (low, high) bounds computed using MAE.
        bands_1std: (low, high) bounds computed using ±1 std.
        bands_2std: (low, high) bounds computed using ±2 std.
        bands_3std: (low, high) bounds computed using ±3 std.
    """

    dates: pd.Series
    ghi_pred: np.ndarray
    bands_mae: list[Tuple[float, float]]
    bands_1std: list[Tuple[float, float]]
    bands_2std: list[Tuple[float, float]]
    bands_3std: list[Tuple[float, float]]
