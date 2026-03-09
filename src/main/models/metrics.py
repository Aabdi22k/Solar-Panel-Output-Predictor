"""Model evaluation metrics.

This module provides simple error statistics and "band accuracy" metrics that
measure the percentage of ground-truth points falling within prediction bands.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def mae_band_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, band: float
) -> float:
    """Compute the percentage of points within a ± band of the prediction.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        band: Non-negative tolerance band around predictions.

    Returns:
        Percentage (0-100) of points where
            y_true lies within [y_pred - band, y_pred + band].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ok = (y_true >= (y_pred - band)) & (y_true <= (y_pred + band))
    return float(ok.mean() * 100.0)


def compute_error_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """Compute MAE and the standard deviation of absolute errors.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        A tuple (mae, std), where:
          - mae is mean absolute error.
          - std is the standard deviation of absolute errors.
    """
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    mae = float(errors.mean())
    std = float(errors.std())
    return mae, std


def compute_accuracy_bands(
    y_true: np.ndarray, y_pred: np.ndarray, mae: float, std: float
) -> Dict[str, float]:
    """Compute accuracy percentages for MAE and standard deviation bands.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        mae: Mean absolute error to use as the MAE band width.
        std: Standard deviation to use as the 1σ band width.

    Returns:
        A dict containing:
          - "MAE": Percent within ±MAE
          - "1std": Percent within ±1σ
          - "2std": Percent within ±2σ
          - "3std": Percent within ±3σ
    """
    return {
        "MAE": mae_band_accuracy(y_true, y_pred, mae),
        "1std": mae_band_accuracy(y_true, y_pred, std),
        "2std": mae_band_accuracy(y_true, y_pred, 2 * std),
        "3std": mae_band_accuracy(y_true, y_pred, 3 * std),
    }
