"""Model inference helpers.

This module contains helpers that transform feature tables and produce model
predictions for GHI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_ghi(
    model, scaler, forecast_df: pd.DataFrame, *, date_col: str = "date"
) -> np.ndarray:
    """Predict GHI values from a forecast feature DataFrame.

    Args:
        model: Trained model object that implements predict(features).
        scaler: Feature scaler/transformer object that 
            implements transform(features).
        forecast_df: Feature DataFrame containing a date column 
            plus feature columns.
        date_col: Name of the date column to drop before scaling/prediction.

    Returns:
        A NumPy array of predicted GHI values aligned to the input rows.

    Raises:
        KeyError: If date_col is not present in forecast_df.
    """
    features = forecast_df.drop(columns=[date_col])
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)
