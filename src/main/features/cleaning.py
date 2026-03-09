"""Data cleaning utilities.

This module contains lightweight helpers that remove rows which would break
feature engineering, training, or prediction.
"""

from __future__ import annotations

import pandas as pd


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values and reset the index.

    This is intentionally strict: any missing values are removed to avoid
    downstream failures during scaling, training, or inference.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with all-NA rows removed and a clean 0..N-1 index.
    """
    return df.dropna().reset_index(drop=True)
