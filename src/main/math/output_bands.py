"""Prediction-to-output conversion and uncertainty bands.

This module converts predicted GHI values into estimated energy output (kWh)
given array area and panel efficiency, and builds low/high output intervals.
"""

from __future__ import annotations

from typing import Iterable, Tuple


def _to_output_kwh(
    array_area_m2: float, efficiency: float, ghi_wh_m2: float
) -> float:
    """Convert a GHI value into estimated energy output in kWh.

    Args:
        array_area_m2: Solar array area in square meters.
        efficiency: Panel efficiency as a fraction (0-1).
        ghi_wh_m2: Global Horizontal Irradiance value in Wh/m^2.

    Returns:
        Estimated energy output in kWh.

    Notes:
        This conversion assumes the input GHI is expressed in Wh/m^2 for the
        relevant time window. The division by 1000 converts Wh to kWh.
    """

    return (array_area_m2 * efficiency * ghi_wh_m2) / 1000.0


def output_bands(
    *,
    array_area_m2: float,
    efficiency: float,
    ghi_pred: Iterable[float],
    band: float,
) -> list[Tuple[float, float]]:
    """Compute output (kWh) intervals using a symmetric ± band around GHI.

    Args:
        array_area_m2: Solar array area in square meters.
        efficiency: Panel efficiency as a fraction (0-1).
        ghi_pred: Iterable of predicted GHI values.
        band: Band width applied to each prediction
            (low = pred - band, high = pred + band).

    Returns:
        A list of (low_kwh, high_kwh) tuples for each predicted value.
    """

    results: list[Tuple[float, float]] = []
    for ghi in ghi_pred:
        low = _to_output_kwh(array_area_m2, efficiency, ghi - band)
        high = _to_output_kwh(array_area_m2, efficiency, ghi + band)
        results.append((low, high))
    return results


def output_bands_std(
    *,
    array_area_m2: float,
    efficiency: float,
    ghi_pred: Iterable[float],
    std: float,
) -> dict[str, list[Tuple[float, float]]]:
    """Compute output (kWh) intervals for ±1σ, ±2σ, and ±3σ bands.

    Args:
        array_area_m2: Solar array area in square meters.
        efficiency: Panel efficiency as a fraction (0-1).
        ghi_pred: Iterable of predicted GHI values.
        std: Standard deviation of prediction error in the same units as GHI.

    Returns:
        A dict with keys "1std", "2std", "3std", each mapping to a list of
        (low_kwh, high_kwh) tuples.
    """

    ghi_pred = list(ghi_pred)
    return {
        "1std": output_bands(
            array_area_m2=array_area_m2,
            efficiency=efficiency,
            ghi_pred=ghi_pred,
            band=std,
        ),
        "2std": output_bands(
            array_area_m2=array_area_m2,
            efficiency=efficiency,
            ghi_pred=ghi_pred,
            band=2 * std,
        ),
        "3std": output_bands(
            array_area_m2=array_area_m2,
            efficiency=efficiency,
            ghi_pred=ghi_pred,
            band=3 * std,
        ),
    }
