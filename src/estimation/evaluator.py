"""Evaluate predicted vs actual speedup (MAE / RMSE / R²)."""

import math
from collections.abc import Sequence


def evaluate(predicted: Sequence[float], actual: Sequence[float]) -> dict[str, float]:
    if not actual:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    errors = [p - a for p, a in zip(predicted, actual, strict=True)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean = sum(actual) / len(actual)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((a - mean) ** 2 for a in actual)
    return {"mae": mae, "rmse": rmse, "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0}
