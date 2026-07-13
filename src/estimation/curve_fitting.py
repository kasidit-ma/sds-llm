"""Fit the acceptance/speedup model to benchmark ground truth (benchmark_results.csv)."""

import csv
import math
from typing import Any

from estimation.acceptance_model import AcceptanceModel
from estimation.evaluator import evaluate
from estimation.speedup_estimator import estimated_speedup


def _features(row: dict[str, str], scope: str) -> list[float]:
    return [
        float(row[f"{scope}_repetition_rate"]),
        float(row[f"{scope}_heaps_beta"]),
        math.log1p(float(row[f"{scope}_pbe_p90_at_1"])),
        math.log1p(float(row[f"{scope}_pbe_p90_at_2"])),
        math.log1p(float(row[f"{scope}_pbe_p90_at_4"])),
        # config features — speedup varies far more with n/mode than across datasets,
        # so a workload-only model can never fit the grid (R² < 0 without these)
        float(row["n"]),
        1.0 if row["mode"] == "depth" else 0.0,
    ]


def fit_linear_speedup(
    csv_path: str, scope: str = "ctx", mode: str = "depth", n: int = 3
) -> dict[str, Any]:
    """Direct linear fit ``speedup ≈ a + b·heaps_beta + c·pbe_mean@1``, one row per dataset.

    The simple, explainable baseline model (slide version) next to the sigmoid
    acceptance model of ``fit_from_benchmark``.
    """
    import numpy as np

    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["mode"] == mode and int(r["n"]) == n]
    if not rows:
        raise ValueError(f"no rows for mode={mode} n={n} in {csv_path}")
    X = np.array([[1.0, float(r[f"{scope}_heaps_beta"]), float(r[f"{scope}_pbe_mean_at_1"])]
                  for r in rows])
    y = np.array([float(r["actual_speedup"]) for r in rows])
    a, b, c = (float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0])
    pred = X @ np.array([a, b, c])
    table = [{"dataset": r["dataset"], "beta": float(X[i, 1]), "pbe_mean_at_1": float(X[i, 2]),
              "speedup": float(y[i]), "predict": float(pred[i])} for i, r in enumerate(rows)]
    formula = f"speedup = {a:.3f} + {b:.3f}*beta + {c:.3f}*pbe_mean@1"
    return {"scope": scope, "mode": mode, "n": n, "a": a, "b": b, "c": c,
            "formula": formula, "table": table, **evaluate(list(pred), list(y))}


def fit_from_benchmark(csv_path: str, scope: str = "ctx") -> dict[str, Any]:
    """Fit on ``ctx`` features (observable before generation — deployable) or ``gen``
    (computed on the generated text itself — the predictability upper bound)."""
    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["mode"] != "baseline"]  # baseline has no drafts
    if not rows:
        raise ValueError(f"no speculative rows in {csv_path}")
    model = AcceptanceModel().fit(
        [_features(r, scope) for r in rows], [float(r["acceptance_rate"]) for r in rows]
    )
    assert model.theta is not None
    predicted = [
        estimated_speedup(model.predict(_features(r, scope)), int(r["K"])) for r in rows
    ]
    actual = [float(r["actual_speedup"]) for r in rows]
    # ponytail: no leave-one-out — add once the benchmark CSV spans multiple datasets
    return {"scope": scope, "theta": [float(t) for t in model.theta], "n_rows": len(rows),
            "predicted": predicted, "actual": actual, **evaluate(predicted, actual)}
