"""Predict per-token acceptance probability from workload features.

p_accept = sigmoid(theta . features), features = [repetition_rate, beta, log1p(PBE@1/2/4)].
Fit is least squares on logit-transformed acceptance (linear in theta) — no scipy/sklearn.
"""

import math
from collections.abc import Sequence

import numpy as np


class AcceptanceModel:
    def __init__(self) -> None:
        self.theta: np.ndarray | None = None

    def fit(
        self, features: Sequence[Sequence[float]], acceptance: Sequence[float]
    ) -> "AcceptanceModel":
        X = np.column_stack([np.ones(len(features)), np.asarray(features, dtype=float)])
        y = np.clip(np.asarray(acceptance, dtype=float), 1e-6, 1 - 1e-6)  # logit-safe
        self.theta = np.linalg.lstsq(X, np.log(y / (1 - y)))[0]
        return self

    def predict(self, features: Sequence[float]) -> float:
        if self.theta is None:
            raise ValueError("call fit() first")
        z = float(self.theta @ np.asarray([1.0, *features], dtype=float))
        return 1.0 / (1.0 + math.exp(-z))
