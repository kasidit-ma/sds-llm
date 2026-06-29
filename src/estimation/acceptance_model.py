"""Predict per-token acceptance probability from workload features. Stub — Milestone 4.

p_accept = sigmoid(theta . features), features = [repetition_rate, beta, log1p(PBE@1/2/4)].
"""

from typing import Any


class AcceptanceModel:
    def fit(self, features: Any, acceptance: Any) -> "AcceptanceModel":
        raise NotImplementedError("Milestone M4 — see plan")

    def predict(self, features: Any) -> float:
        raise NotImplementedError("Milestone M4 — see plan")
