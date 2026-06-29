"""Bundle all workload features for one token stream into a feature dict. Stub — Milestone 2."""

from collections.abc import Sequence
from typing import Any


def characterize(tokens: Sequence[int], cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Return {heaps_beta, pbe_p90_at_1/2/4, repetition_rate, ...}."""
    raise NotImplementedError("Milestone M2 — see plan")
