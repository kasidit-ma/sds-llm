"""Choose a decoding configuration from workload features. Stub — Milestone 4.

Returns e.g. {"use_speculative": True, "mode": "depth", "n": 4, "K": 10, "estimated_speedup": 1.8}.
"""

from typing import Any


def decide(features: dict[str, float], estimated_speedup: float) -> dict[str, Any]:
    raise NotImplementedError("Milestone M4 — see plan")
