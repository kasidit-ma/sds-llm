"""Choose a decoding configuration from workload features.

Returns e.g. {"use_speculative": True, "mode": "depth", "n": 4, "K": 10, "estimated_speedup": 1.8}.
"""

from typing import Any

_DEFAULT_N, _DEFAULT_K = 3, 10  # configs/ngram.yaml defaults


def decide(features: dict[str, float], estimated_speedup: float) -> dict[str, Any]:
    # low branching -> one long chain suffices (depth); high branching -> hedge (width)
    mode = "depth" if features.get("pbe_p90_at_1", 0.0) <= 2.0 else "width"
    return {
        "use_speculative": estimated_speedup > 1.0,
        "mode": mode,
        "n": _DEFAULT_N,
        "K": _DEFAULT_K,
        "estimated_speedup": estimated_speedup,
    }
