"""Bundle all workload features for one token stream into a feature dict."""

from collections.abc import Sequence
from typing import Any

from workload.heaps import heaps_beta
from workload.pbe import pbe_stats_at_k
from workload.repetition import repetition_rate


def characterize(tokens: Sequence[int], cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Return {heaps_beta, pbe_p90_at_1/2/4, pbe_mean_at_1/2/4, repetition_rate}."""
    cfg = cfg or {}
    steps = cfg.get("pbe_steps", [1, 2, 4])
    prefix_len = cfg.get("pbe_prefix_len", 3)
    pbe = {k: pbe_stats_at_k(tokens, k, prefix_len) for k in steps}  # one scan per k
    return {
        "heaps_beta": heaps_beta(tokens),
        "repetition_rate": repetition_rate(tokens, n=3),  # n matches drafter order (ngram.yaml)
        **{f"pbe_p90_at_{k}": p90 for k, (p90, _) in pbe.items()},
        **{f"pbe_mean_at_{k}": mean for k, (_, mean) in pbe.items()},
    }
