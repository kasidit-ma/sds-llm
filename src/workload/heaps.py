"""Heaps' law vocabulary-growth exponent.

Fit V = K * n^beta over growing prefixes: regress log(V) on log(n) and return the slope.
Higher beta => vocabulary keeps growing => more diverse => harder to draft.
"""

import math
import statistics
from collections.abc import Sequence


def heaps_beta(tokens: Sequence[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    stride = max(1, len(tokens) // 1000)  # ponytail: cap fit points at ~1000, O(N) walk
    seen: set[int] = set()
    xs: list[float] = []
    ys: list[float] = []
    for i, tok in enumerate(tokens, start=1):
        seen.add(tok)
        if i % stride == 0 or i == len(tokens):
            xs.append(math.log(i))
            ys.append(math.log(len(seen)))
    if len(set(xs)) < 2:
        return 0.0
    return statistics.linear_regression(xs, ys).slope
