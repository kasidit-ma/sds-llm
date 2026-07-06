"""Prefix branching entropy / branching counts (PBE_P90@k).

branching_count@k(prefix) = number of unique k-token continuations seen after prefix.
PBE_P90@k = 90th percentile of branching_count@k over all prefixes.
"""

from collections import defaultdict
from collections.abc import Sequence


def branching_count_at_k(
    tokens: Sequence[int], k: int, prefix_len: int
) -> dict[tuple[int, ...], int]:
    continuations: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for i in range(len(tokens) - prefix_len - k + 1):
        prefix = tuple(tokens[i : i + prefix_len])
        continuations[prefix].add(tuple(tokens[i + prefix_len : i + prefix_len + k]))
    return {prefix: len(conts) for prefix, conts in continuations.items()}


def pbe_stats_at_k(tokens: Sequence[int], k: int, prefix_len: int) -> tuple[float, float]:
    """(P90, mean) of branching_count@k from a single pass over the stream."""
    counts = sorted(branching_count_at_k(tokens, k, prefix_len).values())
    if not counts:
        return 0.0, 0.0
    p90 = float(counts[int(0.9 * (len(counts) - 1))])  # nearest-rank P90
    return p90, sum(counts) / len(counts)


def pbe_p90_at_k(tokens: Sequence[int], k: int, prefix_len: int) -> float:
    return pbe_stats_at_k(tokens, k, prefix_len)[0]


def pbe_mean_at_k(tokens: Sequence[int], k: int, prefix_len: int) -> float:
    return pbe_stats_at_k(tokens, k, prefix_len)[1]
