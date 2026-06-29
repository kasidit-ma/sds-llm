"""Prefix branching entropy / branching counts (PBE_P90@k). Stub — Milestone 2.

branching_count@k(prefix) = number of unique k-token continuations seen after prefix.
PBE_P90@k = 90th percentile of branching_count@k over all prefixes.
"""

from collections.abc import Sequence


def branching_count_at_k(
    tokens: Sequence[int], k: int, prefix_len: int
) -> dict[tuple[int, ...], int]:
    raise NotImplementedError("Milestone M2 — see plan")


def pbe_p90_at_k(tokens: Sequence[int], k: int, prefix_len: int) -> float:
    raise NotImplementedError("Milestone M2 — see plan")
