"""N-gram repetition rate.

repetition_rate@n = 1 - unique_ngram_count@n / total_ngram_count@n.
"""

from collections.abc import Sequence


def repetition_rate(tokens: Sequence[int], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    unique = len({tuple(tokens[i : i + n]) for i in range(total)})
    return 1.0 - unique / total
