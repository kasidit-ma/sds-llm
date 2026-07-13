"""Verifier contract: count how many draft tokens match the target."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class AbstractVerifier(ABC):
    @abstractmethod
    def verify(self, target_suffix: Sequence[int], draft: Sequence[int]) -> int:
        """Number of leading draft tokens that match ``target_suffix`` (stop at first mismatch)."""

    @abstractmethod
    def verify_best(
        self, target_suffix: Sequence[int], drafts: Sequence[Sequence[int]]
    ) -> tuple[int, int]:
        """Among candidate ``drafts``, return ``(best_index, n_accepted)`` of the longest match."""
