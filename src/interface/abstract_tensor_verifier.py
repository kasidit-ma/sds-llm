"""Tensor verifier contract. Stub — Milestone 5 (verify tensor drafts)."""

from abc import ABC, abstractmethod
from typing import Any


class AbstractTensorVerifier(ABC):
    @abstractmethod
    def verify(self, target_suffix: Any, draft: Any) -> int:
        """Count accepted tokens of a tensor draft."""
        raise NotImplementedError("Milestone M6 — see plan")

    @abstractmethod
    def verify_best(self, target_suffix: Any, drafts: Any) -> tuple[int, int]:
        """Best ``(index, n_accepted)`` among tensor candidate drafts."""
        raise NotImplementedError("Milestone M6 — see plan")
