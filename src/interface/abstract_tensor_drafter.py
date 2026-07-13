"""Tensor drafter contract. Stub — Milestone 5 (numpy-backed n-gram drafting)."""

from abc import ABC, abstractmethod
from typing import Any


class AbstractTensorDrafter(ABC):
    @abstractmethod
    def build_datastore(self, corpus_tokens: Any) -> None:
        """Index a tensor of token ids."""
        raise NotImplementedError("Milestone M6 — see plan")

    @abstractmethod
    def propose(self, context: Any, K: int, mode: str = "depth") -> Any:
        """Propose draft tokens as a tensor."""
        raise NotImplementedError("Milestone M6 — see plan")
