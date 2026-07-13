"""Drafter contract: build an n-gram datastore, then propose draft tokens."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

# A "depth" draft is one chain of tokens; a "width" draft is several candidate chains.
Draft = list[int]
WidthDraft = list[list[int]]


class AbstractDrafter(ABC):
    @abstractmethod
    def build_datastore(self, corpus_tokens: Sequence[int]) -> None:
        """Index the corpus so future continuations can be looked up."""

    @abstractmethod
    def propose(self, context: Sequence[int], K: int, mode: str = "depth") -> Draft | WidthDraft:
        """Propose draft tokens following ``context`` (≤ ``K`` tokens of budget).

        ``mode="depth"`` returns one chain (``list[int]``); ``mode="width"`` returns several
        candidate chains (``list[list[int]]``). Returns an empty list when nothing is known.
        """
