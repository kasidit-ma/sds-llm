"""Playback contract: replay a target token sequence, optionally with a drafter."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from interface.abstract_drafter import AbstractDrafter


@dataclass
class StepLog:
    """Raw counters produced by a playback run; turned into ratios by ``metrics``."""

    total_tokens: int
    steps: int
    drafted: int = 0
    accepted: int = 0
    rejected: int = 0
    per_step_accepted: list[int] = field(default_factory=list)


class AbstractPlayback(ABC):
    @abstractmethod
    def run(
        self,
        target_tokens: Sequence[int],
        drafter: AbstractDrafter | None = None,
        K: int = 10,
        mode: str = "depth",
    ) -> StepLog:
        """Replay ``target_tokens`` one step at a time.

        ``drafter=None`` is the baseline (one token per step). With a drafter, each step proposes
        a draft, accepts the matching prefix plus one free correct token, and advances.
        """
