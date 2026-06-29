"""Step-by-step speculative-decoding simulator over a recorded token sequence.

Baseline (no drafter) advances one token per step. With a drafter, each step proposes a draft,
the verifier accepts its matching prefix, and the simulator commits accepted + 1 tokens (the +1
is the verifier's own correct next token, which a real verifier produces for free).
"""

from collections.abc import Sequence
from typing import cast

from interface.abstract_drafter import AbstractDrafter
from interface.abstract_playback import AbstractPlayback, StepLog
from verifier.greedy_verifier import GreedyVerifier


class SpeculativePlayback(AbstractPlayback):
    def __init__(self, verifier: GreedyVerifier | None = None) -> None:
        self.verifier = verifier or GreedyVerifier()

    def run(
        self,
        target_tokens: Sequence[int],
        drafter: AbstractDrafter | None = None,
        K: int = 10,
        mode: str = "depth",
    ) -> StepLog:
        target = list(target_tokens)
        total = len(target)

        if drafter is None:
            return StepLog(total_tokens=total, steps=total)

        log = StepLog(total_tokens=total, steps=0)
        pos = 0
        while pos < total:
            # Drafter sees what is committed; its first token is a guess for target[pos].
            draft = drafter.propose(target[:pos], K, mode)
            suffix = target[pos:]

            if mode == "width":
                candidates = cast("list[list[int]]", draft)
                _, accepted = self.verifier.verify_best(suffix, candidates)
                # All candidates are verified in parallel, so all proposed tokens count as drafted
                # (not just the winning branch) — keeps acceptance_rate comparable with depth.
                drafted = sum(len(c) for c in candidates)
            else:
                chain = cast("list[int]", draft)
                accepted = self.verifier.verify(suffix, chain)
                drafted = len(chain)

            log.drafted += drafted
            log.accepted += accepted
            log.rejected += max(drafted - accepted, 0)
            log.per_step_accepted.append(accepted)
            log.steps += 1
            pos += accepted + 1  # accepted draft tokens + 1 free correct token (the baseline token)

        return log
