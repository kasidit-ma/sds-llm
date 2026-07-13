"""Greedy verifier: accept the longest matching prefix of a draft against the target."""

from collections.abc import Sequence

from interface.abstract_verifier import AbstractVerifier


class GreedyVerifier(AbstractVerifier):
    def verify(self, target_suffix: Sequence[int], draft: Sequence[int]) -> int:
        n = 0
        for t, d in zip(target_suffix, draft, strict=False):
            if t != d:
                break
            n += 1
        return n

    def verify_best(
        self, target_suffix: Sequence[int], drafts: Sequence[Sequence[int]]
    ) -> tuple[int, int]:
        best_idx, best_acc = 0, -1
        for i, draft in enumerate(drafts):
            acc = self.verify(target_suffix, draft)
            if acc > best_acc:
                best_idx, best_acc = i, acc
        if best_acc < 0:  # no candidates
            return 0, 0
        return best_idx, best_acc
