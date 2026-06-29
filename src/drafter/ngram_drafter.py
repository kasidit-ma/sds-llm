"""N-gram drafter: predict continuations from a frequency datastore.

Datastore maps each (n-1)-token key to a Counter of the tokens that followed it. Depth draft
greedily extends one chain; width draft branches into several candidate chains.
"""

from collections import Counter
from collections.abc import Sequence

from interface.abstract_drafter import AbstractDrafter, Draft, WidthDraft


class NGramDrafter(AbstractDrafter):
    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.store: dict[tuple[int, ...], Counter[int]] = {}

    def build_datastore(self, corpus_tokens: Sequence[int]) -> None:
        """Count, for every (n-1)-gram in the corpus, the distribution of the next token."""
        self.store = {}
        k = self.n - 1
        tokens = list(corpus_tokens)
        for i in range(len(tokens) - k):
            key = tuple(tokens[i : i + k])
            nxt = tokens[i + k]
            self.store.setdefault(key, Counter())[nxt] += 1

    def _key(self, context: Sequence[int]) -> tuple[int, ...]:
        k = self.n - 1
        return tuple(context[-k:]) if k else ()

    def _ranked_next(self, key: tuple[int, ...]) -> list[int]:
        """Candidate next tokens for ``key``, most frequent first (deterministic tie-break)."""
        counter = self.store.get(key)
        if not counter:
            return []
        return [tok for tok, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]

    def _best_next(self, key: tuple[int, ...]) -> int | None:
        ranked = self._ranked_next(key)
        return ranked[0] if ranked else None

    def _extend(self, context: list[int], length: int) -> list[int]:
        """Greedily extend ``context`` by up to ``length`` most-frequent tokens."""
        chain: list[int] = []
        cur = context
        for _ in range(length):
            nxt = self._best_next(self._key(cur))
            if nxt is None:
                break
            chain.append(nxt)
            cur = cur + [nxt]
        return chain

    def propose(self, context: Sequence[int], K: int, mode: str = "depth") -> Draft | WidthDraft:
        if K <= 0 or not self.store:
            return []
        ctx = list(context)
        if mode == "depth":
            return self._extend(ctx, K)
        if mode == "width":
            return self._propose_width(ctx, K)
        raise ValueError(f"unknown mode: {mode!r}")

    def _propose_width(self, ctx: list[int], K: int, S: int = 5, T: int = 2) -> WidthDraft:
        """Top-S next tokens, each extended up to T tokens. Budget K caps total drafted tokens."""
        first = self._ranked_next(self._key(ctx))[:S]
        seqs: WidthDraft = []
        budget = K
        for tok in first:
            if budget <= 0:
                break
            take = min(T, budget)
            seq = [tok] + self._extend(ctx + [tok], take - 1)
            seqs.append(seq)
            budget -= len(seq)
        return seqs
