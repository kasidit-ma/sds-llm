"""Abstract verifier: accepts/rejects drafter proposals against the target LLM."""
from abc import ABC, abstractmethod


class Verifier(ABC):
    @abstractmethod
    def verify_with_history(self, prompt_token, draft_token):
        """prompt_token: List[int] context, draft_token: List[List[int]] candidates.
        -> List[List[int]] accepted tokens (matched prefix + 1 bonus target token)."""

    @abstractmethod
    def verify_with_llm(self, token) -> bool:
        """Check a token stream is the target's own greedy continuation. -> valid."""
