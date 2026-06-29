"""Abstract drafter: proposes candidate draft tokens for the verifier to check."""
from abc import ABC, abstractmethod


class Drafter(ABC):
    name: str = "base"
    model_free: bool = False

    @abstractmethod
    def get_draft_token(self, prompt_token, n_token, user_id=None, hidden_state=None):
        """prompt_token: List[int] context, n_token: how many to propose.
        -> List[List[int]] candidate draft sequences."""

    @abstractmethod
    def build_datastore(self, token) -> bool:
        """Populate a retrieval datastore from a token stream. -> success."""

    @abstractmethod
    def describe_datastore(self):
        """-> (peak_data: List, total_data: int)."""
