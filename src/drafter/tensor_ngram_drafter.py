"""Tensor-backed n-gram drafter. Stub — Milestone 5 (numpy)."""

from typing import Any

from interface.abstract_tensor_drafter import AbstractTensorDrafter


class TensorNGramDrafter(AbstractTensorDrafter):
    def build_datastore(self, corpus_tokens: Any) -> None:
        raise NotImplementedError("Milestone M5 — see plan")

    def propose(self, context: Any, K: int, mode: str = "depth") -> Any:
        raise NotImplementedError("Milestone M5 — see plan")
