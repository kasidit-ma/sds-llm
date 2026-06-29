"""Tensor-backed greedy verifier. Stub — Milestone 5 (numpy)."""

from typing import Any

from interface.abstract_tensor_verifier import AbstractTensorVerifier


class TensorGreedyVerifier(AbstractTensorVerifier):
    def verify(self, target_suffix: Any, draft: Any) -> int:
        raise NotImplementedError("Milestone M5 — see plan")

    def verify_best(self, target_suffix: Any, drafts: Any) -> tuple[int, int]:
        raise NotImplementedError("Milestone M5 — see plan")
