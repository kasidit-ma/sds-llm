"""Simulation verifier: compares draft against known ground-truth tokens.
No LLM or GPU needed — runs on any machine."""
from base import Verifier  # flat import: run self-check from this dir


class GreedyVerifier(Verifier):
    """Accept draft tokens that match ground truth exactly (greedy simulation).

    verify_with_history takes complete_tokens as the ground truth reference
    (not a real forward pass). Mirrors the spec-decode-greedy pattern so
    speedup can be measured as steps_baseline / steps_spec without a GPU.
    """

    def verify_with_history(self, prompt_token, draft_token, complete_tokens=None):
        """Compare draft against complete_tokens at the prompt position.

        complete_tokens: full ground-truth token stream (required for simulation).
        Returns List[List[int]]: accepted prefix + 1 recovery token (same shape as
        the LLM verifier, so bench.py works with both).
        """
        if complete_tokens is None:
            raise ValueError("GreedyVerifier requires complete_tokens for simulation")

        draft = draft_token[0] if draft_token else []
        prefix_len = len(prompt_token)
        accepted = []

        for i, dt in enumerate(draft):
            gt_idx = prefix_len + i
            if gt_idx < len(complete_tokens) and dt == complete_tokens[gt_idx]:
                accepted.append(dt)
            else:
                break

        # recovery token: first ground-truth token that wasn't in the accepted draft
        recovery_idx = prefix_len + len(accepted)
        if recovery_idx < len(complete_tokens):
            accepted.append(complete_tokens[recovery_idx])

        return [accepted]

    def verify_with_llm(self, token) -> bool:
        raise NotImplementedError("GreedyVerifier is simulation-only; use LLMVerifier for real inference")


if __name__ == "__main__":
    v = GreedyVerifier()
    gt = [0, 1, 2, 3, 4, 5, 6, 7]

    # draft fully matches → accept all + bonus
    assert v.verify_with_history([0, 1], [[2, 3, 4]], complete_tokens=gt) == [[2, 3, 4, 5]]
    # draft mismatches at pos 0 → recovery only
    assert v.verify_with_history([0, 1], [[9, 9]], complete_tokens=gt) == [[2]]
    # draft partially matches → accepted prefix + recovery
    assert v.verify_with_history([0, 1], [[2, 9]], complete_tokens=gt) == [[2, 3]]
    # empty draft → just recovery token
    assert v.verify_with_history([0, 1], [[]], complete_tokens=gt) == [[2]]
    print("greedy verifier OK")
