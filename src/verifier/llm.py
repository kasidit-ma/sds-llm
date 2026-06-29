"""Greedy target-LLM verifier (Qwen2.5-7B default). Needs torch+transformers+GPU
to run a real model, but the accept rule (`accept_greedy`) is pure and testable."""
from base import Verifier  # flat import: run self-check from this dir (matches workload/)


def accept_greedy(draft, target_argmax):
    """Greedy speculative-decoding accept rule.

    draft: List[int] proposed tokens. target_argmax: List[int] of len(draft)+1 —
    the target model's greedy token at each draft position plus one bonus slot.
    Accept the longest prefix where draft matches the target; at the first
    mismatch keep the target's (correct) token and stop. If all match, append the
    bonus token. Returns the accepted/corrected tokens (always >= 1 token)."""
    out = []
    for i, dt in enumerate(draft):
        out.append(target_argmax[i])      # target's greedy token here (== dt if matched)
        if dt != target_argmax[i]:
            return out                    # mismatch: corrected token, stop
    out.append(target_argmax[len(draft)])  # all accepted -> free bonus token
    return out


class LLMVerifier(Verifier):
    def __init__(self, model_name="Qwen/Qwen2.5-7B", device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device).eval()

    def _argmax(self, ids, start, n):
        """Greedy next-token argmax for n positions starting at logits index `start`."""
        with self.torch.no_grad():
            logits = self.model(self.torch.tensor([ids], device=self.device)).logits[0]
        return logits[start:start + n].argmax(-1).tolist()

    def verify_with_history(self, prompt_token, draft_token):
        # ponytail: verify the first candidate only; tree-verify is YAGNI until measured
        draft = draft_token[0] if draft_token else []
        ids = list(prompt_token) + list(draft)
        # logits[t] predicts token t+1; draft[0] is checked by logits[len(prompt)-1]
        argmax = self._argmax(ids, len(prompt_token) - 1, len(draft) + 1)
        return [accept_greedy(draft, argmax)]

    def verify_with_llm(self, token) -> bool:
        """True iff `token` is the model's own greedy continuation of token[:1]."""
        token = list(token)
        argmax = self._argmax(token, 0, len(token) - 1)
        return argmax == token[1:]


if __name__ == "__main__":
    # pure accept rule — no model needed
    assert accept_greedy([5, 6, 7], [5, 6, 9, 0]) == [5, 6, 9]   # mismatch at pos 2
    assert accept_greedy([5, 6], [5, 6, 7]) == [5, 6, 7]         # all match + bonus
    assert accept_greedy([8], [5, 0]) == [5]                     # mismatch at pos 0
    assert accept_greedy([], [4]) == [4]                         # empty draft -> bonus only
    print("accept_greedy OK")
