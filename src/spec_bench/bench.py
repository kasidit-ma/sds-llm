"""Spec-decoding benchmark harness: drafter → verifier → measure actual speedup.

Two modes:
- simulation (GreedyVerifier): compares against ground-truth tokens, counts steps,
  no GPU needed. speedup = baseline_steps / spec_steps.
- real (LLMVerifier): wall-clock timing with a real model (requires GPU).

Usage:
    python bench.py   # runs simulation self-check, no GPU needed
"""
import time

def _get_iter_samples():
    try:
        from data import iter_samples  # flat import when run from this dir
        return iter_samples
    except ModuleNotFoundError:
        return None  # caller (run.py) patches bench.iter_samples directly


# ── simulation mode (GreedyVerifier) ──────────────────────────────────────────

def sim_baseline_steps(samples: list[tuple]) -> int:
    """Count autoregressive steps (1 per output token) across all samples."""
    return sum(len(cont) for _, cont in samples)


def sim_spec(drafter, verifier, samples: list[tuple], draft_len: int = 4) -> tuple:
    """Simulation: drafter proposes, GreedyVerifier checks against ground truth.
    Returns (spec_steps, mean_accepted_per_step)."""
    total_accepted, total_steps = 0, 0
    for prompt, continuation in samples:
        complete = list(prompt) + list(continuation)
        context = list(prompt)
        while len(context) < len(complete):
            draft = drafter.get_draft_token(context, draft_len)
            accepted = verifier.verify_with_history(context, draft,
                                                    complete_tokens=complete)[0]
            if not accepted:  # safety: always advance at least 1
                context.append(complete[len(context)])
                total_steps += 1
                continue
            context.extend(accepted)
            total_accepted += len(accepted)
            total_steps += 1
    return total_steps, total_accepted / max(total_steps, 1)


def benchmark_sim(drafter, verifier, token_stream: list[int],
                  context_len: int = 64, stride: int = 32,
                  max_samples: int = 200, draft_len: int = 4,
                  _iter_samples=None) -> dict:
    """Simulation benchmark (no GPU). speedup = baseline_steps / spec_steps."""
    _iter = _iter_samples or _get_iter_samples()
    drafter.build_datastore(token_stream)
    samples = list(_iter(token_stream, context_len, stride))[:max_samples]
    baseline_steps = sim_baseline_steps(samples)
    spec_steps, mean_accepted = sim_spec(drafter, verifier, samples, draft_len)
    return {
        "mode": "simulation",
        "baseline_steps": baseline_steps,
        "spec_steps": spec_steps,
        "actual_speedup": baseline_steps / spec_steps if spec_steps > 0 else None,
        "mean_accepted_per_step": mean_accepted,
        "n_samples": len(samples),
    }


# ── real mode (LLMVerifier, requires GPU) ─────────────────────────────────────

def run_baseline_wall(verifier, samples: list[tuple]) -> float:
    """Wall-clock baseline: 1 LLM forward pass per output token."""
    total_tokens, t0 = 0, time.perf_counter()
    for prompt, continuation in samples:
        for i in range(len(continuation)):
            verifier.verify_with_llm(list(prompt) + list(continuation[:i + 1]))
            total_tokens += 1
    return (time.perf_counter() - t0) / max(total_tokens, 1)


def run_spec_wall(drafter, verifier, samples: list[tuple], draft_len: int = 4) -> tuple:
    """Wall-clock spec: drafter proposes, LLM verifier accepts in batch."""
    total_tokens, total_passes, t0 = 0, 0, time.perf_counter()
    for prompt, continuation in samples:
        context = list(prompt)
        gen = 0
        while gen < len(continuation):
            draft = drafter.get_draft_token(context, draft_len)
            accepted = verifier.verify_with_history(context, draft)[0]
            context.extend(accepted)
            gen += len(accepted)
            total_tokens += len(accepted)
            total_passes += 1
    elapsed = time.perf_counter() - t0
    return elapsed / max(total_tokens, 1), total_tokens / max(total_passes, 1)


def benchmark_real(drafter, verifier, token_stream: list[int],
                   context_len: int = 64, stride: int = 32,
                   max_samples: int = 200, draft_len: int = 4) -> dict:
    """Real benchmark (requires LLMVerifier + GPU). speedup = T_base / T_spec."""
    drafter.build_datastore(token_stream)
    samples = list(iter_samples(token_stream, context_len, stride))[:max_samples]
    baseline_sec = run_baseline_wall(verifier, samples)
    spec_sec, mean_accepted = run_spec_wall(drafter, verifier, samples, draft_len)
    return {
        "mode": "real",
        "baseline_sec_per_token": baseline_sec,
        "spec_sec_per_token": spec_sec,
        "actual_speedup": baseline_sec / spec_sec if spec_sec > 0 else None,
        "mean_accepted_per_pass": mean_accepted,
        "n_samples": len(samples),
    }


if __name__ == "__main__":
    import sys, os, importlib.util
    here = os.path.dirname(os.path.abspath(__file__))

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    drafter_base = _load("drafter_base", os.path.join(here, "../drafter/base.py"))
    sys.modules["base"] = drafter_base
    NgramDrafter = _load("ngram", os.path.join(here, "../drafter/ngram.py")).NgramDrafter

    verifier_base = _load("verifier_base", os.path.join(here, "../verifier/base.py"))
    sys.modules["base"] = verifier_base
    GreedyVerifier = _load("greedy", os.path.join(here, "../verifier/greedy.py")).GreedyVerifier

    stream = list(range(10)) * 100   # 1000 tokens, repeating 0..9 cycle
    drafter = NgramDrafter(prefix_len=3)
    verifier = GreedyVerifier()
    result = benchmark_sim(drafter, verifier, stream, context_len=10, stride=5,
                           max_samples=20, draft_len=4)
    assert result["actual_speedup"] is not None
    assert result["actual_speedup"] > 1, f"expected speedup>1, got {result['actual_speedup']}"
    assert result["mean_accepted_per_step"] > 1
    print("bench sim self-check OK:", result)
