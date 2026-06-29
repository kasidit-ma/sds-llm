"""Load tokenized benchmark samples from workload results (pre-tokenized JSON).
Reuses workload/results/*.json so no re-download or re-tokenize is needed."""
import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent.parent / "workload" / "results"


def load_dataset_tokens(dataset_id: str) -> list[list[int]]:
    """Load pre-tokenized docs for one dataset from workload/results/<id>.json.
    The JSON stores the full flat token stream (no per-doc split) — we window it
    into fixed-length chunks suitable for spec-bench evaluation."""
    raise NotImplementedError("workload/results/*.json stores aggregate stats, "
                              "not raw token lists — need to re-tokenize or store "
                              "per-doc token lists; see workload/run.py for how to do it")


def iter_samples(token_stream: list[int], context_len: int = 64, stride: int = 32):
    """Slide a window over a flat token stream -> (prompt, continuation) pairs.
    prompt = context_len tokens, continuation = next context_len tokens."""
    n = len(token_stream)
    i = 0
    while i + context_len * 2 <= n:
        yield token_stream[i:i + context_len], token_stream[i + context_len:i + context_len * 2]
        i += stride


def list_datasets() -> list[str]:
    return sorted(p.stem for p in RESULTS.glob("*.json"))
