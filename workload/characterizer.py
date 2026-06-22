"""Tokenize a corpus and compute Heaps' beta + Prefix Branching Entropy."""
import logging
from collections import Counter
import numpy as np

log = logging.getLogger("workload")


def heaps(tokens: list[int]) -> dict:
    """Fit V = K * N^beta via log-log regression over logspace sample points."""
    N = len(tokens)
    pts = np.unique(np.logspace(1, np.log10(N), 30).astype(int))
    pts = pts[pts <= N]
    seen, Ns, Vs = set(), [], []
    last = 0
    for p in pts:
        for t in tokens[last:p]:
            seen.add(t)
        last = p
        Ns.append(p)
        Vs.append(len(seen))
    x, y = np.log(Ns), np.log(Vs)
    beta, logK = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (beta * x + logK)) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"beta": float(beta), "K": float(np.exp(logK)),
            "r_squared": float(r2), "vocab_size": len(set(tokens)),
            "token_count": N,
            "curve": {"N": [int(v) for v in Ns], "V": [int(v) for v in Vs]}}


def _stats(vals: list[int]) -> dict:
    a = np.array(vals)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "min": int(a.min()),
            "max": int(a.max()), "std": float(a.std())}


def pbe(docs: list[list[int]], steps: list[int], prefix_len: int) -> dict:
    """For each step n: prefix (prefix_len toks) -> set of token at offset +n.
    Branching count = |set| per prefix. Returns stats over prefixes per step.
    Prefixes are built within each doc (no cross-doc spillover)."""
    out = {}
    for n in steps:
        branch: dict[tuple, set] = {}
        off = prefix_len + n - 1  # @1 -> immediate next token after prefix
        for toks in docs:
            for i in range(len(toks) - off):
                key = tuple(toks[i:i + prefix_len])
                branch.setdefault(key, set()).add(toks[i + off])
        counts = [len(s) for s in branch.values()]
        if counts:
            stats = _stats(counts)
            # compact histogram (branching_count -> n_prefixes) for ECDF reconstruction
            stats["hist"] = dict(Counter(counts))
            out[f"pbe@{n}"] = stats
        else:
            out[f"pbe@{n}"] = None
    return out


def characterize(docs: list[list[int]], cfg: dict, dataset_id: str) -> dict:
    flat = [t for d in docs for t in d]
    h = heaps(flat)
    if h["token_count"] < cfg["min_tokens"]:
        log.warning("%s: %d tokens < min_tokens %d — beta unreliable",
                    dataset_id, h["token_count"], cfg["min_tokens"])
    return {
        "dataset_id": dataset_id,
        "tokenizer": cfg["tokenizer"],
        "token_count": h.pop("token_count"),
        "vocab_size": h.pop("vocab_size"),
        "heaps": h,
        "pbe": pbe(docs, cfg["pbe_steps"], cfg["pbe_prefix_len"]),
    }
