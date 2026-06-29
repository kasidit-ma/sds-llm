#!/usr/bin/env python
"""Run the n-gram speculative-decoding simulator on one dataset and print metrics.

Example:
    python run.py --dataset code_humaneval --n 3 --K 10 --S 5 --T 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from drafter.ngram_drafter import NGramDrafter  # noqa: E402
from metrics.playback_metrics import PlaybackMetrics, compute_metrics  # noqa: E402
from playback.speculative_playback import SpeculativePlayback  # noqa: E402
from spec_bench.data import load_config, load_tokenizer, load_tokens  # noqa: E402


def _run_mode(samples: list[list[int]], n: int, K: int, mode: str | None) -> PlaybackMetrics:
    """Aggregate a single playback mode across all samples into one StepLog-derived metric."""
    from interface.abstract_playback import StepLog

    playback = SpeculativePlayback()
    agg = StepLog(total_tokens=0, steps=0)
    for tokens in samples:
        if len(tokens) < 2:
            continue
        if mode is None:
            log = playback.run(tokens, drafter=None)
        else:
            drafter = NGramDrafter(n=n)
            # ponytail: per-sample datastore; switch to global corpus later if needed
            drafter.build_datastore(tokens)
            log = playback.run(tokens, drafter=drafter, K=K, mode=mode)
        agg.total_tokens += log.total_tokens
        agg.steps += log.steps
        agg.drafted += log.drafted
        agg.accepted += log.accepted
        agg.rejected += log.rejected
        agg.per_step_accepted.extend(log.per_step_accepted)
    return compute_metrics(agg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="code_humaneval")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--S", type=int, default=5)  # reserved for width tuning (M5)
    ap.add_argument("--T", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    tokenizer = load_tokenizer(cfg)
    print(f"Loading {args.dataset} with tokenizer {cfg['tokenizer']} ...")
    samples = load_tokens(args.dataset, cfg, tokenizer)
    print(f"{len(samples)} samples, {sum(len(s) for s in samples)} tokens\n")

    results = {
        "baseline": _run_mode(samples, args.n, args.K, None),
        "depth": _run_mode(samples, args.n, args.K, "depth"),
        "width": _run_mode(samples, args.n, args.K, "width"),
    }

    cols = ["speculative_steps", "acceptance_rate", "avg_accepted_per_step", "actual_speedup"]
    print(f"{'mode':<10}" + "".join(f"{c:>22}" for c in cols))
    for mode, m in results.items():
        d = m.as_dict()
        print(f"{mode:<10}" + "".join(f"{d[c]:>22.4f}" for c in cols))


if __name__ == "__main__":
    main()
