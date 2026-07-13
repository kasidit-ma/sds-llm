"""Offline benchmark grid.

Runs the simulator across datasets x modes {baseline, depth, width} x n x K, joins each row with
its workload character, and writes artifacts/results/benchmark_results.csv (the ground-truth
acceptance/speedup table the predictor is fit on).
"""

import csv
from pathlib import Path
from typing import Any

from drafter.ngram_drafter import NGramDrafter
from interface.abstract_playback import StepLog
from metrics.playback_metrics import PlaybackMetrics, compute_metrics
from playback.speculative_playback import SpeculativePlayback
from spec_bench.data import Sample, load_config, load_tokenizer, load_tokens
from workload.characterizer import characterize


def run_mode(samples: list[Sample], n: int, K: int, mode: str | None) -> PlaybackMetrics:
    """Aggregate a single playback mode across all samples into one StepLog-derived metric.

    Steps are counted only over the generation part (after each sample's prefill boundary).
    """
    playback = SpeculativePlayback()
    agg = StepLog(total_tokens=0, steps=0)
    for tokens, boundary in samples:
        if len(tokens) - boundary < 2:
            continue
        if mode is None:
            log = playback.run(tokens, drafter=None, start=boundary)
        else:
            drafter = NGramDrafter(n=n)
            # ponytail: per-sample datastore over the WHOLE sample (context + future target) —
            # optimistic upper bound; switch to context-only/online store when honesty matters
            drafter.build_datastore(tokens)
            log = playback.run(tokens, drafter=drafter, K=K, mode=mode, start=boundary)
        agg.total_tokens += log.total_tokens
        agg.steps += log.steps
        agg.drafted += log.drafted
        agg.accepted += log.accepted
        agg.rejected += log.rejected
        agg.per_step_accepted.extend(log.per_step_accepted)
    return compute_metrics(agg)


def run_benchmark(experiment_cfg: dict[str, Any], out_csv: str | Path) -> None:
    ds_cfg = load_config()
    ids = experiment_cfg["datasets"] or [e["id"] for e in ds_cfg["datasets"]]
    tokenizer = load_tokenizer(ds_cfg)
    K = experiment_cfg["K"]
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume: keep datasets whose grid is already complete in the CSV, redo the rest
    expected = sum(1 if m == "baseline" else len(experiment_cfg["n_values"])
                   for m in experiment_cfg["modes"])
    kept: list[dict[str, str]] = []
    if out.exists():
        with out.open(newline="") as fh:
            prev = list(csv.DictReader(fh))
        counts: dict[str, int] = {}
        for r in prev:
            counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
        done = {d for d in counts if counts[d] == expected and d in ids}
        kept = [r for r in prev if r["dataset"] in done]
    if kept and "ctx_pbe_mean_at_1" not in kept[0]:  # stale schema -> full rerun
        kept = []

    writer: csv.DictWriter[str] | None = None
    with out.open("w", newline="") as fh:
        if kept:
            writer = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
            fh.flush()
            print(f"resuming: {len({r['dataset'] for r in kept})} dataset(s) already done",
                  flush=True)
        for i, ds_id in enumerate(ids, start=1):
            if any(r["dataset"] == ds_id for r in kept):
                continue
            samples = load_tokens(ds_id, ds_cfg, tokenizer)
            print(f"[{i}/{len(ids)}] {ds_id}: {len(samples)} samples, characterizing ...",
                  flush=True)
            # features per scope: ctx = observable before generation (deployable),
            # gen = upper bound
            ctx_tokens = [t for s in samples for t in s.tokens[: s.boundary]]
            gen_tokens = [t for s in samples for t in s.tokens[s.boundary :]]
            feats = {f"ctx_{k}": v for k, v in characterize(ctx_tokens, ds_cfg).items()}
            feats |= {f"gen_{k}": v for k, v in characterize(gen_tokens, ds_cfg).items()}
            for mode in experiment_cfg["modes"]:
                n_values = [0] if mode == "baseline" else experiment_cfg["n_values"]
                for n in n_values:
                    m = run_mode(samples, n, K, None if mode == "baseline" else mode)
                    row = {"dataset": ds_id, "mode": mode, "n": n, "K": K,
                           **m.as_dict(), **feats}
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    fh.flush()  # progress is tail-able while the grid runs
                    print(f"  {mode:<9} n={n} speedup={row['actual_speedup']:.3f}", flush=True)
