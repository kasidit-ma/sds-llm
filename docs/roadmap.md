# Roadmap

```
                          ┌─────────── M1 (implemented) ───────────┐
 dataset.yaml ─► data ─► tokenizer ─► drafter ─► verifier ─► playback ─► metrics
                                       (n-gram)   (greedy)    (steps)    (speedup)
                                                                  │
                          ┌───────── M2–M4 (implemented) ─────────┘
 tokens ─► workload character ─► acceptance/speedup predictor ─► decision (no-spec / depth / width)
           (Heaps β, PBE@k,        (fit on benchmark ground truth)
            repetition)
```

| Milestone | Modules | Tasks | Est. | State |
|-----------|---------|-------|------|-------|
| **M1** simulator | `spec_bench/data`, `drafter/ngram_drafter`, `verifier/greedy_verifier`, `playback/speculative_playback`, `metrics/playback_metrics` | baseline playback → static datastore → n-gram drafter (depth + width) → break-at-first-mismatch verifier → loop + StepLog + metrics | — | ✅ done |
| **M2** workload character | `workload/{heaps,pbe,repetition,characterizer}` | Heaps β via curve fit (`V = K·n^β`), PBE_P90@1/@2/@4, repetition | ~1 d | ✅ done |
| **M2b** dataset character | `workload/` (new) | `describe_dataset()` — n_rows, tokens per row, context:target ratio, language/domain tag → summary table for the report. ~~Scope ctx / gen~~ done: features per scope in benchmark CSV, speedup counted gen-only | ~0.5 d | partial |
| **M3** benchmark | `spec_bench/{benchmark,speedup}` | grid (dataset × mode × n × K) → `artifacts/results/*.csv` (ground truth); `run.py --benchmark` | ~1 d | ✅ done |
| **M4** estimation + decision | `estimation/*`, `decision/policy` | fit `p_accept = sigmoid(θ·[repetition, β, log1p(PBE@1/2/4)])` on benchmark CSV → estimated speedup `1 + Σ pᵏ` → MAE/RMSE/R² → decide no-spec / depth / width (leave-one-out deferred until multi-dataset CSV) | ~1–2 d | ✅ done |
| **M5** plots | `spec_bench/plots` | from the benchmark CSV + fit: estimated-vs-actual speedup scatter, speedup per dataset × mode bars, acceptance-vs-feature scatters, speedup-vs-n curves — `run.py --plots` | ~1 d | ✅ done |
| **M6** tensor | `*/tensor_*` | batched tensor drafter/verifier (perf rewrite of M1 — same results, faster) | ~2 d | stub |

## Deferred (post-M5, if time allows)

Report write-up (datasets → method → results → limitations), context-overlap features,
**online** datastore (built during playback instead of upfront), multi-language datasets,
unlimited draft budget (RQ2), per-mode speedup formula (the geometric `1 + Σ pᵏ` assumes a
depth chain; width is capped at `T + 1` per step, so its estimates don't track actuals —
visible in `artifacts/plots/est_vs_actual.png`).
