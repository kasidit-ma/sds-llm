# Architecture

Pipeline:

```
data (HF tokens) -> drafter (n-gram) -> verifier (greedy) -> playback loop -> metrics
                                                                   |
                                            workload character -> estimation -> decision
```

## Data model

`spec_bench/data.py` yields one `Sample(tokens, boundary)` per dataset row:
`tokens[:boundary]` is the **context** (prompt/prefill — free, never decoded),
`tokens[boundary:]` is the **generation target**. Playback starts at the boundary, so steps
and speedup are counted only over the generated part. Every benchmark row carries the
workload features twice: `ctx_*` (computed on context tokens — observable *before*
generation, what the decision system may use) and `gen_*` (computed on the generation
itself — the predictability upper bound).

## Module map (`src/`)

- `interface/` — ABCs every implementation honors (drafter, verifier, playback; + tensor ABCs).
- `drafter/` — `ngram_drafter` (list), `tensor_ngram_drafter` (M6 stub).
- `verifier/` — `greedy_verifier` (list), `tensor_greedy_verifier` (M6 stub).
- `playback/` — `speculative_playback` (drafter=None ⇒ baseline; `start=` skips the prefill).
- `metrics/` — `playback_metrics` (accepts, steps, `actual_speedup`).
- `spec_bench/` — `data` (HF loader + context/target split), `benchmark`/`speedup` (grid → CSV).
- `workload/` — `heaps`, `pbe`, `repetition`, `characterizer` (M2 features).
- `estimation/` — acceptance model, speedup estimator, curve fitting, evaluator (M4).
- `decision/` — `policy.decide(...)` → no-spec / depth / width (M4).

Status per milestone: [roadmap.md](roadmap.md).
