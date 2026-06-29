# Architecture

> Stub — expand as milestones land.

Pipeline:

```
data (HF tokens) -> drafter (n-gram) -> verifier (greedy) -> playback loop -> metrics
                                                                   |
                                            workload character -> estimation -> decision
```

Module map (`src/`):

- `interface/` — ABCs every implementation honors (drafter, verifier, playback; + tensor ABCs).
- `drafter/` — `ngram_drafter` (list), `tensor_ngram_drafter` (M5).
- `verifier/` — `greedy_verifier` (list), `tensor_greedy_verifier` (M5).
- `playback/` — `speculative_playback` (drafter=None ⇒ baseline).
- `metrics/` — `playback_metrics` (accepts, steps, `actual_speedup`).
- `spec_bench/` — `data` (HF loader), `benchmark`/`speedup` (M3 grid → CSV).
- `workload/` — `heaps`, `pbe`, `repetition`, `characterizer` (M2 features).
- `estimation/` — acceptance model, speedup estimator, curve fitting, evaluator (M4).
- `decision/` — `policy.decide(...)` → no-spec / depth / width (M4).
