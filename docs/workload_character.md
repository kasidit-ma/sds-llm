# Workload character

Implemented in `src/workload/`; all features are computed **twice per dataset** — over the
context tokens (`ctx_*`) and over the generation tokens (`gen_*`).

**Heaps β** — vocabulary-growth exponent of `V = K·n^β`. Fitted, not a single-point ratio:
unique-token count `V` is checkpointed along the stream (stride keeps it at ~1000 points)
and β is the slope of the `log V` vs `log n` regression
(`statistics.linear_regression` in `heaps.py`).
Higher β ⇒ vocabulary keeps growing ⇒ more diverse ⇒ harder to predict.

**Branching count @k** — for a prefix, the number of *unique* k-token continuations observed.

**PBE_P90@k** — the **90th percentile** (not the median) of `branching_count@k` over all prefixes.
Low PBE_P90@1 ⇒ most prefixes have few next-token choices ⇒ easy to draft.

**Repetition rate @n** — `1 - unique_ngram_count@n / total_ngram_count@n`, with n=3
(fixed in `characterizer.py` to match the drafter's default n-gram order).
Higher ⇒ more repeated patterns ⇒ n-gram drafts hit more often.

Config: `pbe_steps: [1, 2, 4]`, `pbe_prefix_len: 3` (see `configs/dataset.yaml`).
