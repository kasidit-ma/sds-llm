# Workload character

> Stub — formulas implemented in `src/workload/` at M2.

**Heaps β** — `β = log(V) / log(N)`, where `V` = unique tokens, `N` = total tokens.
Higher β ⇒ more diverse vocabulary ⇒ harder to predict.

**Branching count @k** — for a prefix, the number of *unique* k-token continuations observed.

**PBE_P90@k** — the **90th percentile** (not the median) of `branching_count@k` over all prefixes.
Low PBE_P90@1 ⇒ most prefixes have few next-token choices ⇒ easy to draft.

**Repetition rate @n** — `1 - unique_ngram_count@n / total_ngram_count@n`.
Higher ⇒ more repeated patterns ⇒ n-gram drafts hit more often.

Config: `pbe_steps: [1, 2, 4]`, `pbe_prefix_len: 3` (see `configs/dataset.yaml`).
