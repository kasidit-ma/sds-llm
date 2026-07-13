# sds-llm

Workload-aware **n-gram speculative-decoding** simulator and decision system.

**What it does** — characterizes a text workload (Heaps β, PBE@k, repetition), predicts the
acceptance rate and speedup that speculative decoding would achieve on it, and decides whether
to use speculative decoding at all — and whether to draft *depth* (one long chain) or *width*
(several short candidates).

**Problem it solves** — deciding if speculative decoding is worth it for a *new* dataset
normally requires a full GPU benchmark. This project predicts the answer from cheap, statistical
features of the text alone.

**Who it's for** — researchers/engineers evaluating n-gram speculative decoding across workloads
(code, math, NL, dialogue) without running a live model.

It is a **simulator, not a live LLM**: a recorded token sequence is the "target", an n-gram
drafter proposes continuations, a greedy verifier checks them against the recording, and speedup
is measured in decode steps. No model weights, no GPU — the whole thing runs on token-id lists.

Development status lives in [docs/roadmap.md](docs/roadmap.md).

---

## Tech stack

- **Python ≥ 3.10**, `src/` layout — pure Python; the simulator itself is stdlib-only
- [`datasets`](https://huggingface.co/docs/datasets) + [`transformers`](https://huggingface.co/docs/transformers) — load HF datasets, tokenize (default tokenizer: `Qwen/Qwen2.5-7B`)
- `numpy` — least-squares fit of the acceptance model
- `pyyaml` — configs
- Dev: `ruff` (lint/format) · `pyright` (types) · `pytest`

---

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
python run.py --dataset code_humaneval                 # baseline vs depth vs width
python run.py --dataset code_humaneval --n 4 --K 12    # tune n-gram order / draft budget
python run.py --dataset code_humaneval --peek 2        # inspect raw rows + context/target split
python run.py --dataset code_humaneval --features      # workload features (Heaps β, PBE, …)
python run.py --benchmark                              # grid from experiment.yaml → results CSV
python run.py --plots                                  # benchmark CSV → PNGs in artifacts/plots/
python run.py --fit                                    # linear speedup fit table (a + b·β + c·PBE)
./scripts/run_all_checks.sh                            # ruff + pyright + pytest
```

---

## Example

```console
$ python run.py --dataset code_humaneval
mode        speculative_steps   acceptance_rate   avg_accepted_per_step   actual_speedup
baseline            8872             0.0000               0.0000              1.0000
depth               1575             0.4933               4.7149              5.6330
width               3141             0.6641               1.8548              2.8246
```

Steps are counted over the generation part only (the prompt is prefill). Code is
low-branching, so **depth** wins big (long correct chains); **width** accepts more often but
commits fewer tokens per step.

Fit the acceptance/speedup predictor on a benchmark CSV:

```bash
PYTHONPATH=src python -c \
  "from estimation.curve_fitting import fit_from_benchmark; \
   print(fit_from_benchmark('artifacts/results/benchmark_results.csv'))"
```

### How a step works

Each verify step commits `accepted + 1` tokens — the accepted draft prefix plus one free
correct token from the verifier — so `actual_speedup = baseline_steps / speculative_steps`:

```
target : 4  2  5
draft  : 4  2  9
         ✓  ✓  ✗        accepted = 2
commit : 4  2  5         → 3 tokens in ONE step (2 accepted + 1 free)
```

| Mode | Draft shape | Good for |
|------|-------------|----------|
| **depth** | one long chain `[a, b, c, …]` | low branching / high repetition (code, logs) |
| **width** | several short candidates `[[a,b],[a,c],[d,e]]` | branchy prefixes — verifier picks the best candidate |

---

## Configuration

Datasets and the tokenizer live in [configs/dataset.yaml](configs/dataset.yaml):

```yaml
tokenizer: Qwen/Qwen2.5-7B   # swappable
max_samples: 5000            # cap per dataset
min_tokens: 10000            # warn below this (Heaps β unreliable)
pbe_steps: [1, 2, 4]         # PBE@1 / @2 / @4
pbe_prefix_len: 3

datasets:
  - id: code_humaneval
    hf_path: openai/openai_humaneval
    split: test
    family: code
    context_columns: [prompt]                    # prompt / prefill (free — not decoded)
    target_columns: [canonical_solution]         # what the simulator "generates"
  - id: nl_coqa
    hf_path: stanfordnlp/coqa
    split: train
    family: nl
    context_columns: [story, questions]          # list columns are joined with \n
    target_columns: [answers.input_text]         # dotted key reaches into nested dicts
  # … 12 datasets total across code / math / nl / dialogue
```

- `context_columns` + `target_columns` — split each row into prompt (prefill, boundary) vs
  generation; steps and speedup are counted **only over the target part**. A plain
  `text_column` is also supported (no prompt: the whole text is the target).
- Workload features are computed per scope: `ctx_*` (observable before generation — what the
  decision system may use) and `gen_*` (the generated text itself — the predictability
  upper bound). Both land in the benchmark CSV; `fit_from_benchmark(csv, scope="ctx"|"gen")`.
- `hf_config` — passed as the dataset config name; `streaming: true` — stream instead of download.

`configs/ngram.yaml` holds `n / K / S / T` defaults; `configs/experiment.yaml` is the benchmark grid.

---

## Metrics

`metrics/playback_metrics.py` turns a run's `StepLog` into:

| Metric | Meaning |
|--------|---------|
| `baseline_steps` | total tokens (baseline = 1 token/step) |
| `speculative_steps` | steps the simulator actually took |
| `actual_speedup` | `baseline_steps / speculative_steps` — the headline number |
| `acceptance_rate` | `accepted_tokens / drafted_tokens` — how often drafts were right |
| `max_accepted_per_step` | best single-step acceptance |
| `avg_accepted_per_step` | mean accepted draft tokens per step |

---

## Project structure

```
sds-llm/
├── run.py                 # CLI: load a dataset → run baseline/depth/width → print metrics table
├── pyproject.toml         # deps + ruff / pyright / pytest config (src layout)
│
├── configs/
│   ├── dataset.yaml       # tokenizer + 12 HF datasets (code/math/nl/dialogue) + PBE params
│   ├── ngram.yaml         # n / K / S / T defaults
│   └── experiment.yaml    # benchmark grid
│
├── src/
│   ├── interface/         # ABCs every implementation honors (tensor contracts are stubs)
│   ├── drafter/           # NGramDrafter: depth + width drafting
│   ├── verifier/          # GreedyVerifier: greedy prefix match + multi-seq best
│   ├── playback/          # SpeculativePlayback (drafter=None ⇒ baseline)
│   ├── metrics/           # StepLog → PlaybackMetrics (speedup, acceptance)
│   ├── spec_bench/        # dataset loading/tokenizing, benchmark grid → CSV, aggregation
│   ├── workload/          # Heaps β, PBE@k, repetition, characterizer
│   ├── estimation/        # acceptance model fit, speedup estimator, evaluator
│   └── decision/          # policy: no-spec / depth / width
│
├── tests/                 # mirrors src/; all non-tensor modules tested
├── scripts/               # run_tests / run_lint / run_typecheck / run_all_checks .sh
├── docs/                  # roadmap, architecture, experiment_plan, workload_character
└── artifacts/             # results/ plots/ logs/ (generated)
```

- Imports use the `src/` layout: `from drafter.ngram_drafter import NGramDrafter`.
- **Format-on-save** via ruff in [.vscode/settings.json](.vscode/settings.json) (needs the *Ruff* extension).
- Tensor variants are stubs raising `NotImplementedError("Milestone M6")` — see the [roadmap](docs/roadmap.md).

---

## Common dev tasks

**Add a dataset** — three steps, no code:

1. Add an entry to [configs/dataset.yaml](configs/dataset.yaml): `id`, `hf_path`, `split`,
   `family`, and which columns are prompt vs generation (`context_columns` / `target_columns`).
2. `python run.py --dataset <id> --peek 2` — check the raw columns and that the
   context/target split looks right (rows with an empty target are skipped automatically).
3. `python run.py --dataset <id>` — run it. Done.

**Run the full benchmark** — `python run.py --benchmark` sweeps the grid in
[configs/experiment.yaml](configs/experiment.yaml) (`datasets: []` = all) and writes
`artifacts/results/benchmark_results.csv`; feed that CSV to `fit_from_benchmark` (see Example).

**Add a drafter / verifier** — implement the ABCs in `src/interface/`
(`build_datastore()`/`propose()`, `verify()`/`verify_best()`); `SpeculativePlayback` accepts any
drafter, so nothing else changes. Mirror the module in `tests/`.

**Where to read next** — [docs/architecture.md](docs/architecture.md) (design),
[docs/roadmap.md](docs/roadmap.md) (what's done / what's next).
