# sds-llm

Workload-aware **n-gram speculative-decoding** simulator and decision system.

The project characterizes a workload (Heaps β, PBE@k, repetition), predicts acceptance/speedup,
and decides whether to use speculative decoding and whether to draft *depth* (one long chain) or
*width* (several short candidates) — then runs it.

It is a **simulator, not a live LLM**: a recorded token sequence is the "target", an n-gram
drafter proposes continuations, a greedy verifier checks them against the recording, and speedup
is measured in decode steps. No model weights, no GPU — the whole thing runs on token-id lists.

---

## How it works (the simulation model)

Real speculative decoding spends one expensive model step to verify many cheap drafted tokens at
once. We *simulate* that step-saving without running a model:

1. **Target** — a recorded token sequence (the greedy output we treat as ground truth).
2. **Drafter** — builds an n-gram datastore from the tokens, then proposes the next few tokens
   from the current context.
3. **Verifier** — greedily compares the draft to the target, accepting the matching prefix and
   stopping at the first mismatch.
4. **Commit** — each step commits `accepted + 1` tokens: the accepted draft tokens *plus one free
   token* (the verifier always produces the correct next token for free). So `pos += accepted + 1`.
5. **Speedup** — baseline emits 1 token/step, so `actual_speedup = baseline_steps / speculative_steps`.

**Worked example** — at some position the target continues `[4, 2, 5]` and the drafter proposes
`[4, 2, 9]`:

```
target : 4  2  5
draft  : 4  2  9
         ✓  ✓  ✗        accepted = 2
commit : 4  2  5         → 3 tokens in ONE step (2 accepted + 1 free correct token)
```

A baseline decode would have taken 3 steps for those 3 tokens; speculative took 1.

### Depth vs Width drafting

| Mode | Draft shape | Good for |
|------|-------------|----------|
| **depth** | one long chain `[a, b, c, …]` | low branching / high repetition (code, logs) — long deterministic continuations |
| **width** | several short candidates `[[a,b],[a,c],[d,e]]` | prefixes that branch many ways — verifier picks the candidate that accepts the most |

---

## Pipeline

```
                          ┌─────────── M1 (implemented) ───────────┐
 dataset.yaml ─► data ─► tokenizer ─► drafter ─► verifier ─► playback ─► metrics
                                       (n-gram)   (greedy)    (steps)    (speedup)
                                                                  │
                          ┌──────────── M2–M4 (stubs) ────────────┘
 tokens ─► workload character ─► acceptance/speedup predictor ─► decision (no-spec / depth / width)
           (Heaps β, PBE@k,        (fit on benchmark ground truth)
            repetition)
```

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
│   └── experiment.yaml    # benchmark grid (M3)
│
├── src/
│   ├── interface/         # ABCs every implementation honors            [done]
│   │   ├── abstract_drafter.py        # build_datastore() + propose()
│   │   ├── abstract_verifier.py       # verify() + verify_best()
│   │   ├── abstract_playback.py       # run() + StepLog dataclass
│   │   └── abstract_tensor_*.py       # tensor contracts                [stub, M5]
│   ├── drafter/
│   │   ├── ngram_drafter.py           # NGramDrafter: depth + width      [done]
│   │   └── tensor_ngram_drafter.py    #                                  [stub, M5]
│   ├── verifier/
│   │   ├── greedy_verifier.py         # GreedyVerifier: greedy + multi-seq best  [done]
│   │   └── tensor_greedy_verifier.py  #                                  [stub, M5]
│   ├── playback/
│   │   └── speculative_playback.py    # SpeculativePlayback (drafter=None ⇒ baseline)  [done]
│   ├── metrics/
│   │   └── playback_metrics.py        # StepLog → PlaybackMetrics (speedup, acceptance)  [done]
│   ├── spec_bench/
│   │   ├── data.py                    # parse dataset.yaml, HF load, tokenize  [done]
│   │   ├── benchmark.py               # grid → CSV                      [stub, M3]
│   │   └── speedup.py                 # aggregate                       [stub, M3]
│   ├── workload/                      # heaps, pbe, repetition, characterizer  [stub, M2]
│   ├── estimation/                    # acceptance_model, speedup_estimator, …  [stub, M4]
│   └── decision/
│       └── policy.py                  # decide(no-spec / depth / width) [stub, M4]
│
├── tests/                 # mirrors src/; M1 modules tested, M2/M4 tests skipped
├── scripts/               # run_tests / run_lint / run_typecheck / run_all_checks .sh
├── docs/                  # architecture, experiment_plan, workload_character
└── artifacts/             # results/ plots/ logs/ (generated)
```

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
    text_combine: [prompt, canonical_solution]   # concat these columns
  - id: nl_wikitext103
    hf_path: Salesforce/wikitext
    hf_config: wikitext-103-raw-v1               # dataset config name
    split: train
    family: nl
    text_column: text                            # single column
  # … 12 datasets total across code / math / nl / dialogue
```

- `text_column` vs `text_combine` — read one column, or concat several into the sample text.
- `hf_config` — passed as the dataset config name; `streaming: true` — stream instead of download.

`configs/ngram.yaml` holds `n / K / S / T` defaults; `configs/experiment.yaml` is the M3 grid.

---

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
python run.py --dataset code_humaneval                 # baseline vs depth vs width
python run.py --dataset code_humaneval --n 4 --K 12    # tune n-gram order / draft budget
```

Example output (real HumanEval, Qwen tokenizer):

```
mode        speculative_steps   acceptance_rate   avg_accepted_per_step   actual_speedup
baseline           30864             0.0000               0.0000              1.0000
depth               5627             0.4854               4.5079              5.4850
width              11018             0.6192               1.8098              2.8012
```

Code is low-branching, so **depth** wins big (long correct chains); **width** accepts more often
but commits fewer tokens per step.

---

## Development

```bash
./scripts/run_all_checks.sh     # ruff (lint) + pyright (types) + pytest
```

- `src/` layout — imports look like `from drafter.ngram_drafter import NGramDrafter`.
- **Format-on-save** is enabled via ruff in [.vscode/settings.json](.vscode/settings.json)
  (needs the *Ruff* extension).
- Stubs raise `NotImplementedError("Milestone Mx")`; their tests are present but skipped.

---

## Roadmap

| Milestone | Modules | State |
|-----------|---------|-------|
| **M1** simulator | `spec_bench/data`, `drafter/ngram_drafter`, `verifier/greedy_verifier`, `playback/speculative_playback`, `metrics/playback_metrics` | ✅ done |
| **M2** workload character | `workload/{heaps,pbe,repetition,characterizer}` — Heaps β, PBE_P90@k, repetition | stub |
| **M3** benchmark | `spec_bench/{benchmark,speedup}` — grid → `artifacts/results/*.csv` (ground truth) | stub |
| **M4** estimation + decision | `estimation/*`, `decision/policy` — predict acceptance/speedup, choose mode | stub |
| **M5** tensor + plots | `*/tensor_*`, plots into `artifacts/plots/` | stub |
