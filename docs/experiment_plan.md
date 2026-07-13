# Experiment plan

**Offline phase** (implemented, M3) — run the simulator for real to build ground truth:
for each dataset × mode {baseline, depth, width} × n ∈ {1,2,3,4} × budget K, record
`acceptance_rate` and `actual_speedup` into `artifacts/results/benchmark_results.csv`.
Ground truth is measured **gen-only**: the prompt is prefill, decoding starts at each
sample's boundary. The baseline is one row per dataset (written as n=0).

**Modeling phase** (implemented, M4) — fit acceptance from workload features
(Heaps β, PBE_P90@1/@2/@4, repetition rate) with
`fit_from_benchmark(csv, scope="ctx"|"gen")`: the **ctx** fit is the deployable predictor
(features observable before generating), the **gen** fit is the predictability upper
bound. Estimated speedup = `1 + Σ pᵏ`. Evaluate with MAE / RMSE / R²; leave-one-out
validation is deferred until the multi-dataset CSV exists (see the ponytail note in
`src/estimation/curve_fitting.py`).

**Decision phase** (implemented, M4) — given a new workload, compute features → estimate
speedup → choose no-spec / depth / width without trying every config.

**Plots** — estimated-vs-actual scatter, speedup per dataset × mode, acceptance-vs-feature,
speedup-vs-n: roadmap milestone M5 ([roadmap.md](roadmap.md)).
