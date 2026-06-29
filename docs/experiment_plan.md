# Experiment plan

> Stub — fill as the offline benchmark (M3) and predictor (M4) land.

**Offline phase** — run the simulator for real to build ground truth:
for each dataset × mode {baseline, depth, width} × n ∈ {1,2,3,4} × budget K, record
`acceptance_rate` and `actual_speedup` into `artifacts/results/benchmark_results.csv`.

**Modeling phase** — fit acceptance (and thus estimated speedup) from workload features
(Heaps β, PBE_P90@1/@2/@4, repetition rate). Evaluate with MAE / RMSE / R² and
estimated-vs-actual plots.

**Decision phase** — given a new workload, compute features → estimate speedup → choose
no-spec / depth / width without trying every config.
