"""Comparison tables (PBE P90 + PBE mean) per dataset. Prints + writes summary.md."""
import json
from pathlib import Path

import speedup


def _md_table(rows, header, cell):
    """header = column names list; cell(r) -> list of stringified values."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(r)) + " |" for r in rows]
    return "\n".join(lines)


def table(results_dir: Path, out_name: str = "workload_summary.md") -> str:
    rows = [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]
    rows.sort(key=lambda r: (r["family"], r["dataset_id"]))

    bench_path = results_dir.parent / "bench_results.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}

    def _actual(r):
        b = bench.get(r["dataset_id"])
        return f"{b['actual_speedup']:.2f}" if b else "—"

    p90 = _md_table(
        rows, ["dataset", "family", "tokens", "β", "P90@1", "P90@2", "P90@4",
               "est_speedup", "actual_speedup"],
        lambda r: [r["dataset_id"], r["family"], f"{r['token_count']:,}",
                   f"{r['heaps']['beta']:.3f}",
                   *[f"{r['pbe'][f'pbe@{n}']['p90']:.0f}" for n in (1, 2, 4)],
                   f"{speedup.speedup(r):.2f}", _actual(r)])
    mean = _md_table(
        rows, ["dataset", "family", "mean@1", "mean@2", "mean@4"],
        lambda r: [r["dataset_id"], r["family"],
                   *[f"{r['pbe'][f'pbe@{n}']['mean']:.2f}" for n in (1, 2, 4)]])

    md = (f"## PBE P90\n\n{p90}\n\n## PBE mean\n\n{mean}\n\n"
          f"{_speedup_section(rows, bench)}")
    (results_dir.parent / out_name).write_text(md)
    return md


def _speedup_section(rows, bench=None) -> str:
    bench = bench or {}
    return f"""## Speedup model

**Purpose.** Curve fitting here learns an empirical mapping from workload
characteristics (Heaps' β and PBE P90 at n=1,2,4) to the speedup of n-gram
speculative decoding, so we can *predict* a workload's speedup before running the
full benchmark: `est_speedup = f(β, P90@1, P90@2, P90@4)`.

> จุดประสงค์: เรียนความสัมพันธ์เชิงประจักษ์ระหว่างลักษณะ workload (β, PBE P90 ที่ n=1,2,4)
> กับ speedup จริงของ n-gram speculative decoding เพื่อ *ทำนาย* speedup ของ workload ใหม่
> ก่อนรัน benchmark จริง

**Pipeline.**
```
dataset → measure β, P90@1, P90@2, P90@4
        → [run n-gram spec decoding]  ← MISSING (no benchmark yet)
        → actual_speedup
        → fit f()  →  est_speedup  →  compare → error (MAE/RMSE/R²)
```

**est_speedup (theoretical proxy, NOT measured).** Block efficiency of an n-gram
drafter (draft cost≈0, γ=4):
`est_speedup = 1 + a₁ + a₁a₂ + a₁a₂a₃ + a₁a₂a₃a₄`, where `aₖ` = P(branching=1) at
step k. This is a **lower bound**: a context that branches >1 way may still be
drafted correctly, so it under-counts. Absolute values depend on the γ=4 /
cost≈0 assumptions — only the cross-dataset **ranking** is robust.

**Fit to proxy (est_speedup):**
```
{speedup.fit(rows)}
```

**Fit to actual_speedup — linear:**
```
{speedup.fit_actual(rows, bench)}
```

**Fit to actual_speedup — polynomial (degree 2):**
```
{speedup.fit_poly(rows, bench)}
```
β carries most of the signal; polynomial adds little over linear given only 12 datasets.

**Evaluation & improvement (next stage, once `actual_speedup` is measured).**
`error = est_speedup − actual_speedup`. A mismatch is **feedback to refine the
estimator, not a failed experiment**: if error is high, break it down per family
(code/math/dialogue/nl) to see whether one function suffices or a `family` term
is needed; check whether the 4 features are enough (else add acceptance_rate /
repetition_rate as a later stage); compare model variants and keep the lowest
MAE/RMSE. Fill the `actual_speedup` column and re-run `table.py` to fit against
real data.
{_error_section(rows, bench)}"""


def _error_section(rows, bench) -> str:
    if not bench:
        return ""
    import numpy as np
    pairs = [(r, bench[r["dataset_id"]]) for r in rows if r["dataset_id"] in bench]
    if not pairs:
        return ""
    est = np.array([speedup.speedup(r) for r, _ in pairs])
    actual = np.array([b["actual_speedup"] for _, b in pairs])
    errs = est - actual
    lines = ["\n**est vs actual_speedup (simulation):**\n"]
    lines.append("| dataset | est_speedup | actual_speedup | error |")
    lines.append("|---|---|---|---|")
    for (r, b), e in zip(pairs, errs):
        lines.append(f"| {r['dataset_id']} | {speedup.speedup(r):.2f} | {b['actual_speedup']:.2f} | {e:+.2f} |")
    mae = float(np.abs(errs).mean())
    lines.append(f"\nMAE={mae:.3f}  RMSE={float(np.sqrt(np.mean(errs**2))):.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(Path(__file__).parent / "results"))
