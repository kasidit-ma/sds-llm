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
    pred = speedup.predict(rows, bench) if bench else {}

    def _actual(r):
        b = bench.get(r["dataset_id"])
        return f"{b['actual_speedup']:.2f}" if b else "—"

    def _pred(r):
        p = pred.get(r["dataset_id"])
        return f"{p:.2f}" if p is not None else "—"

    p90 = _md_table(
        rows, ["dataset", "family", "tokens", "β", "P90@1", "P90@2", "P90@4",
               "pred_speedup", "actual_speedup"],
        lambda r: [r["dataset_id"], r["family"], f"{r['token_count']:,}",
                   f"{r['heaps']['beta']:.3f}",
                   *[f"{r['pbe'][f'pbe@{n}']['p90']:.0f}" for n in (1, 2, 4)],
                   _pred(r), _actual(r)])
    mean = _md_table(
        rows, ["dataset", "family", "mean@1", "mean@2", "mean@4"],
        lambda r: [r["dataset_id"], r["family"],
                   *[f"{r['pbe'][f'pbe@{n}']['mean']:.2f}" for n in (1, 2, 4)]])

    md = (f"## PBE P90\n\n{p90}\n\n## PBE mean\n\n{mean}\n\n"
          f"{_speedup_section(rows, bench, pred)}")
    (results_dir.parent / out_name).write_text(md)
    return md


def _speedup_section(rows, bench=None, pred=None) -> str:
    bench = bench or {}
    pred = pred or {}
    return f"""## Speedup model

**Purpose.** Curve fitting here learns an empirical mapping from workload
characteristics (Heaps' β and PBE P90 at n=1,2,4) to the speedup of n-gram
speculative decoding, so we can *predict* a workload's speedup before running the
full benchmark: `pred_speedup = f(β, P90@1, P90@2, P90@4)`.

> จุดประสงค์: เรียนความสัมพันธ์เชิงประจักษ์ระหว่างลักษณะ workload (β, PBE P90 ที่ n=1,2,4)
> กับ speedup จริงของ n-gram speculative decoding เพื่อ *ทำนาย* speedup ของ workload ใหม่
> ก่อนรัน benchmark จริง

**Pipeline.**
```
dataset → measure β, P90@1, P90@2, P90@4
        → run n-gram spec decoding → actual_speedup
        → fit f() (least squares) → pred_speedup → residual (MAE/RMSE/R²)
```

**pred_speedup (linear regression).** A least-squares fit of measured
`actual_speedup` on the workload features: `pred_speedup = w0 + w1·β + w2·P90@1 +
w3·P90@2 + w4·P90@4`. Features enter flat (no product chain), so no unmeasured
depth is required. β carries most of the signal.

**Fit to actual_speedup — linear:**
```
{speedup.fit_actual(rows, bench)}
```

**Fit to actual_speedup — polynomial (degree 2):**
```
{speedup.fit_poly(rows, bench)}
```
β carries most of the signal; polynomial adds little over linear given only 12 datasets.

**Evaluation & improvement.** `residual = pred_speedup − actual_speedup`. A high
residual is **feedback to refine the estimator, not a failed experiment**: break it
down per family (code/math/dialogue/nl) to see whether one function suffices or a
`family` term is needed; check whether the 4 features are enough (else add
acceptance_rate / repetition_rate as a later stage); compare model variants and keep
the lowest MAE/RMSE. Re-run with real (non-simulation) `actual_speedup` to validate.
{_error_section(rows, bench, pred)}"""


def _error_section(rows, bench, pred) -> str:
    if not bench or not pred:
        return ""
    import numpy as np
    pairs = [(r, bench[r["dataset_id"]]) for r in rows if r["dataset_id"] in bench]
    if not pairs:
        return ""
    est = np.array([pred[r["dataset_id"]] for r, _ in pairs])
    actual = np.array([b["actual_speedup"] for _, b in pairs])
    errs = est - actual
    lines = ["\n**pred vs actual_speedup (simulation):**\n"]
    lines.append("| dataset | pred_speedup | actual_speedup | residual |")
    lines.append("|---|---|---|---|")
    for (r, b), e in zip(pairs, errs):
        lines.append(f"| {r['dataset_id']} | {pred[r['dataset_id']]:.2f} | {b['actual_speedup']:.2f} | {e:+.2f} |")
    mae = float(np.abs(errs).mean())
    lines.append(f"\nMAE={mae:.3f}  RMSE={float(np.sqrt(np.mean(errs**2))):.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(Path(__file__).parent / "results"))
