"""Theoretical n-gram spec-decoding speedup per dataset + curve fit from PBE P90.

`speedup()` returns a THEORETICAL LOWER-BOUND PROXY, not measured wall-clock:
block efficiency = expected tokens per verifier pass, with per-position greedy
acceptance a_k = fraction of contexts with branching=1 (the acceptance *ceiling*).
A context that branches >1 way may still be drafted correctly, so the proxy
under-counts -> lower bound. Trust the cross-dataset *ranking*, not the absolutes.
Validating the fitted function needs a real benchmark `actual_speedup` column.
# ponytail: drafter cost = 0 (n-gram); add a (gamma*c) penalty if a model drafter is benchmarked
"""
import json
from pathlib import Path
import numpy as np

DRAFT_LEN = 4  # gamma: draft tokens proposed per step


def accept(stat: dict) -> float:
    """Per-position acceptance = P(branching == 1) from the compact histogram."""
    h = stat["hist"]
    return h.get("1", 0) / sum(h.values())


def speedup(result: dict) -> float:
    """1 + a1 + a1*a2 + a1*a2*a3 + a1*a2*a3*a4  (gamma=4, a3 interpolated)."""
    pbe = result["pbe"]
    a1, a2, a4 = (accept(pbe[f"pbe@{n}"]) for n in (1, 2, 4))
    a3 = (a2 + a4) / 2  # ponytail: linear interp of the one unmeasured depth; remeasure @3 if it matters
    a = [a1, a2, a3, a4][:DRAFT_LEN]
    s, prod = 1.0, 1.0
    for ak in a:
        prod *= ak
        s += prod
    return s


def _feature_val(r, c):
    if c == "beta":      return r["heaps"]["beta"]
    if c == "tokens":    return np.log10(r["token_count"])          # log-scale
    if c == "accept@1":  return accept(r["pbe"]["pbe@1"])
    if c == "accept@2":  return accept(r["pbe"]["pbe@2"])
    if c == "accept@4":  return accept(r["pbe"]["pbe@4"])
    return r["pbe"][f"pbe@{c}"]["p90"]                              # P90@n


def _features(rows, cols):
    """Design matrix for the named feature columns (+ intercept)."""
    X = np.array([[_feature_val(r, c) for c in cols] for r in rows])
    return np.column_stack([X, np.ones(len(rows))])


def _fit_one(rows, cols, y):
    X = _features(rows, cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    # ponytail: in-sample error only; held-out split is meaningless at 3 rows/family
    fam = np.array([r["family"] for r in rows])
    per_fam = {f: float(np.abs(resid[fam == f]).mean()) for f in sorted(set(fam))}
    return {"coef": coef, "r2": r2, "mae": float(np.abs(resid).mean()),
            "rmse": float(np.sqrt(np.mean(resid ** 2))), "per_fam": per_fam}


def _fit_report(rows, y, label) -> str:
    models = [["beta", "1", "2", "4"], ["beta", "1", "2"], ["beta", "1"], ["beta"]]
    lines = [f"fit: {label} ~ features ({len(rows)} datasets, linear least squares, in-sample)", ""]
    for cols in models:
        m = _fit_one(rows, cols, y)
        names = ["P90@" + c if c != "beta" else "β" for c in cols]
        terms = " + ".join(f"{c:+.3f}·{n}" for c, n in zip(m["coef"][:-1], names))
        per_fam = "  ".join(f"{f}={e:.2f}" for f, e in m["per_fam"].items())
        lines.append(f"  [{', '.join(names):<22}] R²={m['r2']:.3f} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f}")
        lines.append(f"      {label} = {terms} {m['coef'][-1]:+.3f}")
        lines.append(f"      per-family MAE: {per_fam}")
    return "\n".join(lines)


def fit(rows) -> str:
    """Fit nested models predicting est_speedup (proxy) from (beta, P90@1/2/4)."""
    return _fit_report(rows, np.array([speedup(r) for r in rows]), "est_speedup")


def fit_actual(rows, bench: dict) -> str:
    """Fit nested models predicting actual_speedup from (beta, P90@1/2/4).
    bench: {dataset_id: {"actual_speedup": float}} from bench_results.json."""
    pairs = [(r, bench[r["dataset_id"]]["actual_speedup"])
             for r in rows if r["dataset_id"] in bench]
    if not pairs:
        return "(no actual_speedup data — run `python run.py` first)"
    fit_rows, y = zip(*pairs)
    return _fit_report(list(fit_rows), np.array(y), "actual_speedup")


def fit_poly(rows, bench: dict) -> str:
    """Polynomial fits (degree 2) on actual_speedup.
    # ponytail: 12 rows → max ~4 params safely; degree-2 β only = 3 params (fine);
    # β+P90@1 with interaction = 6 params (borderline, report in-sample only)
    """
    pairs = [(r, bench[r["dataset_id"]]["actual_speedup"])
             for r in rows if r["dataset_id"] in bench]
    if not pairs:
        return "(no actual_speedup data)"
    fit_rows, y_list = zip(*pairs)
    y = np.array(y_list)
    betas = np.array([r["heaps"]["beta"] for r in fit_rows])
    p90_1 = np.array([r["pbe"]["pbe@1"]["p90"] for r in fit_rows])
    fam = np.array([r["family"] for r in fit_rows])

    def _r(X, label):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        resid = y - pred
        r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        mae = float(np.abs(resid).mean())
        rmse = float(np.sqrt(np.mean(resid**2)))
        per_fam = {f: float(np.abs(resid[fam == f]).mean()) for f in sorted(set(fam))}
        pf = "  ".join(f"{f}={e:.2f}" for f, e in per_fam.items())
        terms = " ".join(f"{c:+.3f}·{n}" for c, n in zip(coef[:-1], label))
        return (f"  R²={r2:.3f} MAE={mae:.3f} RMSE={rmse:.3f}\n"
                f"      actual_speedup = {terms} {coef[-1]:+.3f}\n"
                f"      per-family MAE: {pf}"), coef

    lines = [f"poly fit: actual_speedup (12 datasets, in-sample)\n"]

    # degree-2 β only: β, β²
    X2b = np.column_stack([betas, betas**2, np.ones(len(y))])
    rep, _ = _r(X2b, ["β", "β²"])
    lines.append(f"  [β, β²              ]{rep}")

    # degree-2 β + P90@1 + interaction: β, P90@1, β², P90@1², β·P90@1
    X2bp = np.column_stack([betas, p90_1, betas**2, p90_1**2,
                             betas * p90_1, np.ones(len(y))])
    rep, _ = _r(X2bp, ["β", "P90@1", "β²", "P90@1²", "β·P90@1"])
    lines.append(f"  [β, P90@1, +cross  ]{rep}")

    return "\n".join(lines)


def _load():
    rows = [json.loads(f.read_text()) for f in sorted((Path(__file__).parent / "results").glob("*.json"))]
    rows.sort(key=lambda r: (r["family"], r["dataset_id"]))
    return rows


if __name__ == "__main__":
    rows = _load()
    for r in rows:
        print(f"  {r['dataset_id']:<22} β={r['heaps']['beta']:.3f}  speedup={speedup(r):.3f}")
    print()
    print(fit(rows))

    by_id = {r["dataset_id"]: r for r in rows}
    assert all(speedup(r) >= 1 for r in rows), "speedup must be >= 1"
    assert speedup(by_id["dialogue_dolly"]) > speedup(by_id["math_math"]), "easy should beat hard"
    betas = [r["heaps"]["beta"] for r in rows]
    sched = [speedup(r) for r in rows]
    corr = np.corrcoef(betas, sched)[0, 1]
    assert corr > 0, f"speedup should rise with β, got corr={corr:.2f}"
    print(f"\nself-check OK (corr(β, speedup)={corr:.2f})")
