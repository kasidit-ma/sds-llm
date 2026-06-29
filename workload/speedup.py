"""n-gram spec-decoding speedup: linear regression of actual_speedup on workload
features (Heaps' β, PBE P90@1/2/4).

`predicted_speedup = w0 + w1·β + w2·P90@1 + w3·P90@2 + w4·P90@4`, fit by least
squares to measured `actual_speedup` (from bench_results.json). P90@1/2/4 enter as
flat features — no product chain, so no unmeasured depth (e.g. @3) is needed.
# ponytail: drafter cost = 0 (n-gram); add a (gamma*c) penalty if a model drafter is benchmarked
"""
import json
from pathlib import Path
import numpy as np


def accept(stat: dict) -> float:
    """Per-position acceptance = P(branching == 1) from the compact histogram."""
    h = stat["hist"]
    return h.get("1", 0) / sum(h.values())


def predict(rows, bench, cols=("beta", "1", "2", "4")) -> dict:
    """In-sample regression prediction of actual_speedup per dataset.
    # ponytail: in-sample (12 rows); no held-out split, same as _fit_one"""
    pairs = [(r, bench[r["dataset_id"]]["actual_speedup"])
             for r in rows if r["dataset_id"] in bench]
    fit_rows, y = zip(*pairs)
    X = _features(list(fit_rows), list(cols))
    coef, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
    return {r["dataset_id"]: float(p) for r, p in zip(fit_rows, X @ coef)}


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
    bench = json.loads((Path(__file__).parent / "bench_results.json").read_text())
    pred = predict(rows, bench)
    for r in rows:
        print(f"  {r['dataset_id']:<22} β={r['heaps']['beta']:.3f}  "
              f"pred={pred[r['dataset_id']]:.3f}  "
              f"actual={bench[r['dataset_id']]['actual_speedup']:.3f}")
    print()
    print(fit_actual(rows, bench))

    assert all(p >= 1 for p in pred.values()), "predicted speedup must be >= 1"
    r2 = _fit_one(rows, ["beta", "1", "2", "4"], np.array(
        [bench[r["dataset_id"]]["actual_speedup"] for r in rows]))["r2"]
    assert np.isfinite(r2), "fit R² must be finite"
    print(f"\nself-check OK (fit R²={r2:.3f})")
