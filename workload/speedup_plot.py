"""3-panel speedup plot: (1) β vs actual + fit line, (2) est vs actual, (3) bar chart."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
COLORS = {"code": "#4C72B0", "dialogue": "#55A868", "math": "#C44E52", "nl": "#DD8452"}
MARKERS = {"code": "o", "dialogue": "s", "math": "D", "nl": "^"}


def _load():
    rows = [json.loads(f.read_text()) for f in sorted((HERE / "results").glob("*.json"))]
    rows.sort(key=lambda r: (r["family"], r["dataset_id"]))
    bench = json.loads((HERE / "bench_results.json").read_text())
    return rows, bench


def plot(out_stem="speedup_plot"):
    rows, bench = _load()
    paired = [(r, bench[r["dataset_id"]]) for r in rows if r["dataset_id"] in bench]
    if not paired:
        print("no bench_results.json — run `python run.py` first")
        return

    betas = np.array([r["heaps"]["beta"] for r, _ in paired])
    actual = np.array([b["actual_speedup"] for _, b in paired])
    from speedup import speedup as est_speedup
    est = np.array([est_speedup(r) for r, _ in paired])
    labels = [r["dataset_id"].replace("_", "\n") for r, _ in paired]
    families = [r["family"] for r, _ in paired]
    c_arr = [COLORS[f] for f in families]
    m_arr = [MARKERS[f] for f in families]

    # fit lines: linear and degree-2 polynomial on β
    beta_line = np.linspace(betas.min() - 0.01, betas.max() + 0.01, 100)
    coef1 = np.polyfit(betas, actual, 1)
    fit_line1 = np.polyval(coef1, beta_line)
    coef2 = np.polyfit(betas, actual, 2)
    fit_line2 = np.polyval(coef2, beta_line)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── panel 1: β vs actual_speedup + fit line ───────────────────────────────
    ax = axes[0]
    for (r, b), c, m in zip(paired, c_arr, m_arr):
        ax.scatter(r["heaps"]["beta"], b["actual_speedup"],
                   color=c, marker=m, s=120, edgecolors="black", zorder=3)
        ax.annotate(r["dataset_id"].split("_", 1)[1],
                    (r["heaps"]["beta"], b["actual_speedup"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 3))
    ax.plot(beta_line, fit_line1, "k--", lw=1.5,
            label=f"linear: {coef1[0]:.2f}·β {coef1[1]:+.2f}")
    ax.plot(beta_line, fit_line2, "k-", lw=1.5,
            label=f"poly2: {coef2[0]:.2f}·β² {coef2[1]:+.2f}·β {coef2[2]:+.2f}")
    ax.set_xlabel("Heaps' β")
    ax.set_ylabel("actual_speedup")
    ax.set_title("β vs actual speedup")
    ax.legend(fontsize=8)

    # ── panel 2: est vs actual scatter (diagonal = perfect) ───────────────────
    ax = axes[1]
    lo, hi = min(actual.min(), est.min()) - 0.1, max(actual.max(), est.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
    for (r, b), e, c, m in zip(paired, est, c_arr, m_arr):
        ax.scatter(b["actual_speedup"], e, color=c, marker=m, s=120,
                   edgecolors="black", zorder=3)
        ax.annotate(r["dataset_id"].split("_", 1)[1],
                    (b["actual_speedup"], e),
                    fontsize=7, textcoords="offset points", xytext=(4, 2))
    ax.set_xlabel("actual_speedup")
    ax.set_ylabel("est_speedup (proxy)")
    ax.set_title("est vs actual speedup")
    mae = float(np.abs(est - actual).mean())
    ax.text(0.05, 0.95, f"MAE={mae:.3f}", transform=ax.transAxes,
            fontsize=9, va="top")
    ax.legend(fontsize=8)

    # ── panel 3: bar chart sorted by actual_speedup ───────────────────────────
    ax = axes[2]
    order = np.argsort(actual)[::-1]
    x = np.arange(len(order))
    bars_a = ax.bar(x - 0.2, actual[order], width=0.35,
                    color=[c_arr[i] for i in order], edgecolor="black", label="actual")
    bars_e = ax.bar(x + 0.2, est[order], width=0.35,
                    color=[c_arr[i] for i in order], edgecolor="black",
                    alpha=0.4, hatch="//", label="est (proxy)")
    short_labels = [labels[i] for i in order]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("speedup")
    ax.set_title("speedup per dataset")
    ax.legend(fontsize=8)

    # shared family legend
    fam_handles = [mpatches.Patch(color=COLORS[f], label=f) for f in COLORS]
    fig.legend(handles=fam_handles, title="family", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9)

    fig.suptitle("n-gram Speculative Decoding — Speedup Analysis", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"saved {out_stem}.png / .svg")


if __name__ == "__main__":
    plot()
