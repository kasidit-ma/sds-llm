"""ECDF of per-context (per-prefix) PBE branching counts, one curve per dataset.
y=0.9 line marks P90: where each curve crosses it = that dataset's P90 branching."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _ecdf(hist: dict[str, int]):
    """hist = {branching_count: n_prefixes} -> (x sorted, cumulative fraction)."""
    xs = np.array(sorted(int(k) for k in hist))
    counts = np.array([hist[str(x)] for x in xs])
    cum = np.cumsum(counts) / counts.sum()
    return xs, cum


def plot(results_dir: Path, out_stem: str = "pbe_ecdf"):
    rows = [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]
    if not rows:
        print("no results")
        return
    steps = sorted(int(k.split("@")[1]) for k in rows[0]["pbe"])
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))

    # heavy tail (a few prefixes branch into thousands) — clip x to ~1.5x worst P90,
    # shared across all panels so they're directly comparable
    xmax = max(r["pbe"][f"pbe@{n}"]["p90"] for n in steps for r in rows) * 1.5 + 1

    fig, axes = plt.subplots(1, len(steps), figsize=(6 * len(steps), 6), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), steps):
        for r, c in zip(rows, colors):
            xs, cum = _ecdf(r["pbe"][f"pbe@{n}"]["hist"])
            ax.step(xs, cum, where="post", color=c, lw=1.5, label=r["dataset_id"])
        ax.axhline(0.9, color="grey", ls="--", lw=0.8)  # P90 reference
        ax.set_xlabel(f"PBE@{n} branching count per context")
        ax.set_title(f"PBE@{n}")
        ax.set_xlim(1, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # branching count = จำนวนเต็ม
        ax.margins(x=0)
    np.atleast_1d(axes)[0].set_ylabel("ECDF (fraction of contexts ≤ x)")
    np.atleast_1d(axes)[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("PBE branching ECDF per context (dashed = P90)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(results_dir.parent / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"saved {out_stem}.png / .svg")


if __name__ == "__main__":
    plot(Path(__file__).parent / "results")
