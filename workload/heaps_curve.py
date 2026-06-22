"""Heaps' law vocabulary-growth curves: unique tokens (V) vs tokens read (N),
one curve per dataset, log-log. Slope = beta. Steeper = learns new words faster."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot(results_dir: Path, out_stem: str = "heaps_curve"):
    rows = [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]
    if not rows:
        print("no results")
        return
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for r, c in zip(rows, colors):
        cur = r["heaps"]["curve"]
        ax.plot(cur["N"], cur["V"], color=c, lw=1.8, marker=".", ms=4,
                label=f"{r['dataset_id']} (β={r['heaps']['beta']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N — tokens read (log)")
    ax.set_ylabel("V — unique tokens seen (log)")
    ax.set_title("Heaps' law: vocabulary growth (slope = β)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(results_dir.parent / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"saved {out_stem}.png / .svg")


if __name__ == "__main__":
    plot(Path(__file__).parent / "results")
