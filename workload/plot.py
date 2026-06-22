"""Scatter: PBE@1 mean (x) vs Heaps' beta (y), marker per family, color per dataset."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = {"code": "o", "nl": "^", "math": "D", "dialogue": "s"}


def plot(results_dir: Path, out_stem: str = "workload_character"):
    files = sorted(results_dir.glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        print("no results to plot")
        return

    ys = [r["heaps"]["beta"] for r in rows]
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    steps = sorted(int(k.split("@")[1]) for k in rows[0]["pbe"])  # [1, 2, 4]

    fig, axes = plt.subplots(1, len(steps), figsize=(6 * len(steps), 7), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), steps):
        xs = [r["pbe"][f"pbe@{n}"]["mean"] for r in rows]
        for r, x, y, c in zip(rows, xs, ys, colors):
            ax.scatter(x, y, c=[c], marker=MARKERS.get(r["family"], "o"),
                       s=140, edgecolors="black", label=r["dataset_id"])
        ax.set_xlabel(f"PBE@{n} mean (→ harder to draft)")
        ax.set_title(f"PBE@{n}")
    np.atleast_1d(axes)[0].set_ylabel("Heaps' β (→ more diverse vocab)")
    last = np.atleast_1d(axes)[-1]
    leg_ds = last.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="dataset")
    # second legend: marker shape per family
    fam_handles = [Line2D([], [], marker=m, color="black", ls="", markersize=9,
                          markerfacecolor="white", label=fam)
                   for fam, m in MARKERS.items()]
    last.add_artist(leg_ds)
    last.legend(handles=fam_handles, bbox_to_anchor=(1.02, 0.45), loc="upper left",
                fontsize=8, title="family (marker)")
    fig.suptitle("Workload Character")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(results_dir.parent / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"saved {out_stem}.png / .svg")


if __name__ == "__main__":
    plot(Path(__file__).parent / "results")
