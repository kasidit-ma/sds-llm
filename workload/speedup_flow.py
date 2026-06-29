"""Flow diagram for the speedup + curve-fitting pipeline -> speedup_flow.png/.svg.
2 stages: (A) measure speedup per dataset, (B) curve-fit then apply to new workloads.
# ponytail: matplotlib boxes+arrows — Excalidraw MCP can't write a png to disk for docx.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).parent


def _box(ax, x, y, w, h, text, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, color="#1e1e1e", style="-", label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.5, linestyle=style,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom",
                fontsize=8, color=color)


def plot(out_stem="speedup_flow"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12);  ax.set_ylim(0, 9);  ax.axis("off")
    ax.text(6, 8.6, "Speedup & Curve Fitting — n-gram Spec Decoding",
            ha="center", fontsize=15, weight="bold")

    # ── STAGE A zone ──────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.3, 5.4), 11.4, 2.7, boxstyle="round,pad=0.02",
                                facecolor="#dbe4ff", edgecolor="#4a9eed", alpha=0.35, lw=1))
    ax.text(0.6, 7.8, "STAGE A — Measure speedup (per dataset)",
            fontsize=12, color="#2563eb", weight="bold")
    _box(ax, 0.6, 6.3, 1.7, 0.9, "12 datasets\n(corpus)", "#a5d8ff", "#4a9eed")
    _arrow(ax, 2.3, 6.75, 2.9, 6.75)
    _box(ax, 2.9, 6.3, 1.7, 0.9, "tokenize\nQwen2.5", "#a5d8ff", "#4a9eed")
    _box(ax, 5.2, 7.0, 3.0, 0.8, "features:\nHeaps b + PBE P90@1/2/4", "#fff3bf", "#f59e0b", 9)
    _arrow(ax, 4.6, 6.85, 5.2, 7.35, "#f59e0b")
    _box(ax, 5.2, 5.9, 3.0, 0.8, "n-gram spec-decode sim\n(baseline / spec steps)", "#d0bfff", "#8b5cf6", 9)
    _arrow(ax, 4.6, 6.65, 5.2, 6.25, "#8b5cf6")
    _box(ax, 8.8, 5.9, 2.3, 0.8, "actual_speedup\n(label)", "#b2f2bb", "#22c55e")
    _arrow(ax, 8.2, 6.3, 8.8, 6.3, "#22c55e")

    # ── STAGE B zone ──────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.3, 0.6), 11.4, 4.4, boxstyle="round,pad=0.02",
                                facecolor="#e5dbff", edgecolor="#8b5cf6", alpha=0.35, lw=1))
    ax.text(0.6, 4.65, "STAGE B — Curve fitting, then apply",
            fontsize=12, color="#5b21b6", weight="bold")
    _arrow(ax, 6.7, 5.9, 2.1, 4.15, "#06b6d4", label="collect 12 rows")
    _box(ax, 0.6, 3.3, 2.7, 0.85, "features + actual_speedup\n(12 datasets)", "#c3fae8", "#06b6d4", 9)
    _arrow(ax, 3.3, 3.72, 3.9, 3.72)
    _box(ax, 3.9, 3.3, 1.9, 0.85, "linear\nleast squares", "#d0bfff", "#8b5cf6")
    _arrow(ax, 5.8, 3.72, 6.4, 3.72)
    _box(ax, 6.4, 3.2, 3.7, 1.05, "pred = w0 + w1*b + w2*P90@1\n+ w3*P90@2 + w4*P90@4",
         "#ffd8a8", "#f59e0b", 9)
    _arrow(ax, 10.1, 3.72, 10.6, 3.72)
    _box(ax, 10.5, 3.3, 1.1, 0.85, "validate:\nR2, MAE", "#eebefa", "#ec4899", 8)

    # APPLY row
    ax.text(0.6, 2.55, "APPLY (predict before benchmarking):",
            fontsize=11, color="#b45309", weight="bold")
    _box(ax, 0.6, 1.3, 1.9, 0.85, "NEW workload", "#ffd8a8", "#f59e0b")
    _arrow(ax, 2.5, 1.72, 3.1, 1.72)
    _box(ax, 3.1, 1.3, 2.7, 0.85, "measure features only\n(cheap)", "#fff3bf", "#f59e0b", 9)
    _arrow(ax, 5.8, 1.72, 6.4, 1.72)
    _box(ax, 6.4, 1.3, 3.4, 0.85, "predict speedup\n(skip full benchmark)", "#b2f2bb", "#22c55e", 9)
    _arrow(ax, 8.25, 3.2, 8.1, 2.15, "#8b5cf6", style="--", label="fitted model")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{out_stem}.{ext}", bbox_inches="tight", dpi=130)
    print(f"saved {out_stem}.png / .svg")


if __name__ == "__main__":
    plot()
    assert (HERE / "speedup_flow.png").exists(), "speedup_flow.png not written"
    print("self-check OK")
