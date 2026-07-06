"""M5 plots: benchmark CSV (+ fitted predictor) → PNGs in artifacts/plots/."""

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from estimation.curve_fitting import fit_from_benchmark, fit_linear_speedup  # noqa: E402
from estimation.evaluator import evaluate  # noqa: E402

_MODE_COLOR = {"depth": "tab:blue", "width": "tab:orange"}
_MODE_MARKER = {"depth": "o", "width": "X"}
_FAMILY_COLOR = {"code": "tab:blue", "math": "tab:green", "nl": "tab:orange",
                 "dialogue": "tab:purple"}
_FAMILY_MARKER = {"code": "o", "nl": "^", "math": "D", "dialogue": "s"}


def _ds_colors(datasets: "list[str] | set[str]") -> dict[str, Any]:
    """Stable tab20 color per dataset — shared by every per-dataset plot."""
    cmap = plt.get_cmap("tab20")
    return {ds: cmap(i % 20) for i, ds in enumerate(sorted(set(datasets)))}


def _dataset_legend(ax: Any, colors: dict[str, Any]) -> None:
    ds_handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=ds)
                  for ds, c in colors.items()]
    ax.legend(handles=ds_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, title="dataset")


def _spec_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["mode"] != "baseline"]


def _est_vs_actual(rows: list[dict[str, str]], csv_path: str, scope: str, out: Path) -> Path:
    fit = fit_from_benchmark(csv_path, scope=scope)  # same row filter/order as `rows`
    colors = _ds_colors([r["dataset"] for r in rows])
    lim = (0, max(*fit["predicted"], *fit["actual"]) * 1.05)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, mode in zip(axes, ("depth", "width"), strict=True):
        pts = [(p, a, r) for p, a, r in zip(fit["predicted"], fit["actual"], rows, strict=True)
               if r["mode"] == mode]
        for p, a, r in pts:
            ax.scatter(p, a, s=45, color=colors[r["dataset"]], marker=_MODE_MARKER[mode],
                       edgecolors="black", linewidths=0.4)
        ev = evaluate([p for p, _, _ in pts], [a for _, a, _ in pts])
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlim=lim, ylim=lim, xlabel="estimated speedup",
               title=f"{mode} (MAE={ev['mae']:.2f}, R²={ev['r2']:.2f})")
    axes[0].set_ylabel("actual speedup")
    _dataset_legend(axes[-1], colors)
    fig.suptitle(f"Estimated vs actual speedup (scope={scope})")
    return _save(fig, out / "est_vs_actual.png")


def _est_vs_actual_linear(
    csv_path: str, rows: list[dict[str, str]], scope: str, out: Path
) -> Path:
    """Same figure for the simple per-mode linear model (speedup = a + b·β + c·PBE)."""
    n = max(int(r["n"]) for r in rows)  # best-performing n in the grid
    fits = {m: fit_linear_speedup(csv_path, scope=scope, mode=m, n=n)
            for m in ("depth", "width")}
    colors = _ds_colors([row["dataset"] for f in fits.values() for row in f["table"]])
    feat_cols = (("beta", f"{scope}_heaps_beta"), ("pbe_mean_at_1", f"{scope}_pbe_mean_at_1"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey="row", sharex="col")
    for row_axes, mode in zip(axes, ("depth", "width"), strict=True):
        f = fits[mode]
        coef = {"beta": f["b"], "pbe_mean_at_1": f["c"]}
        for ax, (key, label) in zip(row_axes, feat_cols, strict=True):
            other = "pbe_mean_at_1" if key == "beta" else "beta"
            other_bar = sum(row[other] for row in f["table"]) / len(f["table"])
            vals = [row[key] for row in f["table"]]
            pad = (max(vals) - min(vals)) * 0.08
            xs = (min(vals) - pad, max(vals) + pad)
            ax.plot(xs, [f["a"] + coef[key] * x + coef[other] * other_bar for x in xs],
                    color="gray", lw=1, ls="--")
            for row in f["table"]:
                c = colors[row["dataset"]]
                ax.plot([row[key]] * 2, [row["predict"], row["speedup"]], color=c, lw=1)
                ax.scatter(row[key], row["speedup"], s=70, color=c, marker="o",
                           edgecolors="black")
                ax.scatter(row[key], row["predict"], s=55, color=c, marker="D",
                           edgecolors="black")
            ax.set(xlabel=label)
        row_axes[0].set_ylabel("speedup")
        row_axes[0].set_title(f"{mode} (MAE={f['mae']:.2f}, R²={f['r2']:.2f})\n{f['formula']}",
                              fontsize=9)
        row_axes[1].set_title(mode, fontsize=9)
    kind_handles = [
        plt.Line2D([], [], marker="o", ls="", color="black", markerfacecolor="white",
                   markersize=9, label="actual"),
        plt.Line2D([], [], marker="D", ls="", color="black", markerfacecolor="white",
                   markersize=8, label="predicted"),
        plt.Line2D([], [], color="gray", ls="--", label="fit line @ mean of other feature"),
    ]
    axes[0][0].legend(handles=kind_handles, fontsize=8)
    _dataset_legend(axes[0][-1], colors)
    fig.suptitle(f"Linear fit: speedup vs workload features (scope={scope}, "
                 f"n={fits['depth']['n']}; vertical line = residual)")
    return _save(fig, out / "est_vs_actual_linear.png")


def _speedup_by_dataset(rows: list[dict[str, str]], out: Path) -> Path:
    datasets = sorted({r["dataset"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, mode in enumerate(("depth", "width")):
        best = [max(float(r["actual_speedup"]) for r in rows
                    if r["dataset"] == d and r["mode"] == mode) for d in datasets]
        ax.bar([x + (i - 0.5) * 0.4 for x in range(len(datasets))], best, width=0.4,
               color=_MODE_COLOR[mode], label=mode)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="best actual speedup", title="Best speedup per dataset (over n)")
    ax.legend()
    return _save(fig, out / "speedup_by_dataset.png")


def _acceptance_vs_features(rows: list[dict[str, str]], scope: str, out: Path) -> Path:
    feats = ["repetition_rate", "heaps_beta", "pbe_p90_at_1"]
    colors = _ds_colors([r["dataset"] for r in rows])
    fig, axes = plt.subplots(2, len(feats), figsize=(13, 7), sharey=True, sharex="col")
    for row_axes, mode in zip(axes, ("depth", "width"), strict=True):
        for ax, feat in zip(row_axes, feats, strict=True):
            for r in rows:
                if r["mode"] != mode:
                    continue
                ax.scatter(float(r[f"{scope}_{feat}"]), float(r["acceptance_rate"]), s=30,
                           color=colors[r["dataset"]], marker=_MODE_MARKER[mode],
                           edgecolors="black", linewidths=0.3)
        row_axes[0].set_ylabel(f"{mode}\nacceptance rate")
    for ax, feat in zip(axes[-1], feats, strict=True):
        ax.set_xlabel(f"{scope}_{feat}")
    _dataset_legend(axes[0][-1], colors)
    fig.suptitle("Acceptance vs workload features")
    return _save(fig, out / "acceptance_vs_features.png")


def _speedup_vs_n(rows: list[dict[str, str]], out: Path) -> Path:
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("depth", "width"), strict=True):
        for ds in datasets:
            pts = sorted((int(r["n"]), float(r["actual_speedup"])) for r in rows
                         if r["dataset"] == ds and r["mode"] == mode)
            ax.plot(*zip(*pts, strict=True), marker="o", ms=3, lw=1, label=ds)
        ax.set(xlabel="n-gram order", title=mode)
        ax.set_xticks(sorted({int(r["n"]) for r in rows}))
    axes[0].set(ylabel="actual speedup")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Speedup vs n")
    return _save(fig, out / "speedup_vs_n.png")


def _features_by_family(rows: list[dict[str, str]], out: Path) -> Path:
    per_ds = {r["dataset"]: r for r in reversed(rows)}  # features are constant per dataset
    ds_sorted = sorted(per_ds)  # id prefix (code_/math_/…) groups families together
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, feat, label in ((axes[0], "heaps_beta", "Heaps β"),
                            (axes[1], "pbe_p90_at_1", "PBE_P90@1")):
        for i, ds in enumerate(ds_sorted):
            color = _FAMILY_COLOR.get(ds.split("_")[0], "tab:gray")
            ax.bar(i - 0.2, float(per_ds[ds][f"ctx_{feat}"]), width=0.4, color=color)
            ax.bar(i + 0.2, float(per_ds[ds][f"gen_{feat}"]), width=0.4, color=color,
                   alpha=0.45, hatch="//")
        ax.set_xticks(range(len(ds_sorted)), ds_sorted, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=label)
    from matplotlib.patches import Patch

    handles = [Patch(color=c, label=f) for f, c in _FAMILY_COLOR.items()]
    handles += [Patch(facecolor="gray", label="ctx"),
                Patch(facecolor="gray", alpha=0.45, hatch="//", label="gen")]
    axes[1].legend(handles=handles, fontsize=8)
    fig.suptitle("Workload features per dataset (color = family; solid = ctx, hatched = gen)")
    return _save(fig, out / "features_by_family.png")


def _workload_character(rows: list[dict[str, str]], scope: str, out: Path) -> Path:
    """PBE@k mean vs Heaps β — color per dataset, marker per family (pre-refactor plot)."""
    per_ds = {r["dataset"]: r for r in reversed(rows)}
    datasets = sorted(per_ds)
    colors = _ds_colors(datasets)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, k in zip(axes, (1, 2, 4), strict=True):
        for ds in datasets:
            r = per_ds[ds]
            x, y = float(r[f"{scope}_pbe_mean_at_{k}"]), float(r[f"{scope}_heaps_beta"])
            ax.scatter(x, y, color=colors[ds], s=140, edgecolors="black", label=ds,
                       marker=_FAMILY_MARKER.get(ds.split("_")[0], "o"))
            ax.annotate(ds.split("_", 1)[-1], (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7)
        ax.set(xlabel=f"PBE@{k} mean (→ harder to draft)", title=f"PBE@{k}")
    axes[0].set_ylabel("Heaps' β (→ more diverse vocab)")
    leg_ds = axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                             title="dataset")
    fam_handles = [plt.Line2D([], [], marker=m, color="black", ls="", markersize=9,
                              markerfacecolor="white", label=fam)
                   for fam, m in _FAMILY_MARKER.items()]
    axes[-1].add_artist(leg_ds)
    axes[-1].legend(handles=fam_handles, bbox_to_anchor=(1.02, 0.4), loc="upper left",
                    fontsize=8, title="family (marker)")
    fig.suptitle(f"Workload Character (scope={scope}; bottom-left = best for n-gram spec decode)")
    return _save(fig, out / "workload_character.png")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_plots(csv_path: str, out_dir: str = "artifacts/plots", scope: str = "ctx") -> list[Path]:
    """Render the four M5 plots from a benchmark CSV; returns the written paths."""
    rows = _spec_rows(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        _est_vs_actual(rows, csv_path, scope, out),
        _est_vs_actual_linear(csv_path, rows, scope, out),
        _speedup_by_dataset(rows, out),
        _acceptance_vs_features(rows, scope, out),
        _speedup_vs_n(rows, out),
        _features_by_family(rows, out),
        _workload_character(rows, scope, out),
    ]
