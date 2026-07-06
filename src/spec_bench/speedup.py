"""Aggregate benchmark rows into per-dataset/per-mode summaries."""

from collections import defaultdict
from typing import Any


def aggregate_speedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["mode"])].append(row)
    out = []
    for (dataset, mode), grp in groups.items():
        speedups = [float(r["actual_speedup"]) for r in grp]  # rows may come from csv.DictReader
        best = max(grp, key=lambda r: float(r["actual_speedup"]))
        out.append({
            "dataset": dataset,
            "mode": mode,
            "mean_speedup": sum(speedups) / len(speedups),
            "best_speedup": float(best["actual_speedup"]),
            "best_n": int(best["n"]),
        })
    return out
