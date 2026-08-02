"""Flat per-dataset table: bench steps/speedup + characterization + full PBE stats.

Writes workload_full_table.csv (+ .md) next to this file.
"""
import csv
import json
from pathlib import Path

NS = (1, 2, 4)
STATS = ("mean", "median", "p90", "min", "max", "std")

HEADER = ["dataset_name", "baseline_steps", "spec_steps", "actual_speedup", "family",
          "token_count", "vocab_size", "heaps_beta"]
HEADER += [f"pbe@{n}_{s}" for n in NS for s in STATS]


def rows(results_dir: Path) -> list[dict]:
    bench_path = results_dir.parent / "bench_results.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}

    out = []
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        b = bench.get(r["dataset_id"], {})
        row = {
            "dataset_name": r["dataset_id"],
            "baseline_steps": b.get("baseline_steps"),
            "spec_steps": b.get("spec_steps"),
            "actual_speedup": b.get("actual_speedup"),
            "family": r["family"],
            "token_count": r["token_count"],
            "vocab_size": r["vocab_size"],
            "heaps_beta": r["heaps"]["beta"],
        }
        for n in NS:
            p = r["pbe"][f"pbe@{n}"]
            for s in STATS:
                row[f"pbe@{n}_{s}"] = p[s]
        out.append(row)

    out.sort(key=lambda r: (r["family"], r["dataset_name"]))
    return out


def _fmt(key, v):
    if v is None:
        return "—"
    if key in ("actual_speedup", "heaps_beta") or key.endswith(("_mean", "_std")):
        return f"{v:.3f}"
    if key.endswith(("_median", "_p90")):
        return f"{v:.1f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def table(results_dir: Path, out_stem: str = "workload_full_table") -> str:
    data = rows(results_dir)
    base = results_dir.parent / out_stem

    with (base.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(data)

    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(_fmt(k, r[k]) for k in HEADER) + " |" for r in data]
    md = "\n".join(lines)
    base.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    return md


if __name__ == "__main__":
    print(table(Path(__file__).parent / "results"))
