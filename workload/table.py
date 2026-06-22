"""Comparison tables (PBE P90 + PBE mean) per dataset. Prints + writes summary.md."""
import json
from pathlib import Path


def _md_table(rows, header, cell):
    """header = column names list; cell(r) -> list of stringified values."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(r)) + " |" for r in rows]
    return "\n".join(lines)


def table(results_dir: Path, out_name: str = "workload_summary.md") -> str:
    rows = [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]
    rows.sort(key=lambda r: (r["family"], r["dataset_id"]))

    p90 = _md_table(
        rows, ["dataset", "family", "tokens", "β", "P90@1", "P90@2", "P90@4"],
        lambda r: [r["dataset_id"], r["family"], f"{r['token_count']:,}",
                   f"{r['heaps']['beta']:.3f}",
                   *[f"{r['pbe'][f'pbe@{n}']['p90']:.0f}" for n in (1, 2, 4)]])
    mean = _md_table(
        rows, ["dataset", "family", "mean@1", "mean@2", "mean@4"],
        lambda r: [r["dataset_id"], r["family"],
                   *[f"{r['pbe'][f'pbe@{n}']['mean']:.2f}" for n in (1, 2, 4)]])

    md = f"## PBE P90\n\n{p90}\n\n## PBE mean\n\n{mean}\n"
    (results_dir.parent / out_name).write_text(md)
    return md


if __name__ == "__main__":
    print(table(Path(__file__).parent / "results"))
