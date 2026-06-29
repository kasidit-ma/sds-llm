"""Offline benchmark grid. Stub — Milestone 3.

Runs the simulator across datasets x modes {baseline, depth, width} x n x K, joins each row with
its workload character, and writes artifacts/results/benchmark_results.csv (the ground-truth
acceptance/speedup table the predictor is fit on).
"""

from pathlib import Path
from typing import Any


def run_benchmark(experiment_cfg: dict[str, Any], out_csv: str | Path) -> None:
    raise NotImplementedError("Milestone M3 — see plan")
