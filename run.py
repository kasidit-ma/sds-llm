#!/usr/bin/env python
"""Run the n-gram speculative-decoding simulator on one dataset and print metrics.

Example:
    python run.py --dataset code_humaneval --n 3 --K 10 --S 5 --T 2
    python run.py --benchmark          # full grid from configs/experiment.yaml -> CSV
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from spec_bench.benchmark import run_benchmark, run_mode  # noqa: E402
from spec_bench.data import load_config, load_tokenizer, load_tokens, peek  # noqa: E402


def _trunc(s: object, n: int = 300) -> str:
    text = str(s).replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + " …"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="code_humaneval")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--S", type=int, default=5)  # reserved for width tuning (M5)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--benchmark", action="store_true",
                    help="run the full grid from configs/experiment.yaml and write CSV")
    ap.add_argument("--plots", action="store_true",
                    help="render plots from the benchmark CSV into artifacts/plots/")
    ap.add_argument("--fit", action="store_true",
                    help="print the linear speedup fit (a + b*beta + c*pbe) table per mode")
    ap.add_argument("--features", action="store_true",
                    help="print the dataset's workload features and exit")
    ap.add_argument("--peek", type=int, nargs="?", const=1, default=None, metavar="N",
                    help="show N raw rows (columns + content + combined text) and exit")
    args = ap.parse_args()

    if args.peek:
        cfg = load_config()
        entry = next(e for e in cfg["datasets"] if e["id"] == args.dataset)
        ctx_cols = entry.get("context_columns", [])
        tgt_cols = entry.get("target_columns") or [entry["text_column"]]
        print(f"{args.dataset}  ({entry['hf_path']}, split={entry['split']})")
        print(f"context: {ctx_cols}  target: {tgt_cols}  (tune in configs/dataset.yaml)\n")
        for i, (row, ctx, tgt) in enumerate(peek(args.dataset, cfg, args.peek)):
            print(f"--- row {i} ---")
            for col, value in row.items():
                print(f"{col:<22}{_trunc(value)}")
            print(f"{'=> context (prefill)':<22}{_trunc(ctx)}")
            print(f"{'=> target (decoded)':<22}{_trunc(tgt)}\n")
        return

    if args.benchmark or args.plots or args.fit:
        exp_cfg = yaml.safe_load(
            (Path(__file__).resolve().parent / "configs" / "experiment.yaml").read_text()
        )
        if args.benchmark:
            run_benchmark(exp_cfg, exp_cfg["output_csv"])
            print(f"Wrote {exp_cfg['output_csv']}")
        if args.plots:
            from spec_bench.plots import make_plots

            for path in make_plots(exp_cfg["output_csv"]):
                print(f"Wrote {path}")
        if args.fit:
            from estimation.curve_fitting import fit_linear_speedup

            for mode in ("depth", "width"):
                fit = fit_linear_speedup(exp_cfg["output_csv"], mode=mode, n=args.n)
                print(f"\n[{mode}, n={args.n}]  {fit['formula']}"
                      f"   (MAE={fit['mae']:.3f}, R²={fit['r2']:.3f})")
                print(f"{'dataset':<22}{'beta':>8}{'pbe@1':>8}{'speedup':>9}{'predict':>9}")
                for row in fit["table"]:
                    print(f"{row['dataset']:<22}{row['beta']:>8.3f}{row['pbe_mean_at_1']:>8.3f}"
                          f"{row['speedup']:>9.3f}{row['predict']:>9.3f}")
        return

    cfg = load_config()
    tokenizer = load_tokenizer(cfg)
    print(f"Loading {args.dataset} with tokenizer {cfg['tokenizer']} ...")
    samples = load_tokens(args.dataset, cfg, tokenizer)
    print(f"{len(samples)} samples, {sum(len(s.tokens) for s in samples)} tokens\n")

    if args.features:
        from workload.characterizer import characterize

        for scope, toks in (
            ("ctx (observable before generation)",
             [t for s in samples for t in s.tokens[: s.boundary]]),
            ("gen (the generated text itself)",
             [t for s in samples for t in s.tokens[s.boundary :]]),
        ):
            print(f"[{scope}]")
            for name, value in characterize(toks, cfg).items():
                print(f"  {name:<20}{value:.4f}")
        return

    results = {
        "baseline": run_mode(samples, args.n, args.K, None),
        "depth": run_mode(samples, args.n, args.K, "depth"),
        "width": run_mode(samples, args.n, args.K, "width"),
    }

    cols = ["speculative_steps", "acceptance_rate", "avg_accepted_per_step", "actual_speedup"]
    print(f"{'mode':<10}" + "".join(f"{c:>22}" for c in cols))
    for mode, m in results.items():
        d = m.as_dict()
        print(f"{mode:<10}" + "".join(f"{d[c]:>22.4f}" for c in cols))


if __name__ == "__main__":
    main()
