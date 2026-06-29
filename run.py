"""Benchmark n-gram speculative decoding on 12 datasets (simulation, no GPU needed).

Re-tokenizes each dataset via workload/loader.py + datasets.yaml, runs
benchmark_sim (GreedyVerifier against ground truth), and writes
workload/bench_results.json with actual_speedup per dataset.

Usage:
    cd /path/to/sds-llm
    python run.py [--only <dataset_id>] [--draft-len 4] [--max-samples 200]
"""
import argparse, importlib.util, json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run")

ROOT = Path(__file__).parent
WORKLOAD = ROOT / "workload"
SRC = ROOT / "src"


# ── cross-directory import helpers ────────────────────────────────────────────

def _load(reg_name, path):
    """Load a module from path and register it as reg_name in sys.modules."""
    spec = importlib.util.spec_from_file_location(reg_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[reg_name] = mod
    spec.loader.exec_module(mod)
    return mod


# drafter
_load("drafter_base", SRC / "drafter/base.py")
sys.modules["base"] = sys.modules["drafter_base"]    # ngram.py does `from base import Drafter`
NgramDrafter = _load("ngram", SRC / "drafter/ngram.py").NgramDrafter

# verifier
_load("verifier_base", SRC / "verifier/base.py")
sys.modules["base"] = sys.modules["verifier_base"]   # greedy.py does `from base import Verifier`
GreedyVerifier = _load("greedy", SRC / "verifier/greedy.py").GreedyVerifier

# spec_bench — iter_samples inlined here to avoid bench.py's flat `from data import` failing
def _iter_samples(token_stream, context_len=64, stride=32):
    n = len(token_stream)
    i = 0
    while i + context_len * 2 <= n:
        yield token_stream[i:i + context_len], token_stream[i + context_len:i + context_len * 2]
        i += stride

_bench_mod = _load("bench", SRC / "spec_bench/bench.py")
_bench_sim_raw = _bench_mod.benchmark_sim

def benchmark_sim(drafter, verifier, tokens, **kw):
    return _bench_sim_raw(drafter, verifier, tokens, _iter_samples=_iter_samples, **kw)

# workload loader
sys.path.insert(0, str(WORKLOAD))
from loader import iter_texts  # noqa: E402


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single dataset id")
    ap.add_argument("--draft-len", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--context-len", type=int, default=64)
    ap.add_argument("--prefix-len", type=int, default=3, help="n-gram prefix length")
    args = ap.parse_args()

    import yaml
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((WORKLOAD / "datasets.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=True)

    results = {}
    out_path = WORKLOAD / "bench_results.json"
    if out_path.exists():
        results = json.loads(out_path.read_text())

    rows = []
    for ds in cfg["datasets"]:
        did = ds["id"]
        if args.only and did != args.only:
            continue
        if did in results:
            log.info("skip %s (cached)", did)
            rows.append((did, ds["family"], results[did]["actual_speedup"]))
            continue

        log.info("loading %s ...", did)
        try:
            tokens = [t for text in iter_texts(ds, cfg["max_samples"])
                      for t in tok.encode(text, add_special_tokens=False)]
        except Exception as e:
            log.error("%s failed: %s", did, e)
            continue

        if len(tokens) < args.context_len * 2:
            log.warning("%s: too few tokens (%d)", did, len(tokens))
            continue

        drafter = NgramDrafter(prefix_len=args.prefix_len)
        verifier = GreedyVerifier()
        res = benchmark_sim(drafter, verifier, tokens,
                            context_len=args.context_len,
                            max_samples=args.max_samples,
                            draft_len=args.draft_len)
        res["family"] = ds["family"]
        results[did] = res
        out_path.write_text(json.dumps(results, indent=2))
        log.info("%s  speedup=%.3f  mean_accepted=%.2f",
                 did, res["actual_speedup"], res["mean_accepted_per_step"])
        rows.append((did, ds["family"], res["actual_speedup"]))

    if rows:
        print(f"\n{'dataset':<28} {'family':<10} {'actual_speedup':>14}")
        print("-" * 55)
        for did, fam, sp in sorted(rows, key=lambda x: -x[2]):
            print(f"{did:<28} {fam:<10} {sp:>14.3f}")
        print(f"\nresults saved → {out_path}")


if __name__ == "__main__":
    main()
