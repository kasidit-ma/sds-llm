"""Orchestrate: load -> tokenize -> characterize -> save json -> plot."""
import argparse, json, logging
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from loader import iter_texts
from characterizer import characterize

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("workload")

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single dataset id")
    ap.add_argument("--force", action="store_true", help="rerun even if json exists")
    ap.add_argument("--tokenizer", help="override tokenizer from yaml")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "datasets.yaml").read_text())
    if args.tokenizer:
        cfg["tokenizer"] = args.tokenizer
    RESULTS.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=True)

    for ds in cfg["datasets"]:
        if args.only and ds["id"] != args.only:
            continue
        out_path = RESULTS / f"{ds['id']}.json"
        if out_path.exists() and not args.force:
            log.info("skip %s (exists)", ds["id"])
            continue
        log.info("loading %s ...", ds["id"])
        try:
            docs = [tok.encode(t, add_special_tokens=False)
                    for t in iter_texts(ds, cfg["max_samples"])]
        except Exception as e:
            log.error("%s failed: %s", ds["id"], e)
            continue
        if not docs:
            log.error("%s: no texts", ds["id"])
            continue
        result = characterize(docs, cfg, ds["id"])
        result["family"] = ds["family"]
        out_path.write_text(json.dumps(result, indent=2))
        log.info("wrote %s (%d tokens, beta=%.3f)",
                 out_path.name, result["token_count"], result["heaps"]["beta"])

    if not args.no_plot:
        import plot
        plot.plot(RESULTS)


if __name__ == "__main__":
    main()
