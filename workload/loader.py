"""Load HF dataset rows -> plain text strings, per datasets.yaml config."""
from typing import Iterator
from datasets import load_dataset


def _codecontests(row: dict) -> str:
    tests = row["public_tests"]
    tests_str = ("Input:\n" + "\n".join(tests["input"])
                 + "\nOutput:\n" + "\n".join(tests["output"]))
    sols_str = "\n".join(row["solutions"]["solution"])  # solutions is a dict-of-lists
    return row["description"] + "\n" + tests_str + "\n" + sols_str


def _row_to_text(cfg: dict, row: dict) -> str | None:
    if "text_serialize" in cfg:
        return _codecontests(row)  # ponytail: only codecontests serializer exists; add more when needed
    if "text_combine" in cfg:
        parts = [str(row[c]).strip() for c in cfg["text_combine"]]
        return "\n".join(p for p in parts if p)  # skip empty/optional fields
    return str(row[cfg["text_column"]])


def iter_texts(cfg: dict, max_samples: int) -> Iterator[str]:
    """yield one string per row, honoring filter/streaming. Skips blanks."""
    streaming = cfg.get("streaming", False)
    ds = load_dataset(cfg["hf_path"], cfg.get("hf_config"),
                      split=cfg["split"], streaming=streaming)
    filt = cfg.get("filter")
    n = 0
    for row in ds:
        if n >= max_samples:
            break
        if filt and any(row.get(k) != v for k, v in filt.items()):
            continue
        text = _row_to_text(cfg, row)
        if text and text.strip():
            n += 1
            yield text
