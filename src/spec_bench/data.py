"""Load HuggingFace datasets into token-id sequences, driven by configs/dataset.yaml.

Each entry declares either ``context_columns`` + ``target_columns`` (prompt vs generation —
gives a prefill boundary) or a single ``text_column`` (pure continuation, boundary 0).
The tokenizer is swappable via the config's ``tokenizer`` key (default ``Qwen/Qwen2.5-7B``).
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dataset.yaml"


class Sample(NamedTuple):
    """tokens[:boundary] = context (prefill), tokens[boundary:] = generation target."""

    tokens: list[int]
    boundary: int


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_tokenizer(cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["tokenizer"])


def _find_entry(cfg: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for entry in cfg["datasets"]:
        if entry["id"] == dataset_id:
            return entry
    known = ", ".join(e["id"] for e in cfg["datasets"])
    raise KeyError(f"unknown dataset id {dataset_id!r}; known: {known}")


def _col_text(row: dict[str, Any], col: str) -> str:
    """Column value as text; supports dotted keys ("answers.input_text") and joins lists."""
    value: Any = row
    for part in col.split("."):
        value = value.get(part, "") if isinstance(value, dict) else ""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value) if value else ""


def _row_parts(entry: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    """(context_text, target_text) — target is what the simulator 'generates'."""
    if "context_columns" in entry:
        ctx = "\n".join(t for c in entry["context_columns"] if (t := _col_text(row, c)))
        tgt = "\n".join(t for c in entry["target_columns"] if (t := _col_text(row, c)))
        return ctx, tgt
    return "", _col_text(row, entry["text_column"])


def peek(
    dataset_id: str, cfg: dict[str, Any] | None = None, n_rows: int = 1
) -> list[tuple[dict[str, Any], str, str]]:
    """Return the first raw rows and, for each, the (context, target) text split.

    Use this to check a dataset's columns/content and tune ``context_columns`` /
    ``target_columns`` / ``text_column`` in configs/dataset.yaml. No tokenizer needed.
    """
    from datasets import load_dataset

    cfg = cfg or load_config()
    entry = _find_entry(cfg, dataset_id)
    ds = load_dataset(
        entry["hf_path"],
        entry.get("hf_config"),
        split=entry["split"],
        streaming=entry.get("streaming", False),
    )
    out: list[tuple[dict[str, Any], str, str]] = []
    for row in ds:
        out.append((dict(row), *_row_parts(entry, row)))  # type: ignore[arg-type]
        if len(out) >= n_rows:
            break
    return out


def load_tokens(
    dataset_id: str,
    cfg: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> list[Sample]:
    """Return one ``Sample`` (token ids + prefill boundary) per row, capped at ``max_samples``."""
    from datasets import load_dataset

    cfg = cfg or load_config()
    tok = tokenizer if tokenizer is not None else load_tokenizer(cfg)
    entry = _find_entry(cfg, dataset_id)

    ds = load_dataset(
        entry["hf_path"],
        entry.get("hf_config"),
        split=entry["split"],
        streaming=entry.get("streaming", False),
    )

    max_samples = cfg.get("max_samples", 5000)
    sequences: list[Sample] = []
    for row in ds:
        ctx, tgt = _row_parts(entry, row)  # type: ignore[arg-type]
        if not tgt.strip():
            continue
        # ponytail: context/target tokenized separately — the seam token may differ from a
        # joint encode; irrelevant at simulator granularity
        ctx_ids = tok.encode(ctx + "\n") if ctx else []
        sequences.append(Sample(ctx_ids + tok.encode(tgt), len(ctx_ids)))
        if len(sequences) >= max_samples:
            break

    total = sum(len(s.tokens) for s in sequences)
    if total < cfg.get("min_tokens", 0):
        logger.warning(
            "%s: only %d tokens (< min_tokens=%d) — Heaps β may be unreliable",
            dataset_id,
            total,
            cfg["min_tokens"],
        )
    return sequences
