"""Load HuggingFace datasets into token-id sequences, driven by configs/dataset.yaml.

One ``load_tokens`` covers every dataset entry via its config fields (``text_column`` or
``text_combine``, optional ``hf_config``/``streaming``). The tokenizer is swappable via the
config's ``tokenizer`` key (default ``Qwen/Qwen2.5-7B``).
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dataset.yaml"


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


def _row_text(entry: dict[str, Any], row: dict[str, Any]) -> str:
    if "text_combine" in entry:
        parts = [str(row[c]) for c in entry["text_combine"] if row.get(c)]
        return "\n".join(parts)
    return str(row.get(entry["text_column"], ""))


def load_tokens(
    dataset_id: str,
    cfg: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> list[list[int]]:
    """Return one token-id list per sample for ``dataset_id`` (capped at ``max_samples``)."""
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
    sequences: list[list[int]] = []
    for row in ds:
        text = _row_text(entry, row)  # type: ignore[arg-type]
        if not text.strip():
            continue
        sequences.append(tok.encode(text))
        if len(sequences) >= max_samples:
            break

    total = sum(len(s) for s in sequences)
    if total < cfg.get("min_tokens", 0):
        logger.warning(
            "%s: only %d tokens (< min_tokens=%d) — Heaps β may be unreliable",
            dataset_id,
            total,
            cfg["min_tokens"],
        )
    return sequences
