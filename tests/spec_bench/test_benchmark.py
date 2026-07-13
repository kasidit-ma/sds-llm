import csv

from spec_bench.data import Sample, _row_parts


def test_row_parts_nested_and_list_columns():
    entry = {"context_columns": ["story", "questions"], "target_columns": ["answers.input_text"]}
    row = {
        "story": "Once upon a time.",
        "questions": ["Who?", "Where?"],
        "answers": {"input_text": ["A cat", "Home"], "answer_start": [0, 5]},
    }
    ctx, tgt = _row_parts(entry, row)
    assert ctx == "Once upon a time.\nWho?\nWhere?"
    assert tgt == "A cat\nHome"


def test_run_benchmark_and_aggregate(tmp_path, monkeypatch):
    from spec_bench.benchmark import run_benchmark
    from spec_bench.speedup import aggregate_speedup

    monkeypatch.setattr(
        "spec_bench.benchmark.load_config",
        lambda: {"datasets": [{"id": "fake"}], "pbe_steps": [1, 2, 4], "pbe_prefix_len": 3},
    )
    monkeypatch.setattr("spec_bench.benchmark.load_tokenizer", lambda cfg: None)
    monkeypatch.setattr(
        "spec_bench.benchmark.load_tokens",
        lambda ds, cfg, tok: [Sample([1, 2, 3, 4] * 10, 4), Sample([5, 6, 7, 8] * 10, 0)],
    )

    out = tmp_path / "out.csv"
    run_benchmark(
        {"datasets": [], "modes": ["baseline", "depth", "width"], "n_values": [2], "K": 4}, out
    )

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3  # 1 baseline + 1 depth + 1 width
    for col in ["acceptance_rate", "actual_speedup", "ctx_heaps_beta", "gen_pbe_p90_at_1"]:
        assert col in rows[0]
    # baseline decodes only past each sample's boundary: (40-4) + (40-0)
    assert int(rows[0]["baseline_steps"]) == 76

    # resume: dataset already complete -> second run keeps rows, doesn't duplicate or re-run
    run_benchmark(
        {"datasets": [], "modes": ["baseline", "depth", "width"], "n_values": [2], "K": 4}, out
    )
    assert len(list(csv.DictReader(out.open()))) == 3

    summary = aggregate_speedup(rows)
    assert len(summary) == 3
    depth = next(s for s in summary if s["mode"] == "depth")
    assert depth["best_speedup"] > 1.0  # repetitive synthetic tokens must speed up
