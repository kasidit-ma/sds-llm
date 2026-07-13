from typing import cast

from drafter.ngram_drafter import NGramDrafter


def _drafter(corpus, n=2):
    d = NGramDrafter(n=n)
    d.build_datastore(corpus)
    return d


def test_depth_chain_follows_most_frequent():
    # (1,)->2, (2,)->3, (3,)->1 cycle
    d = _drafter([1, 2, 3, 1, 2, 3, 1, 2])
    assert d.propose([1], K=5, mode="depth") == [2, 3, 1, 2, 3]


def test_depth_respects_budget_K():
    d = _drafter([1, 2, 3, 1, 2, 3, 1, 2])
    assert d.propose([1], K=2, mode="depth") == [2, 3]


def test_missing_key_returns_empty():
    d = _drafter([1, 2, 3, 1, 2, 3])
    assert d.propose([99], K=5, mode="depth") == []


def test_empty_corpus_no_error():
    d = _drafter([])
    assert d.propose([1], K=5, mode="depth") == []
    assert d.propose([1], K=5, mode="width") == []


def test_width_returns_multiple_bounded_sequences():
    # (1,) branches to 2 (freq 2) and 3 (freq 1)
    d = _drafter([1, 2, 1, 3, 1, 2])
    seqs = cast("list[list[int]]", d.propose([1], K=10, mode="width"))
    assert seqs == [[2, 1], [3, 1]]
    assert len(seqs) <= 5
    assert all(len(s) <= 2 for s in seqs)


def test_invalid_n_and_mode():
    import pytest

    with pytest.raises(ValueError):
        NGramDrafter(n=0)
    with pytest.raises(ValueError):
        _drafter([1, 2, 3]).propose([1], K=3, mode="sideways")
