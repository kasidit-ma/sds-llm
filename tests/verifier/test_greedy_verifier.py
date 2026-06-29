from verifier.greedy_verifier import GreedyVerifier

V = GreedyVerifier()


def test_partial_match():
    assert V.verify([4, 2, 5], [4, 2, 9]) == 2


def test_full_match():
    assert V.verify([4, 2, 5], [4, 2, 5]) == 3


def test_no_match():
    assert V.verify([4, 2, 5], [9, 9]) == 0


def test_empty_draft_or_target():
    assert V.verify([4, 2, 5], []) == 0
    assert V.verify([], [1, 2]) == 0


def test_verify_best_picks_longest():
    idx, acc = V.verify_best([1, 2, 3], [[1, 9], [1, 2, 9], [5]])
    assert (idx, acc) == (1, 2)


def test_verify_best_no_candidates():
    assert V.verify_best([1, 2, 3], []) == (0, 0)
