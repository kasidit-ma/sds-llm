import pytest


def test_heaps_beta():
    from workload.heaps import heaps_beta

    assert heaps_beta([1, 1, 2, 3]) > 0


def test_pbe_and_repetition():
    from workload.pbe import pbe_p90_at_k
    from workload.repetition import repetition_rate

    assert pbe_p90_at_k([1, 2, 3], k=1, prefix_len=3) >= 0
    assert 0.0 <= repetition_rate([1, 2, 1, 2], n=2) <= 1.0
    assert repetition_rate([1, 2, 1, 2], n=2) == pytest.approx(1 - 2 / 3)


def test_characterize_keys():
    from workload.characterizer import characterize

    feats = characterize([1, 2, 3] * 20)
    assert set(feats) == {
        "heaps_beta",
        "repetition_rate",
        "pbe_p90_at_1",
        "pbe_p90_at_2",
        "pbe_p90_at_4",
        "pbe_mean_at_1",
        "pbe_mean_at_2",
        "pbe_mean_at_4",
    }
