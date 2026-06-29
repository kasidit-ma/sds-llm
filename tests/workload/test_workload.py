import pytest

pytestmark = pytest.mark.skip(reason="Milestone M2 — workload character not implemented yet")


def test_heaps_beta():
    from workload.heaps import heaps_beta

    assert heaps_beta([1, 1, 2, 3]) > 0


def test_pbe_and_repetition():
    from workload.pbe import pbe_p90_at_k
    from workload.repetition import repetition_rate

    assert pbe_p90_at_k([1, 2, 3], k=1, prefix_len=3) >= 0
    assert 0.0 <= repetition_rate([1, 2, 1, 2], n=2) <= 1.0
