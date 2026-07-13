"""Turn acceptance probability into estimated speedup.

E[A] = p + p^2 + ... + p^K ;  estimated_speedup = 1 + E[A].
"""


def expected_accepted(p: float, K: int) -> float:
    return sum(p**i for i in range(1, K + 1))


def estimated_speedup(p: float, K: int) -> float:
    return 1.0 + expected_accepted(p, K)
