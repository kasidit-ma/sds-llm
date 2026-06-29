import pytest

pytestmark = pytest.mark.skip(reason="Milestone M4 — estimation/decision not implemented yet")


def test_estimated_speedup():
    from estimation.speedup_estimator import estimated_speedup

    assert estimated_speedup(0.5, K=10) > 1.0


def test_decision_policy():
    from decision.policy import decide

    out = decide({"pbe_p90_at_1": 1.0}, estimated_speedup=1.8)
    assert out["use_speculative"] is True
