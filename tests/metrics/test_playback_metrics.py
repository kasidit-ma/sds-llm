from interface.abstract_playback import StepLog
from metrics.playback_metrics import compute_metrics


def test_metrics_arithmetic():
    log = StepLog(
        total_tokens=10,
        steps=5,
        drafted=8,
        accepted=6,
        rejected=2,
        per_step_accepted=[2, 1, 0, 3, 0],
    )
    m = compute_metrics(log)
    assert m.baseline_steps == 10
    assert m.speculative_steps == 5
    assert m.acceptance_rate == 0.75
    assert m.max_accepted_per_step == 3
    assert m.avg_accepted_per_step == 1.2
    assert m.actual_speedup == 2.0


def test_metrics_zero_guards():
    m = compute_metrics(StepLog(total_tokens=0, steps=0))
    assert m.acceptance_rate == 0.0
    assert m.actual_speedup == 0.0
    assert m.max_accepted_per_step == 0
