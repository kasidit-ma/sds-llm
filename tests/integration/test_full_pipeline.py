"""End-to-end on synthetic tokens (no HuggingFace): drafter -> playback -> metrics."""

from drafter.ngram_drafter import NGramDrafter
from metrics.playback_metrics import compute_metrics
from playback.speculative_playback import SpeculativePlayback


def test_repetitive_workload_beats_baseline():
    target = [3, 1, 4, 1, 5] * 8  # repeating pattern -> n-grams predict well

    drafter = NGramDrafter(n=3)
    drafter.build_datastore(target)
    playback = SpeculativePlayback()

    baseline = compute_metrics(playback.run(target, drafter=None))
    spec = compute_metrics(playback.run(target, drafter=drafter, K=10, mode="depth"))

    assert baseline.actual_speedup == 1.0
    assert spec.actual_speedup > baseline.actual_speedup
    assert 0.0 < spec.acceptance_rate <= 1.0
    assert spec.speculative_steps < spec.baseline_steps
