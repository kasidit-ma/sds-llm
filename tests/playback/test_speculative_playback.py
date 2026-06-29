from collections.abc import Sequence

from drafter.ngram_drafter import NGramDrafter
from interface.abstract_drafter import AbstractDrafter
from metrics.playback_metrics import compute_metrics
from playback.speculative_playback import SpeculativePlayback


class WrongDrafter(AbstractDrafter):
    """Always proposes tokens that never match the target."""

    def build_datastore(self, corpus_tokens: Sequence[int]) -> None:
        pass

    def propose(self, context, K, mode="depth"):
        return [999] * K


def test_baseline_one_token_per_step():
    log = SpeculativePlayback().run([1, 2, 3, 4], drafter=None)
    assert log.steps == 4
    assert log.total_tokens == 4
    assert compute_metrics(log).actual_speedup == 1.0


def test_useless_drafter_no_speedup():
    target = [1, 2, 3, 4, 5]
    log = SpeculativePlayback().run(target, drafter=WrongDrafter(), K=4, mode="depth")
    assert log.steps == len(target)  # only the free token advances each step
    assert log.accepted == 0
    assert log.drafted > 0
    assert compute_metrics(log).actual_speedup == 1.0


def test_perfect_depth_drafter_speeds_up():
    target = [1, 2, 1, 2, 1, 2, 1, 2]
    d = NGramDrafter(n=2)
    d.build_datastore(target)
    log = SpeculativePlayback().run(target, drafter=d, K=10, mode="depth")
    assert log.steps < len(target)
    assert compute_metrics(log).actual_speedup > 1.0


def test_width_mode_runs_and_speeds_up():
    target = [1, 2, 1, 2, 1, 2, 1, 2]
    d = NGramDrafter(n=2)
    d.build_datastore(target)
    log = SpeculativePlayback().run(target, drafter=d, K=10, mode="width")
    assert log.steps < len(target)
    assert compute_metrics(log).actual_speedup > 1.0
