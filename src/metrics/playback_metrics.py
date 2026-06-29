"""Turn a raw StepLog into reported metrics (acceptance rate, speedup, ...)."""

from dataclasses import asdict, dataclass

from interface.abstract_playback import StepLog


@dataclass
class PlaybackMetrics:
    total_tokens: int
    baseline_steps: int
    speculative_steps: int
    drafted_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    max_accepted_per_step: int
    avg_accepted_per_step: float
    actual_speedup: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(log: StepLog) -> PlaybackMetrics:
    baseline_steps = log.total_tokens  # baseline emits one token per step
    accepts = log.per_step_accepted
    return PlaybackMetrics(
        total_tokens=log.total_tokens,
        baseline_steps=baseline_steps,
        speculative_steps=log.steps,
        drafted_tokens=log.drafted,
        accepted_tokens=log.accepted,
        rejected_tokens=log.rejected,
        acceptance_rate=(log.accepted / log.drafted) if log.drafted else 0.0,
        max_accepted_per_step=max(accepts) if accepts else 0,
        avg_accepted_per_step=(sum(accepts) / len(accepts)) if accepts else 0.0,
        actual_speedup=(baseline_steps / log.steps) if log.steps else 0.0,
    )
