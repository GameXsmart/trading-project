"""Sliced performance metrics.

"This model is 54% accurate" is not a finding. It is an average over regimes in which
the model behaves differently, and averaging those together destroys exactly the
information that would tell you when to trust it. "This model has skill on BTC 4h in
low-volatility regimes and none elsewhere" is a claim someone can act on, and it is the
only kind this module produces.

Five dimensions, each recorded at prediction time rather than reconstructed later:
asset, timeframe, horizon, regime, and volatility bucket. Reconstruction would mean
re-deriving what the market looked like then, and any bug in that derivation would
silently re-slice every past result — including the ones already used to set weights.

Every slice carries its sample size, and slices below a floor report *insufficient
evidence* rather than a number. A Brier score over eleven observations is not a
measurement; printing it next to one over four thousand invites them to be compared.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from mie.learning.records import ResolvedOutcome
from mie.models.types import Outcome

__all__ = ["MetricsTable", "SliceMetrics", "slice_outcomes"]

#: Below this many resolved outcomes a slice reports insufficient evidence.
_MIN_SLICE = 30


@dataclass(slots=True)
class SliceMetrics:
    """How one model performed on one slice."""

    model_id: str
    dimension: str
    value: str
    count: int
    brier: float
    log_loss: float
    accuracy: float
    #: Share of outcomes that were UP / FLAT / DOWN, so a Brier score is interpretable.
    class_balance: dict[str, float] = field(default_factory=dict)
    mean_probability_of_truth: float = 0.0

    @property
    def has_evidence(self) -> bool:
        return self.count >= _MIN_SLICE

    @property
    def verdict(self) -> str:
        if not self.has_evidence:
            return f"insufficient evidence ({self.count} outcomes, need {_MIN_SLICE})"
        return f"brier {self.brier:.4f} over {self.count} outcomes"

    def summary(self) -> str:
        return f"{self.model_id:14} {self.dimension}={self.value:18} {self.verdict}"


@dataclass(slots=True)
class MetricsTable:
    """Every slice, for every model."""

    slices: list[SliceMetrics] = field(default_factory=list)

    def for_model(self, model_id: str) -> list[SliceMetrics]:
        return [s for s in self.slices if s.model_id == model_id]

    def for_dimension(self, dimension: str) -> list[SliceMetrics]:
        return [s for s in self.slices if s.dimension == dimension]

    def with_evidence(self) -> list[SliceMetrics]:
        return [s for s in self.slices if s.has_evidence]

    def best(self, dimension: str = "overall") -> SliceMetrics | None:
        """Lowest Brier among slices with enough evidence on this dimension."""
        candidates = [s for s in self.for_dimension(dimension) if s.has_evidence]
        return min(candidates, key=lambda s: s.brier, default=None)

    def report(self, dimension: str | None = None) -> str:  # pragma: no cover
        chosen = self.for_dimension(dimension) if dimension else self.slices
        lines = ["Sliced metrics", "=" * 78]
        lines.extend("  " + s.summary() for s in sorted(chosen, key=lambda s: (s.model_id, s.value)))
        thin = len(chosen) - len([s for s in chosen if s.has_evidence])
        if thin:
            lines.append(f"  ({thin} slices reported insufficient evidence)")
        return "\n".join(lines)


def slice_outcomes(outcomes: Sequence[ResolvedOutcome]) -> MetricsTable:
    """Compute metrics across every slicing dimension at once."""
    table = MetricsTable()
    if not outcomes:
        return table

    buckets: dict[tuple[str, str, str], list[ResolvedOutcome]] = defaultdict(list)
    for outcome in outcomes:
        model = outcome.model_id
        buckets[(model, "overall", "all")].append(outcome)
        buckets[(model, "asset", outcome.asset)].append(outcome)
        buckets[(model, "timeframe", str(outcome.timeframe))].append(outcome)
        buckets[(model, "horizon", f"{outcome.horizon_bars} bars")].append(outcome)
        buckets[(model, "regime", outcome.regime)].append(outcome)
        buckets[(model, "volatility", outcome.volatility_bucket)].append(outcome)

    for (model_id, dimension, value), group in buckets.items():
        count = len(group)
        counts: dict[Outcome, int] = defaultdict(int)
        for outcome in group:
            counts[outcome.realised_direction] += 1
        table.slices.append(
            SliceMetrics(
                model_id=model_id,
                dimension=dimension,
                value=value,
                count=count,
                brier=round(sum(o.brier for o in group) / count, 5),
                log_loss=round(sum(o.log_loss for o in group) / count, 5),
                accuracy=round(sum(1 for o in group if o.correct) / count, 4),
                class_balance={
                    o.value: round(counts[o] / count, 4) for o in Outcome
                },
                mean_probability_of_truth=round(
                    sum(o.probability_of_truth for o in group) / count, 4
                ),
            )
        )
    return table
