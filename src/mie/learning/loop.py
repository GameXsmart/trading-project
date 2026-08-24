"""The learning loop, and an honest account of whether it learned anything.

The loop is four steps: predictions are written before their outcomes exist, outcomes
resolve from final candles once the horizon has elapsed, metrics are computed sliced,
and weights and calibration are updated from those metrics. Only the fourth step is
learning. The first three are bookkeeping, and §14 exists because bookkeeping is
routinely presented as the fourth.

So the loop's output is a :class:`LearningReport` whose central field is what
*changed*. If no weight moved and no calibration curve was adopted, the report says so
in those words. A system that ran, produced a page of numbers, and altered nothing has
not learned, and reporting the page of numbers as though it had is the specific
dishonesty the requirement names.

The report distinguishes three things that are easy to blur:

* **nothing to learn from** — too few resolved outcomes to say anything;
* **learned nothing** — enough evidence, and it did not support any change;
* **learned something** — a weight or a calibration curve moved, with the sample that
  moved it attached.

Only the third is learning. The second is a finding. The first is a waiting room.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from mie.core.logging import get_logger
from mie.core.timeframes import utcnow
from mie.core.types import Candle
from mie.ensemble.calibration import CalibrationLibrary, CalibrationRecord
from mie.learning.metrics import MetricsTable, slice_outcomes
from mie.learning.records import PredictionRecord, ResolvedOutcome
from mie.learning.weights import WeightKey, WeightLearner, WeightTable, WeightUpdate
from mie.models.types import Outcome, Prediction

log = get_logger(__name__)

__all__ = ["LearningLoop", "LearningReport", "OutcomeResolver"]

#: Resolved outcomes needed before the loop will attempt to learn at all.
_MIN_OUTCOMES = 60


@dataclass(frozen=True, slots=True)
class _Scored:
    """Adapter presenting a (record, outcome) pair the way the calibrator expects."""

    prediction: Prediction
    actual: Outcome


class OutcomeResolver:
    """Resolves predictions whose horizon has elapsed, from final candles only.

    Two rules, both of which sound obvious and are routinely broken.

    **Final candles only.** The bar covering the resolution instant is still forming at
    that instant. Reading it would score the forecast against an incomplete price, which
    is a look-ahead error pointing the other way — and one that flatters or punishes at
    random rather than systematically, which makes it harder to notice.

    **The stored threshold, not a fresh one.** The prediction recorded the band that
    separates a real move from noise. Recomputing it at resolution time would score the
    forecast against a different question from the one it answered.
    """

    def __init__(self, settle_bars: int = 1) -> None:
        self.settle_bars = max(0, settle_bars)

    def resolve(
        self,
        records: Sequence[PredictionRecord],
        candles_by_asset: Mapping[str, Sequence[Candle]],
        now: datetime | None = None,
    ) -> tuple[list[ResolvedOutcome], list[str]]:
        """Resolve what can be resolved. Returns outcomes and the ids left pending."""
        moment = now or utcnow()
        resolved: list[ResolvedOutcome] = []
        pending: list[str] = []

        indexed = {
            asset.upper(): sorted(
                (c for c in candles if c.is_final), key=lambda c: c.open_time
            )
            for asset, candles in candles_by_asset.items()
        }

        for record in records:
            if record.resolved:
                continue
            settle = record.timeframe.delta * self.settle_bars
            if not record.is_due(moment, settle):
                pending.append(record.prediction_id)
                continue
            if not record.verify():
                # Refused, not repaired. Whatever this row now says is not what the
                # model said, and scoring it would launder a corrupted record into a
                # performance number.
                log.error(
                    "prediction_hash_mismatch",
                    prediction=record.prediction_id,
                    model=record.model_id,
                )
                continue

            series = indexed.get(record.asset.upper(), [])
            exit_bar = _bar_covering(series, record.resolves_at, record.timeframe.delta)
            if exit_bar is None:
                pending.append(record.prediction_id)
                continue

            outcome = ResolvedOutcome.score(
                record, exit_bar.close, record.timeframe.close_time(exit_bar.open_time)
            )
            if outcome is None:
                continue
            resolved.append(outcome)
        return resolved, pending


def _bar_covering(
    candles: Sequence[Candle], moment: datetime, tolerance: timedelta | None = None
) -> Candle | None:
    """The final bar closing at ``moment``, or ``None`` if there is not one.

    Two rules, and the second was added because a test caught the first being
    insufficient on its own.

    **At or before, never after.** A bar closing later than the resolution instant
    contains price action the forecast was never asked about.

    **And no earlier than one bar.** "The last bar at or before the moment" sounds
    right and is not: when the resolving bar is missing — a gap, or a bar still forming
    — that rule silently reaches back to whatever *was* available and scores the
    forecast against a price from an arbitrary distance in the past. The outcome would
    look resolved and be fiction. Outside the tolerance the answer is that the price is
    not known yet, and the prediction stays pending.
    """
    window = tolerance if tolerance is not None else timedelta.max
    earliest = None if window is timedelta.max else moment - window
    chosen: Candle | None = None
    for candle in candles:
        if candle.close_time <= moment:
            chosen = candle
        else:
            break
    if chosen is None:
        return None
    if earliest is not None and chosen.close_time <= earliest:
        return None
    return chosen


@dataclass(slots=True)
class LearningReport:
    """What the loop did, and — centrally — whether anything changed."""

    resolved: int = 0
    pending: int = 0
    total_outcomes: int = 0
    metrics: MetricsTable = field(default_factory=MetricsTable)
    weights: WeightTable = field(default_factory=WeightTable)
    calibrations: list[CalibrationRecord] = field(default_factory=list)
    #: Records refused because their content hash did not match.
    corrupted: int = 0
    ran_at: datetime = field(default_factory=utcnow)

    @property
    def weight_changes(self) -> list[WeightUpdate]:
        return self.weights.changes()

    @property
    def adopted_calibrations(self) -> list[CalibrationRecord]:
        return [c for c in self.calibrations if c.is_usable]

    @property
    def had_enough_evidence(self) -> bool:
        return self.total_outcomes >= _MIN_OUTCOMES

    @property
    def learned(self) -> bool:
        """Whether the loop changed future behaviour. The only claim that counts."""
        return bool(self.weight_changes) or bool(self.adopted_calibrations)

    @property
    def verdict(self) -> str:
        if not self.had_enough_evidence:
            return (
                f"nothing to learn from yet: {self.total_outcomes} resolved outcomes, "
                f"need {_MIN_OUTCOMES}"
            )
        if not self.learned:
            return (
                f"learned nothing: {self.total_outcomes} resolved outcomes, no slice "
                f"crossed the evidence gate and no calibration curve improved on "
                f"held-out data"
            )
        return (
            f"learned: {len(self.weight_changes)} weight changes, "
            f"{len(self.adopted_calibrations)} calibration curves adopted, "
            f"from {self.total_outcomes} resolved outcomes"
        )

    def report(self) -> str:
        lines = [
            f"Learning loop @ {self.ran_at:%Y-%m-%d %H:%M}",
            "=" * 78,
            f"  resolved this run: {self.resolved}   still pending: {self.pending}",
            f"  outcomes in evidence: {self.total_outcomes}",
        ]
        if self.corrupted:
            lines.append(f"  REFUSED for hash mismatch: {self.corrupted}")
        lines.append("")
        for update in sorted(self.weight_changes, key=lambda u: -abs(u.delta))[:20]:
            lines.append("  " + update.summary())
        for record in self.adopted_calibrations:
            lines.append("  " + record.summary())
        lines.append("")
        lines.append(f"VERDICT: {self.verdict}")
        return "\n".join(lines)


class LearningLoop:
    """Resolve, measure, reweight, recalibrate — and report what actually moved."""

    def __init__(
        self,
        resolver: OutcomeResolver | None = None,
        learner: WeightLearner | None = None,
        calibration: CalibrationLibrary | None = None,
    ) -> None:
        self.resolver = resolver or OutcomeResolver()
        self.learner = learner or WeightLearner()
        self.calibration = calibration or CalibrationLibrary()

    def run(
        self,
        records: Sequence[PredictionRecord],
        candles_by_asset: Mapping[str, Sequence[Candle]],
        existing_outcomes: Sequence[ResolvedOutcome] = (),
        previous_weights: Mapping[WeightKey, float] | None = None,
        now: datetime | None = None,
    ) -> LearningReport:
        """One pass of the loop.

        ``existing_outcomes`` are outcomes resolved on previous runs. They matter: a
        loop that only ever learned from what it resolved *this* pass would forget
        everything between runs, and the sample would never grow past one horizon.
        """
        newly_resolved, pending = self.resolver.resolve(records, candles_by_asset, now)
        outcomes = [*existing_outcomes, *newly_resolved]

        report = LearningReport(
            resolved=len(newly_resolved),
            pending=len(pending),
            total_outcomes=len(outcomes),
            ran_at=now or utcnow(),
            corrupted=sum(1 for r in records if not r.resolved and not r.verify()),
        )
        if not outcomes:
            return report

        report.metrics = slice_outcomes(outcomes)

        if not report.had_enough_evidence:
            # Measuring is free and worth showing; changing weights on this little
            # evidence is not, and doing it anyway is how a loop starts chasing noise.
            return report

        report.weights = WeightTable(
            updates=self.learner.learn(outcomes, previous_weights)
        )
        report.calibrations = self._recalibrate(records, outcomes)

        log.info(
            "learning_loop_complete",
            resolved=report.resolved,
            outcomes=report.total_outcomes,
            weight_changes=len(report.weight_changes),
            calibrations=len(report.adopted_calibrations),
            learned=report.learned,
        )
        return report

    def _recalibrate(
        self, records: Sequence[PredictionRecord], outcomes: Sequence[ResolvedOutcome]
    ) -> list[CalibrationRecord]:
        """Re-fit calibration from resolved outcomes.

        The curves are judged on held-out data by Phase 7's library, which discards any
        that does not improve out-of-sample. On measured data that is usually all of
        them — which is a finding about the models, not a failure of the loop.
        """
        by_id = {r.prediction_id: r for r in records}
        scored: list[_Scored] = []
        for outcome in outcomes:
            record = by_id.get(outcome.prediction_id)
            if record is None or record.confidence <= 0.0:
                continue
            scored.append(
                _Scored(
                    prediction=Prediction(
                        model_id=record.model_id,
                        model_version=record.model_version,
                        asset=record.asset,
                        timeframe=record.timeframe,
                        horizon=record.horizon,
                        as_of=record.as_of,
                        distribution=record.distribution,
                        confidence=record.confidence,
                        move_threshold_pct=record.move_threshold_pct,
                        regime=record.regime,
                        data_quality=record.data_quality,
                        reference_price=record.reference_price,
                    ),
                    actual=outcome.realised_direction,
                )
            )
        return self.calibration.fit(scored) if scored else []
