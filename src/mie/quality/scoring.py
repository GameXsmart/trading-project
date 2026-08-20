"""Rolling data-quality score.

Detecting bad data is only half of requirement §20. The other half is doing
something with the finding, and the something is this: a score in [0, 1] per
(source, asset, timeframe) that later phases multiply into published confidence.

The scoring rules are deliberately simple and legible, because an opaque quality
score is worse than none — an operator has to be able to look at a 0.4 and say
exactly which events produced it.

    mass    = Σ(severity_weight × recency_decay)      # weighted events in the window
    rate    = 1000 × mass / candles_assessed          # per 1000 bars actually checked
    penalty = 1 - exp(-rate / tolerance)
    score   = clamp(1 - penalty - staleness_penalty)

The **rate**, not the count, is what matters. Forty warnings spread across a year of
hourly history is a healthy feed; forty warnings in an hour is a broken one, and a
count-based score cannot tell those apart — it saturates on the first and so reports
nothing useful about the second.

Errors cost more than warnings, recent events cost more than old ones, and the score
recovers on its own as clean data arrives and old events age out of the window.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mie.config.settings import QualityConfig
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import QualityEvent, QualityEventType

log = get_logger(__name__)

__all__ = ["QualityScore", "QualityScorer"]


@dataclass(slots=True)
class QualityScore:
    """A trust score with the reasoning that produced it."""

    source: str
    asset: str
    timeframe: Timeframe
    score: float
    events_in_window: int
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    last_candle_at: datetime | None = None
    computed_at: datetime = field(default_factory=utcnow)

    @property
    def is_degraded(self) -> bool:
        return self.score < 0.75

    @property
    def is_unusable(self) -> bool:
        """Below this, downstream phases should publish nothing rather than guess."""
        return self.score < 0.35

    def explain(self) -> str:
        if not self.reasons:
            return f"{self.score:.2f} — no quality issues in the window"
        return f"{self.score:.2f} — " + "; ".join(self.reasons)

    def as_details(self) -> dict[str, object]:
        return {
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "reasons": self.reasons,
            "events_in_window": self.events_in_window,
        }


class QualityScorer:
    """Turns a window of quality events into a score."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def score(
        self,
        source: str,
        asset: str,
        timeframe: Timeframe,
        events: Sequence[QualityEvent],
        last_candle_at: datetime | None = None,
        candles_assessed: int | None = None,
        now: datetime | None = None,
    ) -> QualityScore:
        """Score one scope.

        ``candles_assessed`` is how many bars this source delivered during the window.
        It is the denominator that turns an event count into an event rate; without
        it the scorer falls back to a conservative floor, which is deliberately
        pessimistic — an unmeasurable feed should not score as a clean one.
        """
        now = now or utcnow()
        window_s = self.config.score_window_hours * 3600
        in_window = [e for e in events if (now - e.detected_at).total_seconds() <= window_s]

        components: dict[str, float] = {}
        reasons: list[str] = []

        exposure = max(
            self.config.min_exposure_candles,
            candles_assessed if candles_assessed is not None else 0,
        )
        components["exposure"] = float(exposure)

        event_penalty = self._event_penalty(in_window, now, window_s, exposure, reasons)
        components["events"] = event_penalty

        staleness_penalty = self._staleness_penalty(timeframe, last_candle_at, now, reasons)
        components["staleness"] = staleness_penalty

        raw = 1.0 - event_penalty - staleness_penalty
        final = max(self.config.min_score, min(1.0, raw))
        components["raw"] = raw

        return QualityScore(
            source=source,
            asset=asset.upper(),
            timeframe=timeframe,
            score=round(final, 4),
            events_in_window=len(in_window),
            reasons=reasons,
            components=components,
            last_candle_at=last_candle_at,
            computed_at=now,
        )

    def _event_penalty(
        self,
        events: Iterable[QualityEvent],
        now: datetime,
        window_s: float,
        exposure: int,
        reasons: list[str],
    ) -> float:
        """Severity-weighted, recency-decayed, exposure-normalised penalty.

        Decay is exponential with a half-life of a quarter of the window: an incident
        eight hours ago should not weigh the same as one happening right now, but it
        should not vanish either.

        The result is divided by the number of bars actually assessed, so the score
        measures a defect *rate*. The saturating ``1 - exp(-x)`` curve then keeps the
        score responsive across the whole range instead of pinning to the floor the
        moment a handful of warnings appear.
        """
        events = list(events)
        half_life = window_s / 4.0
        mass = 0.0
        by_type: dict[str, float] = {}

        for event in events:
            age = max(0.0, (now - event.detected_at).total_seconds())
            decay = math.exp(-math.log(2) * age / half_life)
            weight = event.severity.weight * decay
            mass += weight
            by_type[str(event.event_type)] = by_type.get(str(event.event_type), 0.0) + weight

        if mass <= 0:
            return 0.0

        rate = 1000.0 * mass / exposure
        penalty = 1.0 - math.exp(-rate / max(0.1, self.config.event_rate_tolerance))

        reasons.append(f"{rate:.1f} weighted events/1k bars")
        for event_type, _ in sorted(by_type.items(), key=lambda kv: -kv[1])[:3]:
            count = sum(1 for e in events if str(e.event_type) == event_type)
            reasons.append(f"{count}× {event_type}")

        return min(0.95, penalty)

    def _staleness_penalty(
        self,
        timeframe: Timeframe,
        last_candle_at: datetime | None,
        now: datetime,
        reasons: list[str],
    ) -> float:
        """Penalise a feed that has stopped producing.

        This is separate from the validator's staleness *event* on purpose: a feed
        that silently stops delivering generates no events at all, so an
        event-only score would rate a dead feed as perfect.
        """
        if last_candle_at is None:
            reasons.append("no data observed")
            return 0.5

        age = (now - last_candle_at).total_seconds()
        allowance = timeframe.seconds * self.config.staleness_multiplier
        if age <= allowance:
            return 0.0

        overdue = age / allowance
        penalty = min(0.7, 0.15 * math.log2(overdue + 1) * 2)
        reasons.append(f"feed {age / 60:.0f}m behind (allowance {allowance / 60:.0f}m)")
        return penalty

    @staticmethod
    def summarise(scores: Sequence[QualityScore]) -> dict[str, object]:
        """Aggregate view for the CLI and, later, the dashboard's data-health panel."""
        if not scores:
            return {"count": 0, "mean": None, "degraded": [], "unusable": []}
        values = [s.score for s in scores]
        return {
            "count": len(scores),
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "degraded": [
                f"{s.source}/{s.asset}/{s.timeframe}" for s in scores if s.is_degraded
            ],
            "unusable": [
                f"{s.source}/{s.asset}/{s.timeframe}" for s in scores if s.is_unusable
            ],
        }


#: Event types that indicate the *pipeline* is broken rather than the market being
#: unusual. Phase 7's confidence layer treats these more harshly than market-shaped
#: anomalies, which are legitimately part of what the models should be seeing.
PIPELINE_FAULTS: frozenset[QualityEventType] = frozenset(
    {
        QualityEventType.SHAPE_INVALID,
        QualityEventType.GRID_MISALIGNED,
        QualityEventType.STALE_FEED,
        QualityEventType.PROVIDER_ERROR,
        QualityEventType.SOURCE_DISCREPANCY,
    }
)
