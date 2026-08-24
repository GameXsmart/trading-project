"""The rules, and an honest account of which of them can actually fire.

Nine phases of measurement produced a short list of things this system genuinely knows
about, and a much longer list of things it does not. The rules here follow that split
rather than the wish list.

**Rules backed by a measured effect.** Volume anomalies and range compression survived
Phase 4's evidence gate — they precede larger-than-usual movement, on real data, after
correction for multiple comparisons. They say nothing about direction, and neither do
the alerts built on them.

**Rules that report facts rather than predictions.** A regime change, a correlation
breakdown, a data feed degrading, a panel disagreeing with itself: these are
observations about the present or about the system's own state. They need no predictive
skill to be worth sending, and they are the bulk of what this system can honestly
interrupt someone for.

**Rules that cannot currently fire, kept anyway.** ``STRONG_PREDICTION`` and
``SUPER_PREDICTION`` require the ensemble to publish, and it never does. They are
implemented, tested against synthetic input, and silent on real data — which is
the correct behaviour, not a gap. Deleting them would hide the finding; leaving them
to fire on weak evidence would fabricate one.

One proxy is flagged explicitly. There is no liquidation feed in this system, so
``LIQUIDATION_SPIKE`` is inferred from a sharp fall in open interest alongside a large
price move. That is a reasonable proxy and it is not the same measurement, so the alert
says so in its own text rather than implying a data source that does not exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Protocol

from mie.alerts.types import Alert, AlertKind, Severity
from mie.core.logging import get_logger
from mie.core.timeframes import utcnow
from mie.core.types import Candle
from mie.ensemble.gate import GateDecision
from mie.ensemble.meta import EnsemblePrediction

log = get_logger(__name__)

__all__ = [
    "DEFAULT_RULES",
    "AlertContext",
    "CorrelationBreakdownRule",
    "DataQualityRule",
    "LiquidationSpikeRule",
    "MajorNewsRule",
    "ModelDisagreementRule",
    "PredictionInvalidatedRule",
    "RegimeChangeRule",
    "Rule",
    "StrongPredictionRule",
    "SuperPredictionRule",
    "VolatilityRule",
    "VolumeAnomalyRule",
]

#: Robust z-score above which a volume bar counts as anomalous. Set from Phase 4's
#: measured detector rather than picked: below this the "anomaly" fires on ordinary
#: bars, and a detector that cries wolf on ordinary volatility is worse than none.
_VOLUME_Z = 3.5

#: Ratio of recent to baseline true range that counts as expansion or compression.
_EXPANSION_RATIO = 1.8
_COMPRESSION_RATIO = 0.55

#: Correlation drop, in absolute terms, that counts as a breakdown. Crypto majors sit
#: around 0.7-0.85 against each other; a fall of this size is a structural change, not
#: a wobble.
_CORRELATION_DROP = 0.35

#: Trust score below which the data itself is the story.
_QUALITY_FLOOR = 0.7


@dataclass(slots=True)
class AlertContext:
    """Everything the rules may look at.

    Assembled once per evaluation. Rules receive this and nothing else, for the same
    reason models receive a prediction context: a rule that can reach the database can
    reach anything, including the future.
    """

    asset: str
    timeframe: str = ""
    at: datetime = field(default_factory=utcnow)
    candles: Sequence[Candle] = ()
    #: Current and previous regime labels, for change detection.
    regime: str = ""
    previous_regime: str = ""
    data_quality: float = 1.0
    #: Correlations against peers now and over a longer baseline.
    correlation_now: Mapping[str, float] = field(default_factory=dict)
    correlation_baseline: Mapping[str, float] = field(default_factory=dict)
    open_interest: Sequence[tuple[datetime, float]] = ()
    news: Sequence[object] = ()
    #: The Phase 7 ensemble result and gate decision, when one was computed.
    ensemble: EnsemblePrediction | None = None
    gate: GateDecision | None = None
    #: Previously published calls still within their horizon, for invalidation checks.
    open_calls: Sequence[object] = ()


class Rule(Protocol):
    """One condition worth interrupting a human for."""

    name: str

    def evaluate(self, context: AlertContext) -> list[Alert]:
        """Return zero or more alerts. Zero is the normal case."""
        ...


def _robust_z(values: Sequence[float]) -> float:
    """Median-absolute-deviation z-score of the last value.

    MAD rather than standard deviation, throughout this repository: the standard
    deviation of a series containing a spike is inflated *by* the spike, so the very
    event being detected raises the threshold for detecting it.
    """
    if len(values) < 30:
        return 0.0
    history = values[:-1]
    centre = median(history)
    spread = median([abs(v - centre) for v in history])
    if spread <= 0:
        return 0.0
    return (values[-1] - centre) / (spread * 1.4826)


def _true_ranges(candles: Sequence[Candle]) -> list[float]:
    ranges = []
    for index in range(1, len(candles)):
        current, previous = candles[index], candles[index - 1]
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return ranges


@dataclass(slots=True)
class VolumeAnomalyRule:
    """Volume far above its own recent norm.

    One of the few detectors that cleared Phase 4's evidence gate: on BTC, ETH and SOL
    it preceded larger-than-usual movement with an edge of +8% to +21% over the
    unconditional rate. It says nothing about direction, and the alert does not either.
    """

    name: str = "volume_anomaly"
    threshold: float = _VOLUME_Z
    window: int = 200

    def evaluate(self, context: AlertContext) -> list[Alert]:
        volumes = [c.volume for c in context.candles[-self.window :] if c.volume > 0]
        if len(volumes) < 50:
            return []
        z = _robust_z(volumes)
        if z < self.threshold:
            return []
        return [
            Alert(
                kind=AlertKind.VOLUME_ANOMALY,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"volume {z:.1f}x above its normal spread",
                detail=(
                    "Measured to precede larger-than-usual movement. It gives no "
                    "information about direction, and none is implied."
                ),
                severity=Severity.IMPORTANT if z >= self.threshold * 1.6 else Severity.NOTABLE,
                context={"robust_z": round(z, 2)},
            )
        ]


@dataclass(slots=True)
class VolatilityRule:
    """Range expanding or compressing sharply against its own baseline."""

    name: str = "volatility"
    expansion: float = _EXPANSION_RATIO
    compression: float = _COMPRESSION_RATIO
    recent: int = 14
    baseline: int = 100

    def evaluate(self, context: AlertContext) -> list[Alert]:
        ranges = _true_ranges(context.candles[-(self.baseline + 1) :])
        if len(ranges) < self.baseline * 0.6:
            return []
        recent = median(ranges[-self.recent :])
        base = median(ranges)
        if base <= 0:
            return []
        ratio = recent / base

        if ratio >= self.expansion:
            return [
                Alert(
                    kind=AlertKind.VOLATILITY_EXPANSION,
                    asset=context.asset,
                    timeframe=context.timeframe,
                    at=context.at,
                    title=f"range expanded to {ratio:.1f}x its baseline",
                    detail="Larger moves than usual, in both directions.",
                    context={"ratio": round(ratio, 3)},
                )
            ]
        if ratio <= self.compression:
            return [
                Alert(
                    kind=AlertKind.VOLATILITY_COMPRESSION,
                    asset=context.asset,
                    timeframe=context.timeframe,
                    at=context.at,
                    title=f"range compressed to {ratio:.2f}x its baseline",
                    detail=(
                        "Compression measurably precedes expansion. It does not "
                        "indicate which way the expansion will resolve."
                    ),
                    context={"ratio": round(ratio, 3)},
                )
            ]
        return []


@dataclass(slots=True)
class RegimeChangeRule:
    """The market moved from one regime to another.

    An observation, not a forecast — which is why it can be sent without any predictive
    skill behind it. It matters because every calibration record and every measured
    edge in this system is conditioned on regime, so a change invalidates the context
    the rest of the output was computed in.
    """

    name: str = "regime_change"

    def evaluate(self, context: AlertContext) -> list[Alert]:
        if not context.regime or not context.previous_regime:
            return []
        if context.regime == context.previous_regime:
            return []
        return [
            Alert(
                kind=AlertKind.REGIME_CHANGE,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"regime changed: {context.previous_regime} to {context.regime}",
                detail=(
                    "Calibration and measured edges are conditioned on regime, so "
                    "prior results may not carry over."
                ),
                context={"from": context.previous_regime, "to": context.regime},
            )
        ]


@dataclass(slots=True)
class CorrelationBreakdownRule:
    """A pair that normally moves together has stopped doing so."""

    name: str = "correlation_breakdown"
    drop: float = _CORRELATION_DROP

    def evaluate(self, context: AlertContext) -> list[Alert]:
        alerts = []
        for peer, baseline in context.correlation_baseline.items():
            now = context.correlation_now.get(peer)
            if now is None or baseline < 0.5:
                continue
            if baseline - now < self.drop:
                continue
            alerts.append(
                Alert(
                    kind=AlertKind.CORRELATION_BREAKDOWN,
                    asset=context.asset,
                    timeframe=context.timeframe,
                    at=context.at,
                    title=f"correlation with {peer} fell {baseline:.2f} to {now:.2f}",
                    detail=(
                        "Cross-asset structure has changed. Anything that assumed "
                        "these move together is now on weaker ground."
                    ),
                    context={"peer": peer, "baseline": baseline, "now": now},
                )
            )
        return alerts


@dataclass(slots=True)
class LiquidationSpikeRule:
    """Open interest collapsing alongside a large move.

    A *proxy*. This system has no liquidation feed, and the alert says so in its own
    text — implying a data source that does not exist would be a small lie that a
    reader could not detect.
    """

    name: str = "liquidation_spike"
    oi_drop_pct: float = 4.0
    move_pct: float = 2.0

    def evaluate(self, context: AlertContext) -> list[Alert]:
        points = [value for _, value in context.open_interest[-24:] if value > 0]
        if len(points) < 6 or len(context.candles) < 6:
            return []
        peak = max(points)
        if peak <= 0:
            return []
        drop = (peak - points[-1]) / peak * 100.0
        closes = [c.close for c in context.candles[-6:] if c.close > 0]
        if len(closes) < 2 or closes[0] <= 0:
            return []
        move = abs(closes[-1] - closes[0]) / closes[0] * 100.0

        if drop < self.oi_drop_pct or move < self.move_pct:
            return []
        return [
            Alert(
                kind=AlertKind.LIQUIDATION_SPIKE,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"open interest fell {drop:.1f}% during a {move:.1f}% move",
                detail=(
                    "Consistent with forced closures. Inferred from open interest, "
                    "not from a liquidation feed — this system has none."
                ),
                context={"oi_drop_pct": round(drop, 2), "move_pct": round(move, 2)},
            )
        ]


@dataclass(slots=True)
class DataQualityRule:
    """The feed itself is the story.

    Critical, and deliberately ranked above any market event. A large move is news; a
    degraded feed means every other number on the screen is suspect, including the ones
    that would otherwise have generated alerts.
    """

    name: str = "data_quality"
    floor: float = _QUALITY_FLOOR

    def evaluate(self, context: AlertContext) -> list[Alert]:
        if context.data_quality >= self.floor:
            return []
        return [
            Alert(
                kind=AlertKind.DATA_QUALITY,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"data quality degraded to {context.data_quality:.2f}",
                detail=(
                    "Published confidence is reduced accordingly. Treat other output "
                    "for this series as provisional until it recovers."
                ),
                context={"score": round(context.data_quality, 3)},
            )
        ]


@dataclass(slots=True)
class ModelDisagreementRule:
    """The panel is split.

    Low severity on purpose. Disagreement is informative but it is not urgent, and it
    is also the normal state of this panel — most models abstain, and those that do not
    frequently point opposite ways.
    """

    name: str = "model_disagreement"

    def evaluate(self, context: AlertContext) -> list[Alert]:
        agreement = context.ensemble.agreement if context.ensemble else None
        if agreement is None or not agreement.is_split:
            return []
        if agreement.participants < 4:
            # Two models disagreeing is not a split panel, it is a small sample.
            return []
        return [
            Alert(
                kind=AlertKind.MODEL_DISAGREEMENT,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"models disagree ({agreement.consensus_share:.0%} consensus)",
                detail=(
                    f"{agreement.summary()}. Disagreement suppresses publication "
                    f"rather than averaging into a confident-looking middle."
                ),
                context={"consensus": agreement.consensus_share},
            )
        ]


@dataclass(slots=True)
class MajorNewsRule:
    """A story with unusually broad coverage."""

    name: str = "major_news"
    importance: float = 0.75

    def evaluate(self, context: AlertContext) -> list[Alert]:
        alerts = []
        for item in context.news:
            score = float(getattr(item, "importance", 0.0))
            if score < self.importance:
                continue
            alerts.append(
                Alert(
                    kind=AlertKind.MAJOR_NEWS,
                    asset=context.asset,
                    timeframe=context.timeframe,
                    at=context.at,
                    title=str(getattr(item, "title", ""))[:180],
                    detail=(
                        f"{getattr(item, 'category', 'other')}, "
                        f"{getattr(item, 'coverage', 1)} sources. Whether news moves "
                        f"price is not yet measurable in this system."
                    ),
                    context={"importance": round(score, 3)},
                )
            )
        return alerts


@dataclass(slots=True)
class PredictionInvalidatedRule:
    """A published call's own invalidation condition has triggered.

    Critical. A forecast that has been falsified and not withdrawn is worse than no
    forecast, because a reader who saw it and not this is acting on something the
    system already knows to be wrong.
    """

    name: str = "prediction_invalidated"

    def evaluate(self, context: AlertContext) -> list[Alert]:
        price = context.candles[-1].close if context.candles else 0.0
        alerts = []
        for call in context.open_calls:
            level = getattr(call, "invalidation_price", None)
            direction = str(getattr(call, "direction", ""))
            if not level or not price or not direction:
                continue
            breached = (direction == "up" and price <= level) or (
                direction == "down" and price >= level
            )
            if not breached:
                continue
            alerts.append(
                Alert(
                    kind=AlertKind.PREDICTION_INVALIDATED,
                    asset=context.asset,
                    timeframe=context.timeframe,
                    at=context.at,
                    title=f"a published {direction} call has been invalidated",
                    detail=(
                        f"Price {price:.2f} crossed its stated invalidation level "
                        f"{level:.2f}. The call no longer stands."
                    ),
                    context={"price": price, "level": level},
                )
            )
        return alerts


@dataclass(slots=True)
class StrongPredictionRule:
    """The ensemble published a directional call.

    Cannot fire on current data: no model has demonstrated skill, so the ensemble
    abstains everywhere. Implemented and tested against synthetic input so that the
    silence is a measured result rather than an unwritten branch.
    """

    name: str = "strong_prediction"

    def evaluate(self, context: AlertContext) -> list[Alert]:
        result = context.ensemble
        if result is None or not result.published:
            return []
        prediction = result.prediction
        return [
            Alert(
                kind=AlertKind.STRONG_PREDICTION,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=(
                    f"{prediction.distribution.most_likely.value} "
                    f"{prediction.distribution.probability(prediction.distribution.most_likely):.0%} "
                    f"over {prediction.horizon.label()}"
                ),
                detail=str(prediction.distribution),
                confidence=prediction.confidence,
                invalidation=list(prediction.invalidation),
                context={"edge": prediction.distribution.directional_edge},
            )
        ]


@dataclass(slots=True)
class SuperPredictionRule:
    """Every super-prediction condition passed.

    Also cannot fire on current data: six of the gate's nine conditions fail at every
    evaluated point across three assets.
    """

    name: str = "super_prediction"

    def evaluate(self, context: AlertContext) -> list[Alert]:
        decision = context.gate
        result = context.ensemble
        if decision is None or not decision.passed:
            return []
        if result is None or not result.published:
            return []
        prediction = result.prediction
        return [
            Alert(
                kind=AlertKind.SUPER_PREDICTION,
                asset=context.asset,
                timeframe=context.timeframe,
                at=context.at,
                title=f"super prediction: {prediction.distribution.most_likely.value}",
                detail=(
                    "All nine gate conditions met, including independent agreement "
                    "across at least six model families and calibration in this regime."
                ),
                confidence=prediction.confidence,
                invalidation=list(prediction.invalidation),
            )
        ]


#: The rules evaluated by default, most consequential first.
DEFAULT_RULES: tuple[Rule, ...] = (
    DataQualityRule(),
    PredictionInvalidatedRule(),
    SuperPredictionRule(),
    StrongPredictionRule(),
    RegimeChangeRule(),
    CorrelationBreakdownRule(),
    LiquidationSpikeRule(),
    VolumeAnomalyRule(),
    VolatilityRule(),
    MajorNewsRule(),
    ModelDisagreementRule(),
)
