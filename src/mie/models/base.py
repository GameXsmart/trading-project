"""Predictor interface and the context models see.

Two design commitments live here.

**Models receive a context, not a database.** :class:`PredictionContext` is assembled
once per prediction point and contains only information that existed at ``as_of``.
A model physically cannot query forward, because it has no handle with which to do so.
This is the structural version of the no-look-ahead rule; a convention that models
"should not" peek would be worth nothing during a walk-forward backtest.

**Independence is enforced by what each model is given, not by good intentions.**
Requirement §12 asks for independent models, and eight models sharing one feature
vector is one model with extra steps: they would agree constantly, and the ensemble
would read that agreement as corroboration. Each model here draws on a different
substrate — price structure, statistical dynamics, historical analogues, regime,
news, cross-asset behaviour, derivatives positioning, event sequences — and
:meth:`Predictor.inputs_used` declares which, so overlap is auditable rather than
assumed away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.models.types import (
    Distribution,
    Horizon,
    Prediction,
    PredictionEvidence,
)

log = get_logger(__name__)

__all__ = ["PredictionContext", "Predictor", "move_threshold"]

#: Fraction of recent per-bar volatility that counts as a "real" move. Below this the
#: outcome is FLAT.
#:
#: Volatility-scaled rather than fixed, and set below 1.0 deliberately: at 1.0 the flat
#: class swallows most observations and the directional classes become rare events that
#: no model can be evaluated on. Around 0.6 the three classes are roughly comparable in
#: frequency on crypto hourly data, which is what makes Brier scores interpretable.
_THRESHOLD_VOLATILITY_FRACTION = 0.6


def move_threshold(
    candles: Sequence[Candle], horizon_bars: int, fraction: float = _THRESHOLD_VOLATILITY_FRACTION
) -> float:
    """Volatility-scaled threshold separating a real move from noise.

    Scales with the square root of the horizon, because volatility accumulates that
    way under a random walk — a fixed threshold would make long horizons trivially
    directional and short ones trivially flat.
    """
    import math

    closes = [c.close for c in candles[-100:] if c.close > 0]
    if len(closes) < 10:
        return 0.5
    returns = [
        abs(closes[i] - closes[i - 1]) / closes[i - 1] * 100.0
        for i in range(1, len(closes))
    ]
    typical = median(returns)
    return max(0.05, typical * fraction * math.sqrt(max(1, horizon_bars)))


@dataclass(slots=True)
class PredictionContext:
    """Everything a model may look at, as of one instant.

    Assembled by the runner from data whose close time is at or before ``as_of``.
    Models take this and nothing else.
    """

    asset: str
    timeframe: Timeframe
    as_of: datetime
    horizon: Horizon
    #: Closed bars, oldest first, ending at or before ``as_of``.
    candles: list[Candle] = field(default_factory=list)
    #: Latest feature vector (Phase 2), if computed.
    features: Mapping[str, float] = field(default_factory=dict)
    #: Feature history for models that need dynamics rather than a snapshot.
    feature_history: list[tuple[datetime, Mapping[str, float]]] = field(default_factory=list)
    #: Market state (Phase 3) as a plain mapping, to avoid a hard dependency.
    state: Mapping[str, object] = field(default_factory=dict)
    #: Candles for other assets, for cross-asset models.
    peers: Mapping[str, list[Candle]] = field(default_factory=dict)
    #: News events (Phase 5) already filtered to this asset and to the past.
    news: list[object] = field(default_factory=list)
    #: Funding and open-interest history (Phase 1).
    funding: list[tuple[datetime, float]] = field(default_factory=list)
    open_interest: list[tuple[datetime, float]] = field(default_factory=list)
    #: Trust score for the underlying data (Phase 1), in [0, 1].
    data_quality: float = 1.0
    regime: str = "unknown"

    @property
    def price(self) -> float:
        return self.candles[-1].close if self.candles else 0.0

    @property
    def threshold_pct(self) -> float:
        return move_threshold(self.candles, self.horizon.bars)

    def returns(self, count: int = 200) -> list[float]:
        """Recent close-to-close percentage returns, oldest first."""
        closes = [c.close for c in self.candles[-(count + 1) :] if c.close > 0]
        return [
            (closes[i] - closes[i - 1]) / closes[i - 1] * 100.0
            for i in range(1, len(closes))
        ]

    def has_enough_history(self, bars: int) -> bool:
        return len(self.candles) >= bars


class Predictor(ABC):
    """One independent view of what happens next."""

    #: Stable identifier, stored with every prediction so results stay attributable.
    model_id: str = "abstract"
    version: str = "1"
    #: Bars of history required before this model will say anything.
    warmup_bars: int = 100

    @abstractmethod
    def inputs_used(self) -> frozenset[str]:
        """Which substrates this model draws on.

        Declared so that overlap between "independent" models is auditable. Two models
        sharing every input are not two opinions, and the ensemble must be able to
        notice that rather than counting their agreement twice.
        """

    @abstractmethod
    def predict(self, context: PredictionContext) -> Prediction:
        """Produce a prediction from the context, or an abstention."""

    # ------------------------------------------------------------------ helpers

    def abstain(self, context: PredictionContext, reason: str) -> Prediction:
        """Decline to forecast.

        A uniform distribution with zero confidence. This is a first-class output:
        a model with nothing to say must be able to say nothing, and be scored on
        having said it, rather than emitting a guess that pollutes the ensemble.
        """
        return Prediction(
            model_id=self.model_id,
            model_version=self.version,
            asset=context.asset,
            timeframe=context.timeframe,
            horizon=context.horizon,
            as_of=context.as_of,
            distribution=Distribution.uniform(),
            confidence=0.0,
            move_threshold_pct=context.threshold_pct,
            regime=context.regime,
            data_quality=context.data_quality,
            reference_price=context.price,
            evidence=[PredictionEvidence(label="abstained", detail=reason)],
        )

    def build(
        self,
        context: PredictionContext,
        distribution: Distribution,
        confidence: float,
        evidence: Sequence[PredictionEvidence] = (),
        counter_evidence: Sequence[PredictionEvidence] = (),
        invalidation: Sequence[str] = (),
        expected_move_pct: float | None = None,
        apply_data_quality: bool = True,
    ) -> Prediction:
        """Assemble a prediction, applying the rules every model must obey.

        ``apply_data_quality`` exists for one caller: the Phase 7 ensemble, whose
        confidence is a product of measured factors that already includes the trust
        score. Multiplying it in again would penalise degraded data twice and make the
        published number stop matching its own published decomposition.
        """
        threshold = context.threshold_pct
        volatility = _expected_volatility(context)

        # Data quality multiplies into confidence here rather than in each model, so
        # no model can forget it. This is where Phase 1's trust score reaches the
        # published output.
        adjusted = max(0.0, min(1.0, confidence))
        if apply_data_quality:
            adjusted *= max(0.0, min(1.0, context.data_quality))

        return Prediction(
            model_id=self.model_id,
            model_version=self.version,
            asset=context.asset,
            timeframe=context.timeframe,
            horizon=context.horizon,
            as_of=context.as_of,
            distribution=distribution,
            confidence=round(adjusted, 4),
            expected_move_pct=round(
                expected_move_pct
                if expected_move_pct is not None
                else distribution.directional_edge * threshold,
                4,
            ),
            expected_volatility_pct=round(volatility, 4),
            evidence=list(evidence)[:8],
            counter_evidence=list(counter_evidence)[:5],
            invalidation=list(invalidation)[:5],
            move_threshold_pct=round(threshold, 4),
            regime=context.regime,
            data_quality=context.data_quality,
            reference_price=context.price,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.model_id}>"


def _expected_volatility(context: PredictionContext) -> float:
    """Expected absolute move over the horizon, from recent realised volatility."""
    import math

    returns = context.returns(100)
    if len(returns) < 5:
        return 0.0
    typical = median([abs(r) for r in returns])
    return typical * math.sqrt(max(1, context.horizon.bars))
