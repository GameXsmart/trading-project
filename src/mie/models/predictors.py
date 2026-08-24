"""Models A–H: eight independent predictors.

Requirement §12 asks for several independent analytical models. The independence is
the hard part and the reason these are thin: each one is a deliberate, narrow mapping
from a *different* substrate built in an earlier phase, rather than eight variations
on one feature vector.

| Model | Substrate | Phase |
|---|---|---|
| A `technical` | indicator structure | 2 |
| B `timeseries` | statistical dynamics of returns | 1 |
| C `similarity` | historical analogues | 4 |
| D `regime` | multi-timeframe state | 3 |
| E `sentiment` | news | 5 |
| F `crossasset` | peer behaviour and lead/lag | 1 |
| G `orderflow` | funding and open interest | 1 |
| H `sequence` | recent event chains | 4 |

Every one of them abstains rather than guessing when its substrate is missing, and
several are expected to abstain most of the time. That is intended: Phase 4 measured
no directional edge in classical patterns, and a model built on the same substrate has
no business being confident. The evaluation in `evaluation.py` decides which of these
survive; nothing here assumes any of them will.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median

from mie.core.types import Candle
from mie.models.base import PredictionContext, Predictor
from mie.models.types import Distribution, Prediction, PredictionEvidence

__all__ = [
    "ALL_MODELS",
    "CrossAssetModel",
    "OrderFlowModel",
    "RegimeModel",
    "SentimentModel",
    "SequenceModel",
    "SimilarityModel",
    "TechnicalModel",
    "TimeSeriesModel",
]


class TechnicalModel(Predictor):
    """Model A — indicator structure.

    Reads the Phase 2 feature vector: trend alignment, momentum, and position within
    the Bollinger envelope. Kept deliberately simple; Phase 4 already showed that
    elaborate technical rules do not survive measurement, so elaborating here would
    add complexity without evidence.
    """

    model_id = "technical"
    warmup_bars = 200

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"features"})

    def predict(self, context: PredictionContext) -> Prediction:
        features = context.features
        if not features or "close" not in features:
            return self.abstain(context, "no feature vector available")

        close = features["close"]
        evidence: list[PredictionEvidence] = []
        counter: list[PredictionEvidence] = []
        signals: list[float] = []

        for key, weight in (("ema_21", 0.8), ("sma_50", 1.0), ("sma_200", 1.2)):
            level = features.get(key)
            if not level or level <= 0:
                continue
            distance = (close - level) / level * 100.0
            signal = math.tanh(distance / 4.0)
            signals.append(signal * weight)
            (evidence if signal > 0 else counter).append(
                PredictionEvidence(
                    label=f"price vs {key}", detail=f"{distance:+.2f}%", contribution=signal
                )
            )

        rsi = features.get("rsi_14")
        if rsi is not None:
            signal = math.tanh((rsi - 50.0) / 20.0) * 0.8
            signals.append(signal)
            (evidence if signal > 0 else counter).append(
                PredictionEvidence(label="RSI", detail=f"{rsi:.0f}", contribution=signal)
            )

        histogram = features.get("macd_12_26_9.histogram")
        if histogram is not None and close > 0:
            signal = math.tanh(histogram / close * 100.0 / 0.4) * 0.8
            signals.append(signal)
            (evidence if signal > 0 else counter).append(
                PredictionEvidence(
                    label="MACD histogram", detail=f"{histogram:+.4g}", contribution=signal
                )
            )

        if not signals:
            return self.abstain(context, "no usable indicators")

        edge = max(-1.0, min(1.0, sum(signals) / max(1.0, len(signals) * 1.0)))
        adx = features.get("adx_14.adx", 20.0)
        # Directional indicators mean much less without a trend to be directional
        # about, so trend strength gates confidence rather than the signal itself.
        confidence = 0.25 + 0.35 * min(1.0, max(0.0, (adx - 15.0) / 30.0))

        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.5),
            confidence=confidence,
            evidence=evidence,
            counter_evidence=counter,
            invalidation=[
                "a close back through the 50-bar mean would remove the trend premise",
            ],
        )


class TimeSeriesModel(Predictor):
    """Model B — statistical dynamics of returns.

    Estimates drift and autocorrelation from recent returns and projects them forward.
    This is the closest thing here to a classical forecaster, and its honest prior is
    that returns are close to unforecastable: it only takes a directional view when
    lag-1 autocorrelation is materially non-zero.
    """

    model_id = "timeseries"
    warmup_bars = 200

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"price"})

    def predict(self, context: PredictionContext) -> Prediction:
        returns = context.returns(200)
        if len(returns) < 100:
            return self.abstain(context, "insufficient return history")

        rho = _autocorrelation(returns, lag=1)
        drift = sum(returns) / len(returns)
        threshold = context.threshold_pct

        # Project the last move forward by the estimated persistence, plus drift.
        projected = returns[-1] * rho * context.horizon.bars + drift * context.horizon.bars
        edge = math.tanh(projected / max(threshold, 1e-9))

        evidence = [
            PredictionEvidence(
                label="lag-1 autocorrelation",
                detail=f"{rho:+.3f}",
                contribution=math.copysign(min(1.0, abs(rho) * 10), projected or 1.0),
            ),
            PredictionEvidence(
                label="mean drift", detail=f"{drift:+.4f}%/bar", contribution=math.tanh(drift)
            ),
        ]

        # Autocorrelation this weak is indistinguishable from zero on this sample size,
        # and projecting it would be reading noise.
        if abs(rho) < 0.03:
            return self.build(
                context,
                distribution=Distribution.uniform().blend(
                    Distribution.from_edge(edge * 0.2), 0.5
                ),
                confidence=0.15,
                evidence=evidence,
                counter_evidence=[
                    PredictionEvidence(
                        label="returns are near-unforecastable",
                        detail=f"|rho| = {abs(rho):.3f} is within noise",
                    )
                ],
                invalidation=["a shift in autocorrelation regime"],
            )

        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.45),
            confidence=0.20 + min(0.35, abs(rho) * 3.0),
            evidence=evidence,
            invalidation=["autocorrelation reverting to zero"],
        )


class SimilarityModel(Predictor):
    """Model C — historical analogues.

    Delegates to the Phase 4 similarity engine: find past moments that resembled this
    one and report what followed. Abstains whenever the engine finds too few genuine
    analogues, which on real data is often — the engine is built to say "this has not
    happened before" and this model inherits that.
    """

    model_id = "similarity"
    warmup_bars = 400

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"features", "analogues"})

    def predict(self, context: PredictionContext) -> Prediction:
        from mie.patterns.similarity import SimilarityEngine

        if len(context.feature_history) < self.warmup_bars:
            return self.abstain(context, "insufficient feature history for analogue search")

        closes = [f.get("close", 0.0) for _, f in context.feature_history]
        result = SimilarityEngine().search(
            context.feature_history,
            closes,
            query_index=len(context.feature_history) - 1,
            horizon=context.horizon.bars,
            asset=context.asset,
            timeframe=context.timeframe,
        )
        if not result.has_evidence or result.estimate is None:
            return self.abstain(
                context,
                f"only {len(result.analogues)} comparable historical situations",
            )

        # The analogue up-rate against the same period's baseline is the edge.
        edge = math.tanh((result.estimate.rate - result.estimate.baseline) * 4.0)
        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.6),
            confidence=0.20 + min(0.35, len(result.analogues) / 400.0),
            evidence=[
                PredictionEvidence(
                    label="historical analogues",
                    detail=(
                        f"{len(result.analogues)} similar moments rose "
                        f"{result.estimate.rate:.0%} vs {result.estimate.baseline:.0%} baseline"
                    ),
                    contribution=edge,
                ),
                PredictionEvidence(
                    label="median analogue outcome",
                    detail=f"{result.median_return_pct:+.2f}%",
                    contribution=math.tanh(result.median_return_pct),
                ),
            ],
            invalidation=["the current state drifting away from its historical analogues"],
            expected_move_pct=result.median_return_pct,
        )


class RegimeModel(Predictor):
    """Model D — multi-timeframe state.

    Consumes the Phase 3 market state: hierarchical bias, alignment and agreement. Its
    distinctive contribution is that it *knows about conflict* — a pullback inside an
    uptrend produces a different call from an aligned trend, where a flat indicator
    reading would look identical.
    """

    model_id = "regime"
    warmup_bars = 200

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"state"})

    def predict(self, context: PredictionContext) -> Prediction:
        state = context.state
        if not state or "bias_score" not in state:
            return self.abstain(context, "no market state available")

        bias = float(state.get("bias_score", 0.0))  # type: ignore[arg-type]
        agreement = float(state.get("agreement", 0.0))  # type: ignore[arg-type]
        alignment = str(state.get("alignment", "unknown"))
        state_confidence = float(state.get("confidence", 0.0))  # type: ignore[arg-type]

        edge = bias * (0.4 + 0.6 * agreement)
        evidence = [
            PredictionEvidence(
                label=f"hierarchy is {alignment}",
                detail=f"agreement {agreement:.0%}",
                contribution=edge,
            )
        ]
        counter: list[PredictionEvidence] = []

        # A pullback is bullish structure with bearish tactics; the model leans with
        # the structure but is explicit that the near term argues the other way.
        if alignment == "pullback_in_uptrend":
            counter.append(
                PredictionEvidence(
                    label="lower timeframes correcting", detail="short-term drag", contribution=-0.3
                )
            )
        elif alignment == "rally_in_downtrend":
            counter.append(
                PredictionEvidence(
                    label="lower timeframes bouncing", detail="short-term lift", contribution=0.3
                )
            )
        elif alignment in ("conflicted", "possible_reversal"):
            return self.build(
                context,
                distribution=Distribution.uniform().blend(Distribution.from_edge(edge * 0.2), 0.4),
                confidence=0.15,
                evidence=evidence,
                counter_evidence=[
                    PredictionEvidence(
                        label="timeframes disagree", detail=alignment, contribution=0.0
                    )
                ],
                invalidation=["the hierarchy resolving one way or the other"],
            )

        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.55),
            confidence=0.20 + 0.4 * state_confidence,
            evidence=evidence,
            counter_evidence=counter,
            invalidation=["a higher-timeframe trend change would invalidate the bias"],
        )


class SentimentModel(Predictor):
    """Model E — news.

    Uses importance-weighted sentiment over recent stories. Its confidence is capped
    hard: Phase 5 could not yet validate that news moves prices, and a model built on
    an unvalidated signal must not speak loudly. If Phase 9 finds it has skill, the cap
    is what should be revisited — not the other way round.
    """

    model_id = "sentiment"
    warmup_bars = 50
    #: Ceiling until news impact is validated against realised prices.
    max_confidence = 0.25

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"news"})

    def predict(self, context: PredictionContext) -> Prediction:
        if not context.news:
            return self.abstain(context, "no news in the lookback window")

        weighted = 0.0
        total_weight = 0.0
        strongest: PredictionEvidence | None = None
        for event in context.news:
            importance = float(getattr(event, "importance", 0.0))
            confidence = float(getattr(event, "confidence", 0.0))
            score = float(getattr(event, "sentiment_score", 0.0))
            weight = importance * confidence
            if weight <= 0:
                continue
            weighted += score * weight
            total_weight += weight
            if strongest is None or abs(score) * weight > abs(strongest.contribution):
                strongest = PredictionEvidence(
                    label=str(getattr(event, "title", "story"))[:70],
                    detail=f"{getattr(event, 'category', '')}, importance {importance:.2f}",
                    contribution=score,
                )

        if total_weight <= 0:
            return self.abstain(context, "no news carrying usable weight")

        mean_sentiment = weighted / total_weight
        edge = math.tanh(mean_sentiment * 1.5)

        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.35),
            confidence=min(self.max_confidence, 0.10 + abs(mean_sentiment) * 0.3),
            evidence=[
                PredictionEvidence(
                    label="aggregate news sentiment",
                    detail=f"{mean_sentiment:+.3f} across {len(context.news)} stories",
                    contribution=edge,
                ),
                *([strongest] if strongest else []),
            ],
            counter_evidence=[
                PredictionEvidence(
                    label="news impact is not yet validated",
                    detail="confidence capped pending measurement",
                )
            ],
            invalidation=["a major contradicting story"],
        )


class CrossAssetModel(Predictor):
    """Model F — peer behaviour.

    Crypto assets move together, and the correlation is high enough that a peer's
    recent move carries information about an asset that has not yet moved. The model
    looks for exactly that: a divergence between this asset and the peer group it
    normally tracks.
    """

    model_id = "crossasset"
    warmup_bars = 100

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"peers"})

    def predict(self, context: PredictionContext) -> Prediction:
        if not context.peers:
            return self.abstain(context, "no peer data")

        own = _recent_return(context.candles, context.horizon.bars)
        if own is None:
            return self.abstain(context, "insufficient own history")

        peer_moves: list[tuple[str, float]] = []
        for name, candles in context.peers.items():
            move = _recent_return(candles, context.horizon.bars)
            if move is not None:
                peer_moves.append((name, move))
        if not peer_moves:
            return self.abstain(context, "no usable peer history")

        peer_median = median([m for _, m in peer_moves])
        divergence = peer_median - own
        threshold = context.threshold_pct

        # The bet: a laggard catches up to its peer group. This is a real, weak effect
        # in correlated markets and is stated as such rather than dressed up.
        edge = math.tanh(divergence / max(threshold, 1e-9))
        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.4),
            confidence=0.18 + min(0.25, abs(divergence) / max(threshold * 3, 1e-9)),
            evidence=[
                PredictionEvidence(
                    label="peer group divergence",
                    detail=(
                        f"peers {peer_median:+.2f}% vs own {own:+.2f}% "
                        f"({len(peer_moves)} peers)"
                    ),
                    contribution=edge,
                )
            ],
            invalidation=["the divergence closing, or peers reversing"],
        )


class OrderFlowModel(Predictor):
    """Model G — derivatives positioning.

    Funding rate and open interest describe leveraged positioning rather than price.
    The classic reading is contrarian at extremes: heavily positive funding means longs
    are paying to stay in, and crowded positioning is fragile.
    """

    model_id = "orderflow"
    warmup_bars = 50

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"funding", "open_interest"})

    def predict(self, context: PredictionContext) -> Prediction:
        if len(context.funding) < 10:
            return self.abstain(context, "insufficient funding history")

        rates = [rate for _, rate in context.funding[-60:]]
        current = rates[-1]
        centre = median(rates)
        spread = median([abs(r - centre) for r in rates]) * 1.4826
        if spread <= 0:
            return self.abstain(context, "funding shows no variation to measure against")

        z = (current - centre) / spread
        # Contrarian: extreme funding is a positioning warning, not a trend signal.
        edge = -math.tanh(z / 3.0)

        evidence = [
            PredictionEvidence(
                label="funding rate",
                detail=f"{current:.5f} ({z:+.1f} robust sigma)",
                contribution=edge,
            )
        ]
        if len(context.open_interest) >= 10:
            values = [v for _, v in context.open_interest[-30:]]
            if values[0] > 0:
                change = (values[-1] - values[0]) / values[0] * 100.0
                evidence.append(
                    PredictionEvidence(
                        label="open interest trend",
                        detail=f"{change:+.1f}% over the window",
                        contribution=math.tanh(change / 20.0) * 0.2,
                    )
                )

        # Only meaningful at extremes; mid-range funding says nothing.
        if abs(z) < 1.5:
            return self.abstain(context, f"funding is unremarkable ({z:+.1f} sigma)")

        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.4),
            confidence=0.18 + min(0.25, (abs(z) - 1.5) / 6.0),
            evidence=evidence,
            invalidation=["funding normalising, or price trending through the positioning"],
        )


class SequenceModel(Predictor):
    """Model H — recent event chains.

    Looks at which patterns fired recently and what the Phase 4 registry measured about
    them. Because that measurement admitted almost nothing, this model abstains most of
    the time by construction — which is the correct behaviour for a substrate that was
    tested and largely failed, and is far better than inventing a signal to justify the
    model's existence.
    """

    model_id = "sequence"
    warmup_bars = 150

    def __init__(self, registry: object | None = None) -> None:
        self.registry = registry

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"patterns"})

    def predict(self, context: PredictionContext) -> Prediction:
        from mie.patterns.detectors import detect_all

        if not context.has_enough_history(self.warmup_bars):
            return self.abstain(context, "insufficient history for pattern detection")

        detections = detect_all(context.candles, context.asset, context.timeframe)
        if not detections:
            return self.abstain(context, "no patterns detected on the latest bar")

        if self.registry is None:
            return self.abstain(
                context, "no measured pattern evidence available; unproven patterns are withheld"
            )

        edges: list[tuple[str, float]] = []
        for detection in detections:
            edge = self.registry.expected_edge(detection, context.horizon.bars)  # type: ignore[attr-defined]
            if edge:
                edges.append((str(detection.kind), edge))

        if not edges:
            return self.abstain(
                context,
                f"{len(detections)} pattern(s) detected, none with a measured edge at this horizon",
            )

        total = sum(edge for _, edge in edges)
        return self.build(
            context,
            distribution=Distribution.from_edge(math.tanh(total * 3.0) * 0.5),
            confidence=0.20 + min(0.3, len(edges) * 0.1),
            evidence=[
                PredictionEvidence(
                    label=f"{name} (measured edge)",
                    detail=f"{edge:+.1%} over baseline",
                    contribution=edge,
                )
                for name, edge in edges
            ],
            invalidation=["the pattern's measured edge failing to reproduce out-of-sample"],
        )


#: Every model, in a fixed order so runs are reproducible.
ALL_MODELS: tuple[type[Predictor], ...] = (
    TechnicalModel,
    TimeSeriesModel,
    SimilarityModel,
    RegimeModel,
    SentimentModel,
    CrossAssetModel,
    OrderFlowModel,
    SequenceModel,
)


# ---------------------------------------------------------------------- helpers


def _autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    """Lag-k autocorrelation of a series."""
    if len(values) <= lag + 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values)
    if variance <= 0:
        return 0.0
    covariance = sum(
        (values[i] - mean) * (values[i - lag] - mean) for i in range(lag, len(values))
    )
    return covariance / variance


def _recent_return(candles: Sequence[Candle], bars: int) -> float | None:
    """Percentage change over the last ``bars`` bars."""
    if len(candles) < bars + 1:
        return None
    entry = candles[-bars - 1].close
    if entry <= 0:
        return None
    return (candles[-1].close - entry) / entry * 100.0
