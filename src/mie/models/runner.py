"""Building prediction contexts, and running models over history.

This module is the only place that touches stored data on the models' behalf, which is
deliberate: it is the single point where look-ahead could enter, so it is the single
point that has to be right.

Every context is built from a bar index `i`, and everything it contains is drawn from
`candles[:i + 1]`. The realised return is read from `candles[i + horizon]`, and the two
never touch. A model receives the context and has no other handle on the world.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.models.base import PredictionContext
from mie.models.types import Horizon

log = get_logger(__name__)

__all__ = ["ContextSource", "build_contexts"]


class ContextSource:
    """Pre-loaded history for one asset, sliced into contexts on demand.

    Loading once and slicing is not just an optimisation: re-querying per point would
    make it easy to accidentally fetch an unbounded range and reintroduce the very
    leak this module exists to prevent.
    """

    def __init__(
        self,
        asset: str,
        timeframe: Timeframe,
        candles: Sequence[Candle],
        feature_history: Sequence[tuple[datetime, Mapping[str, float]]] = (),
        peers: Mapping[str, Sequence[Candle]] | None = None,
        funding: Sequence[tuple[datetime, float]] = (),
        open_interest: Sequence[tuple[datetime, float]] = (),
        news: Sequence[object] = (),
        data_quality: float = 1.0,
    ) -> None:
        self.asset = asset.upper()
        self.timeframe = timeframe
        self.candles = [c for c in candles if c.is_final]
        self.feature_history = sorted(feature_history, key=lambda pair: pair[0])
        self.peers = {k: [c for c in v if c.is_final] for k, v in (peers or {}).items()}
        self.funding = sorted(funding, key=lambda pair: pair[0])
        self.open_interest = sorted(open_interest, key=lambda pair: pair[0])
        self.news = list(news)
        self.data_quality = data_quality

    def context_at(self, index: int, horizon: Horizon) -> PredictionContext | None:
        """Build the context as it stood at ``candles[index]``.

        ``as_of`` is that bar's *close*: the bar has completed, so its data is known,
        and nothing after it is.
        """
        if index < 1 or index >= len(self.candles):
            return None
        as_of = self.timeframe.close_time(self.candles[index].open_time)
        history = self.candles[: index + 1]

        features_upto = [
            (moment, values)
            for moment, values in self.feature_history
            if self.timeframe.close_time(moment) <= as_of
        ]
        latest_features = features_upto[-1][1] if features_upto else {}

        return PredictionContext(
            asset=self.asset,
            timeframe=self.timeframe,
            as_of=as_of,
            horizon=horizon,
            candles=history,
            features=latest_features,
            feature_history=features_upto,
            state=self._state(latest_features),
            peers={
                name: [c for c in candles if self.timeframe.close_time(c.open_time) <= as_of]
                for name, candles in self.peers.items()
            },
            news=[
                event
                for event in self.news
                if getattr(event, "published_at", None) is not None
                and event.published_at <= as_of  # type: ignore[attr-defined]
            ],
            funding=[(t, v) for t, v in self.funding if t <= as_of],
            open_interest=[(t, v) for t, v in self.open_interest if t <= as_of],
            data_quality=self.data_quality,
            regime=_regime_of(history),
        )

    def _state(self, features: Mapping[str, float]) -> dict[str, object]:
        """A compact market-state mapping derived from the latest features.

        Deliberately derived here rather than by importing the Phase 3 engine: the
        state engine reads from the database, and giving a model a database handle
        during a backtest is exactly the door this module keeps shut.
        """
        if not features:
            return {}
        close = features.get("close", 0.0)
        fast = features.get("ema_21")
        slow = features.get("sma_200")
        trend = features.get("structure_trend", 0.0)
        if not close or not fast or not slow:
            return {}

        bias = 0.0
        if slow > 0:
            bias += max(-1.0, min(1.0, (close - slow) / slow * 10.0)) * 0.5
        if fast > 0:
            bias += max(-1.0, min(1.0, (close - fast) / fast * 20.0)) * 0.3
        bias += float(trend) * 0.2

        agreement = 1.0 if (bias > 0) == (trend >= 0) else 0.4
        alignment = (
            "aligned_bullish"
            if bias > 0.15 and trend >= 0
            else "aligned_bearish"
            if bias < -0.15 and trend <= 0
            else "conflicted"
            if abs(bias) > 0.15
            else "rangebound"
        )
        return {
            "bias_score": round(max(-1.0, min(1.0, bias)), 4),
            "agreement": agreement,
            "alignment": alignment,
            "confidence": 0.5,
        }


def build_contexts(
    source: ContextSource,
    horizon: Horizon,
    warmup: int = 400,
    stride: int | None = None,
    max_points: int | None = None,
) -> list[tuple[PredictionContext, float]]:
    """Walk history forward, yielding (context, realised return) pairs.

    ``stride`` defaults to the horizon length so evaluation windows tile without
    overlapping. Overlapping windows are not independent samples, and counting them as
    such inflates the apparent sample size while shrinking every confidence interval.
    """
    step = stride or horizon.bars
    candles = source.candles
    pairs: list[tuple[PredictionContext, float]] = []

    for index in range(warmup, len(candles) - horizon.bars, max(1, step)):
        context = source.context_at(index, horizon)
        if context is None:
            continue
        entry = candles[index].close
        if entry <= 0:
            continue
        realised = (candles[index + horizon.bars].close - entry) / entry * 100.0
        pairs.append((context, realised))
        if max_points and len(pairs) >= max_points:
            break
    return pairs


def _regime_of(candles: Sequence[Candle], window: int = 100) -> str:
    """A coarse regime label for slicing results.

    Intentionally crude and computed from price alone. Its job is to partition the
    evaluation so that "works in trends, fails in chop" is visible; a richer label
    would be harder to reproduce and would not change that partition much.
    """
    recent = candles[-window:]
    if len(recent) < 20:
        return "unknown"
    closes = [c.close for c in recent if c.close > 0]
    if len(closes) < 20:
        return "unknown"

    returns = [
        abs(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))
    ]
    typical = sorted(returns)[len(returns) // 2]
    drift = (closes[-1] - closes[0]) / closes[0]

    # Trend is measured against the market's own noise: the same 5% move is a strong
    # trend in a calm market and nothing in a violent one.
    noise = typical * (len(closes) ** 0.5)
    if noise <= 0:
        return "unknown"
    strength = drift / noise

    if abs(strength) < 0.5:
        return "range_high_vol" if typical > 0.004 else "range_low_vol"
    if strength > 0:
        return "uptrend_high_vol" if typical > 0.004 else "uptrend_low_vol"
    return "downtrend_high_vol" if typical > 0.004 else "downtrend_low_vol"
