"""Market structure: swings, support/resistance, volume profile, Fibonacci levels.

These differ from the streaming indicators in `indicators.py` in an important way:
they describe a *window of structure* rather than a value at a point, so they are
computed over a bounded lookback rather than carried as recursive state.

The lookback is bounded and small, so this is still O(1) with respect to total
history — the constraint Phase 2 has to honour is "no full historical recompute per
bar", not "never look at more than one bar".

One rule governs everything here: **a swing is only confirmed once enough bars have
formed after it**. A pivot identified from the most recent bar is not a pivot, it is a
guess that will be revised, and treating it as confirmed is how look-ahead bias
sneaks into structure analysis. Confirmation costs `right` bars of lag, and that lag
is the honest price of knowing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from mie.core.types import Candle

__all__ = [
    "Level",
    "MarketStructure",
    "StructureAnalyzer",
    "Swing",
    "cluster_levels",
    "fibonacci_levels",
    "find_swings",
    "volume_profile",
]

SwingKind = Literal["high", "low"]


@dataclass(frozen=True, slots=True)
class Swing:
    """A confirmed pivot: an extreme with `left` lower bars before and `right` after."""

    kind: SwingKind
    price: float
    at: datetime
    index: int


@dataclass(frozen=True, slots=True)
class Level:
    """A price level supported by clustered swings.

    ``touches`` is the evidence: a level tested five times is a different object from
    one drawn through a single pivot, and the count is what lets later phases weight
    them instead of treating all lines as equal.
    """

    price: float
    kind: SwingKind
    touches: int
    first_seen: datetime
    last_seen: datetime

    def distance_pct(self, price: float) -> float:
        return 0.0 if price == 0 else (self.price - price) / price * 100.0


def find_swings(
    candles: Sequence[Candle], left: int = 2, right: int = 2
) -> list[Swing]:
    """Confirmed swing highs and lows.

    A bar is a swing high when its high is the strict maximum over the window
    ``[i-left, i+right]``. The last ``right`` bars can never qualify — their
    confirmation window is not complete yet — and they are deliberately not examined.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    swings: list[Swing] = []
    for i in range(left, len(candles) - right):
        window = candles[i - left : i + right + 1]
        pivot = candles[i]
        if pivot.high == max(c.high for c in window) and _is_strict_max(window, pivot, "high"):
            swings.append(Swing("high", pivot.high, pivot.open_time, i))
        if pivot.low == min(c.low for c in window) and _is_strict_max(window, pivot, "low"):
            swings.append(Swing("low", pivot.low, pivot.open_time, i))
    return swings


def _is_strict_max(window: Sequence[Candle], pivot: Candle, field: str) -> bool:
    """Reject plateaus, where several adjacent bars share the same extreme.

    Without this a flat top produces one 'pivot' per bar, and the level clustering
    downstream then reports a wall of overlapping levels that are all the same line.
    """
    value = getattr(pivot, field)
    matches = sum(1 for candle in window if getattr(candle, field) == value)
    return matches == 1


def cluster_levels(
    swings: Sequence[Swing], tolerance_pct: float = 0.5, min_touches: int = 2
) -> list[Level]:
    """Merge nearby swings of the same kind into levels.

    Price rarely turns at exactly the same number twice; it turns in a *zone*.
    Clustering within a percentage tolerance is what converts a scatter of pivots
    into the handful of levels a human would actually draw.
    """
    levels: list[Level] = []
    for kind in ("high", "low"):
        candidates = sorted(
            (s for s in swings if s.kind == kind), key=lambda s: s.price
        )
        cluster: list[Swing] = []
        for swing in candidates:
            if cluster and abs(swing.price - cluster[0].price) / cluster[0].price * 100.0 > tolerance_pct:
                levels.append(_to_level(cluster, kind))
                cluster = []
            cluster.append(swing)
        if cluster:
            levels.append(_to_level(cluster, kind))

    return sorted(
        (lvl for lvl in levels if lvl.touches >= min_touches),
        key=lambda lvl: -lvl.touches,
    )


def _to_level(cluster: Sequence[Swing], kind: SwingKind) -> Level:
    # The mean of the cluster represents the zone better than any single pivot.
    price = sum(s.price for s in cluster) / len(cluster)
    times = sorted(s.at for s in cluster)
    return Level(price, kind, len(cluster), times[0], times[-1])


def volume_profile(
    candles: Sequence[Candle], bins: int = 24, value_area: float = 0.70
) -> dict[str, float] | None:
    """Distribution of traded volume across price, with POC and value area.

    Volume distributed by *price* rather than by time answers a different question
    than a volume bar does: where did participants actually transact? The
    point-of-control and value-area edges are the levels that fall out of it.

    Volume is spread evenly across each bar's high-low span. That is an
    approximation — the true intrabar distribution is unknowable from OHLCV alone —
    and it is the standard one.
    """
    if not candles:
        return None
    highest = max(c.high for c in candles)
    lowest = min(c.low for c in candles)
    if highest <= lowest or bins < 2:
        return None

    step = (highest - lowest) / bins
    buckets = [0.0] * bins

    for candle in candles:
        low_bin = min(bins - 1, max(0, int((candle.low - lowest) / step)))
        high_bin = min(bins - 1, max(0, int((candle.high - lowest) / step)))
        spread = high_bin - low_bin + 1
        share = candle.volume / spread
        for index in range(low_bin, high_bin + 1):
            buckets[index] += share

    total = sum(buckets)
    if total <= 0:
        return None

    poc_index = max(range(bins), key=lambda i: buckets[i])
    # Grow outward from the point of control, always taking the heavier neighbour,
    # until the requested share of volume is enclosed.
    included = {poc_index}
    covered = buckets[poc_index]
    low_edge = high_edge = poc_index
    while covered < total * value_area and len(included) < bins:
        below = buckets[low_edge - 1] if low_edge > 0 else -1.0
        above = buckets[high_edge + 1] if high_edge < bins - 1 else -1.0
        if above >= below:
            high_edge += 1
            covered += buckets[high_edge]
            included.add(high_edge)
        else:
            low_edge -= 1
            covered += buckets[low_edge]
            included.add(low_edge)

    centre = lambda i: lowest + step * (i + 0.5)  # noqa: E731 - local shorthand
    return {
        "poc": centre(poc_index),
        "value_area_high": lowest + step * (high_edge + 1),
        "value_area_low": lowest + step * low_edge,
        "profile_high": highest,
        "profile_low": lowest,
    }


def fibonacci_levels(high: float, low: float, uptrend: bool = True) -> dict[str, float]:
    """Retracement levels between a swing high and low.

    Direction matters: in an uptrend the levels are measured down from the high as
    potential support, and in a downtrend up from the low as potential resistance.
    Computing them in the wrong direction produces numbers that look plausible and
    mean nothing.
    """
    span = high - low
    ratios = {"0.236": 0.236, "0.382": 0.382, "0.5": 0.5, "0.618": 0.618, "0.786": 0.786}
    if uptrend:
        return {f"fib_{name}": high - span * ratio for name, ratio in ratios.items()}
    return {f"fib_{name}": low + span * ratio for name, ratio in ratios.items()}


@dataclass(slots=True)
class MarketStructure:
    """Structural summary of a lookback window."""

    swings: list[Swing]
    levels: list[Level]
    trend: str
    nearest_support: Level | None
    nearest_resistance: Level | None
    profile: dict[str, float] | None
    fib: dict[str, float]

    def as_features(self, price: float) -> dict[str, float]:
        """Flatten to the numeric feature keys the engine stores."""
        features: dict[str, float] = {
            "structure_trend": {"up": 1.0, "down": -1.0, "range": 0.0}[self.trend],
            "swing_count": float(len(self.swings)),
            "level_count": float(len(self.levels)),
        }
        if self.nearest_support:
            features["support"] = self.nearest_support.price
            features["support_distance_pct"] = self.nearest_support.distance_pct(price)
            features["support_touches"] = float(self.nearest_support.touches)
        if self.nearest_resistance:
            features["resistance"] = self.nearest_resistance.price
            features["resistance_distance_pct"] = self.nearest_resistance.distance_pct(price)
            features["resistance_touches"] = float(self.nearest_resistance.touches)
        if self.profile:
            features |= {
                "poc": self.profile["poc"],
                "value_area_high": self.profile["value_area_high"],
                "value_area_low": self.profile["value_area_low"],
            }
        features |= self.fib
        return features


class StructureAnalyzer:
    """Maintains a bounded window of bars and derives structure from it.

    Structure changes slowly, so recomputing it on every single bar is wasted work;
    ``refresh_every`` throttles it and the previous result is reused in between.
    """

    def __init__(
        self,
        lookback: int = 120,
        left: int = 2,
        right: int = 2,
        tolerance_pct: float = 0.5,
        refresh_every: int = 5,
    ) -> None:
        self.lookback = lookback
        self.left = left
        self.right = right
        self.tolerance_pct = tolerance_pct
        self.refresh_every = max(1, refresh_every)
        self._window: deque[Candle] = deque(maxlen=lookback)
        self._counter = 0
        self._cached: MarketStructure | None = None

    @property
    def warmup(self) -> int:
        return self.left + self.right + 2

    def update(self, candle: Candle) -> MarketStructure | None:
        self._window.append(candle)
        self._counter += 1
        if len(self._window) < self.warmup:
            return None
        if self._cached is None or self._counter % self.refresh_every == 0:
            self._cached = self._analyse()
        return self._cached

    def _analyse(self) -> MarketStructure:
        candles = list(self._window)
        swings = find_swings(candles, self.left, self.right)
        levels = cluster_levels(swings, self.tolerance_pct)
        price = candles[-1].close

        below = [lvl for lvl in levels if lvl.price < price]
        above = [lvl for lvl in levels if lvl.price > price]
        support = max(below, key=lambda lvl: lvl.price) if below else None
        resistance = min(above, key=lambda lvl: lvl.price) if above else None

        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]
        trend = _classify_trend(highs, lows)

        window_high = max(c.high for c in candles)
        window_low = min(c.low for c in candles)
        return MarketStructure(
            swings=swings,
            levels=levels,
            trend=trend,
            nearest_support=support,
            nearest_resistance=resistance,
            profile=volume_profile(candles),
            fib=fibonacci_levels(window_high, window_low, uptrend=trend != "down"),
        )


def _classify_trend(highs: Sequence[Swing], lows: Sequence[Swing]) -> str:
    """Higher highs and higher lows, or the reverse; anything else is a range.

    Requiring *both* sides to agree is deliberate. Higher highs with lower lows is a
    broadening formation, not an uptrend, and calling it one would be wrong in the
    most expensive way.
    """
    if len(highs) < 2 or len(lows) < 2:
        return "range"
    higher_highs = highs[-1].price > highs[-2].price
    higher_lows = lows[-1].price > lows[-2].price
    if higher_highs and higher_lows:
        return "up"
    if not higher_highs and not higher_lows:
        return "down"
    return "range"
