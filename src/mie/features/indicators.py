"""Incremental technical indicators.

Each indicator is a small state machine: feed it one closed bar at a time and it
returns its current value, or ``None`` until it has enough history to be defined.

**Why incremental.** A new 1m bar arrives for every watched series every minute. If
each arrival triggered a recompute over months of history, the cost would scale with
history × assets × timeframes and the engine would fall over well before fifty
assets. Here a bar costs O(window) at worst, independent of how much history exists.

**Two update strategies, deliberately mixed.**

* *Windowed* indicators (SMA, Bollinger, Stochastic, realised volatility) recompute
  from a bounded deque of the last N bars. That is still O(1) with respect to total
  history, and it means the incremental result is **bit-identical** to computing the
  same window from scratch — no accumulator drift from summing in a different order.
* *Recursive* indicators (EMA, Wilder smoothing, OBV, VWAP) carry running state,
  because the recursion *is* their definition. A batch implementation performs the
  identical recursion, so these match exactly too.

That distinction is what lets the test suite assert exact equality against
independent reference implementations rather than settling for "close enough".

**Warmup conventions** are stated explicitly per indicator, because the common
disagreement between charting packages is not the formula but the seed. Ours:
EMA seeds from the SMA of its first ``period`` values; Wilder's smoothing likewise.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from datetime import date

from mie.core.types import Candle

__all__ = [
    "ADX",
    "ATR",
    "EMA",
    "MACD",
    "OBV",
    "ROC",
    "RSI",
    "SMA",
    "AnchoredVWAP",
    "BollingerBands",
    "Indicator",
    "RealisedVolatility",
    "Stochastic",
    "WilderMA",
]


class Indicator(ABC):
    """One indicator over one series.

    Subclasses implement :meth:`update`, which is called once per closed bar in
    strict chronological order and returns the indicator's value as of that bar.
    """

    #: Human-readable name; becomes the feature key in storage.
    name: str = "indicator"

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars required before the indicator produces a value."""

    @abstractmethod
    def update(self, candle: Candle) -> float | dict[str, float] | None:
        """Consume one bar and return the current value, or None while warming up."""

    def prime(self, candles: Iterable[Candle]) -> None:
        """Replay history to reach a warm state, discarding intermediate values."""
        for candle in candles:
            self.update(candle)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------- windowed


class SMA(Indicator):
    """Simple moving average of closes.

    Recomputed over the retained window rather than kept as a running sum: a running
    sum accumulates float error differently from a fresh summation, which would make
    the incremental result diverge from a batch one in the last bits.
    """

    def __init__(self, period: int, name: str | None = None) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = name or f"sma_{period}"
        self._window: deque[float] = deque(maxlen=period)

    @property
    def warmup(self) -> int:
        return self.period

    def update(self, candle: Candle) -> float | None:
        self._window.append(candle.close)
        if len(self._window) < self.period:
            return None
        return math.fsum(self._window) / self.period


class RealisedVolatility(Indicator):
    """Standard deviation of log returns over a window, annualised.

    Annualising makes volatility comparable across timeframes — a 2% daily sigma and
    a 0.4% hourly sigma are not obviously related until both are expressed per year.
    Uses the population standard deviation (the sample is the window itself).
    """

    def __init__(self, period: int, bars_per_year: float, name: str | None = None) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.bars_per_year = bars_per_year
        self.name = name or f"realised_vol_{period}"
        self._returns: deque[float] = deque(maxlen=period)
        self._previous_close: float | None = None

    @property
    def warmup(self) -> int:
        return self.period + 1

    def update(self, candle: Candle) -> float | None:
        if self._previous_close is not None and self._previous_close > 0 and candle.close > 0:
            self._returns.append(math.log(candle.close / self._previous_close))
        self._previous_close = candle.close
        if len(self._returns) < self.period:
            return None
        mean = math.fsum(self._returns) / self.period
        variance = math.fsum((r - mean) ** 2 for r in self._returns) / self.period
        return math.sqrt(variance) * math.sqrt(self.bars_per_year) * 100.0


class BollingerBands(Indicator):
    """SMA envelope at ±k population standard deviations.

    Emits ``%b`` (position within the bands) and bandwidth alongside the raw levels:
    the levels are what a chart draws, but the normalised pair is what a model can
    compare across assets whose prices differ by four orders of magnitude.
    """

    def __init__(self, period: int = 20, deviations: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.deviations = deviations
        self.name = f"bb_{period}"
        self._window: deque[float] = deque(maxlen=period)

    @property
    def warmup(self) -> int:
        return self.period

    def update(self, candle: Candle) -> dict[str, float] | None:
        self._window.append(candle.close)
        if len(self._window) < self.period:
            return None
        middle = math.fsum(self._window) / self.period
        variance = math.fsum((v - middle) ** 2 for v in self._window) / self.period
        sigma = math.sqrt(variance)
        upper = middle + self.deviations * sigma
        lower = middle - self.deviations * sigma
        width = upper - lower
        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            # Guard the degenerate zero-width case (a perfectly flat window) rather
            # than dividing by zero: mid-band is the honest answer there.
            "percent_b": 0.5 if width == 0 else (candle.close - lower) / width,
            "bandwidth": 0.0 if middle == 0 else width / middle * 100.0,
        }


class Stochastic(Indicator):
    """%K over the high/low range of the window, with an SMA-smoothed %D."""

    def __init__(self, period: int = 14, smooth_d: int = 3) -> None:
        if period < 1 or smooth_d < 1:
            raise ValueError("periods must be >= 1")
        self.period = period
        self.smooth_d = smooth_d
        self.name = f"stoch_{period}"
        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)
        self._k_history: deque[float] = deque(maxlen=smooth_d)

    @property
    def warmup(self) -> int:
        return self.period + self.smooth_d - 1

    def update(self, candle: Candle) -> dict[str, float] | None:
        self._highs.append(candle.high)
        self._lows.append(candle.low)
        if len(self._highs) < self.period:
            return None

        highest = max(self._highs)
        lowest = min(self._lows)
        span = highest - lowest
        # A flat range means every price is simultaneously the high and the low;
        # 50 (neutral) is the conventional and least misleading reading.
        k = 50.0 if span == 0 else (candle.close - lowest) / span * 100.0
        self._k_history.append(k)
        if len(self._k_history) < self.smooth_d:
            return None
        return {"k": k, "d": math.fsum(self._k_history) / self.smooth_d}


# -------------------------------------------------------------------- recursive


class EMA(Indicator):
    """Exponential moving average, alpha = 2/(period+1).

    Seeded with the SMA of the first ``period`` values. The seed is the usual point
    of disagreement between implementations, so it is fixed here and asserted in the
    tests rather than left to chance.
    """

    def __init__(self, period: int, name: str | None = None) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = name or f"ema_{period}"
        self.alpha = 2.0 / (period + 1.0)
        self._seed: list[float] = []
        self._value: float | None = None

    @property
    def warmup(self) -> int:
        return self.period

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, candle: Candle) -> float | None:
        return self.push(candle.close)

    def push(self, value: float) -> float | None:
        """Feed a raw number, so an EMA can be layered over another indicator."""
        if self._value is None:
            self._seed.append(value)
            if len(self._seed) < self.period:
                return None
            self._value = math.fsum(self._seed) / self.period
            self._seed.clear()
            return self._value
        self._value += self.alpha * (value - self._value)
        return self._value


class WilderMA(Indicator):
    """Wilder's smoothing, alpha = 1/period.

    Distinct from :class:`EMA` and not interchangeable with it — RSI, ATR and ADX are
    all defined in terms of this slower average, and substituting a standard EMA
    silently produces different numbers than every charting package shows.
    """

    def __init__(self, period: int, name: str | None = None) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = name or f"rma_{period}"
        self._seed: list[float] = []
        self._value: float | None = None

    @property
    def warmup(self) -> int:
        return self.period

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, candle: Candle) -> float | None:
        return self.push(candle.close)

    def push(self, value: float) -> float | None:
        if self._value is None:
            self._seed.append(value)
            if len(self._seed) < self.period:
                return None
            self._value = math.fsum(self._seed) / self.period
            self._seed.clear()
            return self._value
        self._value += (value - self._value) / self.period
        return self._value


class RSI(Indicator):
    """Relative strength index using Wilder's smoothing of gains and losses."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"rsi_{period}"
        self._gains = WilderMA(period)
        self._losses = WilderMA(period)
        self._previous_close: float | None = None

    @property
    def warmup(self) -> int:
        return self.period + 1

    def update(self, candle: Candle) -> float | None:
        if self._previous_close is None:
            self._previous_close = candle.close
            return None
        change = candle.close - self._previous_close
        self._previous_close = candle.close

        average_gain = self._gains.push(max(change, 0.0))
        average_loss = self._losses.push(max(-change, 0.0))
        if average_gain is None or average_loss is None:
            return None
        # No losses in the window: RSI is 100 by definition, not a division error.
        if average_loss == 0:
            return 100.0
        rs = average_gain / average_loss
        return 100.0 - (100.0 / (1.0 + rs))


class MACD(Indicator):
    """MACD line, signal line, and histogram.

    The signal EMA is fed only once the MACD line exists, so its own seed is the SMA
    of the first ``signal`` MACD values — not of a series padded with zeros, which
    would drag the early signal line toward zero and invent a crossover.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow")
        self.fast_period, self.slow_period, self.signal_period = fast, slow, signal
        self.name = f"macd_{fast}_{slow}_{signal}"
        self._fast = EMA(fast)
        self._slow = EMA(slow)
        self._signal = EMA(signal)

    @property
    def warmup(self) -> int:
        return self.slow_period + self.signal_period - 1

    def update(self, candle: Candle) -> dict[str, float] | None:
        fast = self._fast.push(candle.close)
        slow = self._slow.push(candle.close)
        if fast is None or slow is None:
            return None
        macd = fast - slow
        signal = self._signal.push(macd)
        if signal is None:
            return None
        return {"macd": macd, "signal": signal, "histogram": macd - signal}


class ATR(Indicator):
    """Average true range (Wilder), plus its percentage of price.

    The percentage form is what travels across assets: a $900 ATR on BTC and a
    $0.004 ATR on DOGE are the same statement about volatility once normalised.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"atr_{period}"
        self._rma = WilderMA(period)
        self._previous_close: float | None = None

    @property
    def warmup(self) -> int:
        return self.period + 1

    def update(self, candle: Candle) -> dict[str, float] | None:
        true_range = _true_range(candle, self._previous_close)
        self._previous_close = candle.close
        value = self._rma.push(true_range)
        if value is None:
            return None
        return {
            "atr": value,
            "atr_pct": 0.0 if candle.close == 0 else value / candle.close * 100.0,
        }


class ADX(Indicator):
    """Average directional index with +DI and -DI (Wilder).

    ADX measures trend *strength* without direction; the DI pair supplies direction.
    Both are needed — a strong ADX says nothing about which way, and a DI crossover
    in a rangebound market is noise.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"adx_{period}"
        self._tr = WilderMA(period)
        self._plus = WilderMA(period)
        self._minus = WilderMA(period)
        self._adx = WilderMA(period)
        self._previous: Candle | None = None

    @property
    def warmup(self) -> int:
        # One bar to establish directional movement, then two Wilder averages in series.
        return self.period * 2 + 1

    def update(self, candle: Candle) -> dict[str, float] | None:
        previous = self._previous
        self._previous = candle
        if previous is None:
            return None

        up_move = candle.high - previous.high
        down_move = previous.low - candle.low
        # Directional movement counts only when one side clearly dominates; an inside
        # bar or an outside bar with equal extension contributes nothing either way.
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        true_range = self._tr.push(_true_range(candle, previous.close))
        plus = self._plus.push(plus_dm)
        minus = self._minus.push(minus_dm)
        if true_range is None or plus is None or minus is None or true_range == 0:
            return None

        plus_di = plus / true_range * 100.0
        minus_di = minus / true_range * 100.0
        di_sum = plus_di + minus_di
        dx = 0.0 if di_sum == 0 else abs(plus_di - minus_di) / di_sum * 100.0
        adx = self._adx.push(dx)
        if adx is None:
            return None
        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


class OBV(Indicator):
    """On-balance volume: cumulative volume signed by the direction of the close.

    Reported as a change rate rather than only the raw cumulative total, because the
    absolute level depends on how far back ingestion happens to start and is
    therefore not comparable between series.
    """

    def __init__(self, change_window: int = 20) -> None:
        self.name = "obv"
        self.change_window = change_window
        self._value = 0.0
        self._previous_close: float | None = None
        self._history: deque[float] = deque(maxlen=change_window + 1)

    @property
    def warmup(self) -> int:
        return 2

    def update(self, candle: Candle) -> dict[str, float] | None:
        if self._previous_close is not None:
            if candle.close > self._previous_close:
                self._value += candle.volume
            elif candle.close < self._previous_close:
                self._value -= candle.volume
            # An unchanged close contributes nothing, by definition.
        self._previous_close = candle.close
        self._history.append(self._value)
        if len(self._history) < 2:
            return None
        oldest = self._history[0]
        return {
            "obv": self._value,
            "obv_change": self._value - oldest,
        }


class AnchoredVWAP(Indicator):
    """Volume-weighted average price, re-anchored each UTC day.

    Crypto has no session close, so VWAP needs an explicit anchor or it becomes an
    all-time average that never moves. UTC midnight is the convention the rest of the
    system already uses for daily bars, so the two agree by construction.
    """

    def __init__(self) -> None:
        self.name = "vwap"
        self._anchor: date | None = None
        self._pv = 0.0
        self._volume = 0.0

    @property
    def warmup(self) -> int:
        return 1

    def update(self, candle: Candle) -> dict[str, float] | None:
        day = candle.open_time.date()
        if self._anchor != day:
            self._anchor = day
            self._pv = 0.0
            self._volume = 0.0

        typical = (candle.high + candle.low + candle.close) / 3.0
        self._pv += typical * candle.volume
        self._volume += candle.volume
        if self._volume == 0:
            # A day of zero-volume bars has no volume-weighted price to report.
            return None
        vwap = self._pv / self._volume
        return {
            "vwap": vwap,
            "vwap_distance_pct": 0.0 if vwap == 0 else (candle.close - vwap) / vwap * 100.0,
        }


class ROC(Indicator):
    """Rate of change over ``period`` bars — the plainest momentum measure there is."""

    def __init__(self, period: int = 10) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"roc_{period}"
        self._closes: deque[float] = deque(maxlen=period + 1)

    @property
    def warmup(self) -> int:
        return self.period + 1

    def update(self, candle: Candle) -> float | None:
        self._closes.append(candle.close)
        if len(self._closes) < self.period + 1:
            return None
        reference = self._closes[0]
        if reference == 0:
            return None
        return (candle.close - reference) / reference * 100.0


# ---------------------------------------------------------------------- helpers


def _true_range(candle: Candle, previous_close: float | None) -> float:
    """Wilder's true range.

    The first bar has no previous close, so its true range is simply its own span —
    the alternative (skipping it) would shift every subsequent average by one bar.
    """
    span = candle.high - candle.low
    if previous_close is None:
        return span
    return max(
        span,
        abs(candle.high - previous_close),
        abs(candle.low - previous_close),
    )
