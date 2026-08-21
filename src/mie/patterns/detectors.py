"""Pattern detectors.

Each detector examines a window of closed bars ending at the bar under test and
answers a single yes/no question, returning a :class:`Detection` when the answer is
yes.

Three rules apply to every detector without exception:

* **Only closed bars.** A pattern confirmed by a forming bar is not a detection, it is
  a guess about the rest of the current bar.
* **No forward reference.** A detector may look only at ``candles[:index + 1]``. This
  is enforced structurally by passing a slice, not by convention.
* **The conventional reading is fixed in advance.** Whether a breakout "should" be
  bullish is declared in `PATTERN_DIRECTIONS` before anything is measured. Deciding a
  pattern's direction after seeing which way price went is not analysis, it is
  relabelling.

Thresholds here are conventional starting points, not tuned parameters. Tuning them
against the same history used to measure them is how a backtest becomes fiction —
Phase 8's walk-forward harness is where any such choice has to prove itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import median

from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.patterns.types import PATTERN_DIRECTIONS, Detection, PatternKind

__all__ = ["DETECTORS", "Detector", "detect_all"]

#: A detector inspects bars up to and including the last one and returns a detection
#: for that bar, or None.
Detector = Callable[[Sequence[Candle], str, Timeframe], "Detection | None"]

_LOOKBACK = 20
_VOLUME_LOOKBACK = 20


def _make(
    kind: PatternKind,
    candles: Sequence[Candle],
    asset: str,
    timeframe: Timeframe,
    quality: float,
    detail: str,
    **context: float,
) -> Detection:
    bar = candles[-1]
    return Detection(
        kind=kind,
        asset=asset.upper(),
        timeframe=timeframe,
        at=bar.open_time,
        direction=PATTERN_DIRECTIONS[kind],
        quality=max(0.0, min(1.0, quality)),
        close=bar.close,
        detail=detail,
        context=dict(context),
    )


def _volume_ratio(candles: Sequence[Candle]) -> float:
    """Current volume against its recent average. 1.0 means typical."""
    window = candles[-_VOLUME_LOOKBACK - 1 : -1]
    if not window:
        return 1.0
    average = sum(c.volume for c in window) / len(window)
    return candles[-1].volume / average if average > 0 else 1.0


def _true_range(candles: Sequence[Candle], index: int) -> float:
    bar = candles[index]
    if index == 0:
        return bar.high - bar.low
    previous = candles[index - 1].close
    return max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous))


def _average_range(candles: Sequence[Candle], period: int = 14) -> float:
    start = max(1, len(candles) - period)
    ranges = [_true_range(candles, i) for i in range(start, len(candles))]
    return sum(ranges) / len(ranges) if ranges else 0.0


# ------------------------------------------------------------------ breakouts


def breakout(candles: Sequence[Candle], asset: str, timeframe: Timeframe) -> Detection | None:
    """Close beyond the prior N-bar extreme on expanded volume.

    Volume confirmation is required because an unconfirmed break of a range is the
    textbook description of a fakeout, and treating the two identically would blend a
    signal with its own opposite.
    """
    if len(candles) < _LOOKBACK + 2:
        return None
    bar = candles[-1]
    prior = candles[-_LOOKBACK - 1 : -1]
    highest = max(c.high for c in prior)
    lowest = min(c.low for c in prior)
    volume_ratio = _volume_ratio(candles)

    if bar.close > highest and volume_ratio > 1.2:
        margin = (bar.close - highest) / highest * 100.0
        return _make(
            PatternKind.BREAKOUT_UP, candles, asset, timeframe,
            quality=min(1.0, 0.4 + margin * 0.2 + (volume_ratio - 1.2) * 0.3),
            detail=f"closed {margin:.2f}% above the {_LOOKBACK}-bar high on {volume_ratio:.1f}x volume",
            margin_pct=margin, volume_ratio=volume_ratio,
        )
    if bar.close < lowest and volume_ratio > 1.2:
        margin = (lowest - bar.close) / lowest * 100.0
        return _make(
            PatternKind.BREAKOUT_DOWN, candles, asset, timeframe,
            quality=min(1.0, 0.4 + margin * 0.2 + (volume_ratio - 1.2) * 0.3),
            detail=f"closed {margin:.2f}% below the {_LOOKBACK}-bar low on {volume_ratio:.1f}x volume",
            margin_pct=margin, volume_ratio=volume_ratio,
        )
    return None


#: A penetration smaller than this fraction of ATR is a tick, not a breach.
_MIN_PENETRATION_ATR = 0.15
#: The rejected portion must be a real share of the bar's own range, or the "failure"
#: is just where the bar happened to close.
_MIN_REJECTION_SHARE = 0.40


def fakeout(candles: Sequence[Candle], asset: str, timeframe: Timeframe) -> Detection | None:
    """Price breached the prior extreme intrabar and was rejected back inside it.

    The failure of a breakout carries the opposite implication to its success, which is
    why the two are separate patterns rather than one pattern with a caveat.

    Both the breach and the rejection must be **material**. Requiring only
    ``high > prior_high and close < prior_high`` looks reasonable and is badly wrong:
    on any gently rising series the bar's high clears the old high almost every bar
    while the close sits below it, so the detector fired on 37% of bars — a "pattern"
    present in a third of all history describes the market rather than signalling
    anything within it.
    """
    if len(candles) < _LOOKBACK + 2:
        return None
    bar = candles[-1]
    prior = candles[-_LOOKBACK - 1 : -1]
    highest = max(c.high for c in prior)
    lowest = min(c.low for c in prior)
    average_range = _average_range(candles)
    bar_range = bar.high - bar.low
    if average_range <= 0 or bar_range <= 0:
        return None

    if bar.high > highest and bar.close < highest:
        penetration = (bar.high - highest) / average_range
        rejection_share = (bar.high - bar.close) / bar_range
        material = (
            penetration >= _MIN_PENETRATION_ATR
            and rejection_share >= _MIN_REJECTION_SHARE
        )
        if material:
            rejection = (bar.high - bar.close) / bar.high * 100.0
            return _make(
                PatternKind.FAKEOUT_UP, candles, asset, timeframe,
                quality=min(1.0, 0.35 + penetration * 0.3 + rejection_share * 0.3),
                detail=(
                    f"breached the {_LOOKBACK}-bar high by {penetration:.2f}x ATR then "
                    f"closed {rejection:.2f}% back inside"
                ),
                penetration_atr=penetration, rejection_share=rejection_share,
            )
    if bar.low < lowest and bar.close > lowest:
        penetration = (lowest - bar.low) / average_range
        rejection_share = (bar.close - bar.low) / bar_range
        material = (
            penetration >= _MIN_PENETRATION_ATR
            and rejection_share >= _MIN_REJECTION_SHARE
        )
        if material:
            rejection = (bar.close - bar.low) / bar.close * 100.0
            return _make(
                PatternKind.FAKEOUT_DOWN, candles, asset, timeframe,
                quality=min(1.0, 0.35 + penetration * 0.3 + rejection_share * 0.3),
                detail=(
                    f"breached the {_LOOKBACK}-bar low by {penetration:.2f}x ATR then "
                    f"closed {rejection:.2f}% back inside"
                ),
                penetration_atr=penetration, rejection_share=rejection_share,
            )
    return None


def liquidity_sweep(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """A long wick through a prior extreme with the body closing back inside.

    Distinguished from a fakeout by requiring a *large* wick relative to recent range:
    the signature of stops being taken out rather than of a marginal failed break.
    """
    if len(candles) < _LOOKBACK + 2:
        return None
    bar = candles[-1]
    prior = candles[-_LOOKBACK - 1 : -1]
    highest = max(c.high for c in prior)
    lowest = min(c.low for c in prior)
    average_range = _average_range(candles)
    if average_range <= 0:
        return None

    body_top = max(bar.open, bar.close)
    body_bottom = min(bar.open, bar.close)
    upper_wick = bar.high - body_top
    lower_wick = body_bottom - bar.low

    if bar.high > highest and bar.close < highest and upper_wick > average_range * 0.8:
        return _make(
            PatternKind.LIQUIDITY_SWEEP_HIGH, candles, asset, timeframe,
            quality=min(1.0, 0.45 + upper_wick / average_range * 0.2),
            detail=f"upper wick {upper_wick / average_range:.1f}x ATR swept the prior high",
            wick_atr=upper_wick / average_range,
        )
    if bar.low < lowest and bar.close > lowest and lower_wick > average_range * 0.8:
        return _make(
            PatternKind.LIQUIDITY_SWEEP_LOW, candles, asset, timeframe,
            quality=min(1.0, 0.45 + lower_wick / average_range * 0.2),
            detail=f"lower wick {lower_wick / average_range:.1f}x ATR swept the prior low",
            wick_atr=lower_wick / average_range,
        )
    return None


# ------------------------------------------------------------- volatility state


def compression(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """Range contracting well below its own recent norm.

    Direction-neutral by construction: compression implies an expansion is coming, and
    claiming to know which way it resolves is precisely the unearned certainty this
    project exists to avoid.
    """
    if len(candles) < 60:
        return None
    recent = _average_range(candles, period=10)
    baseline = _average_range(candles, period=50)
    if baseline <= 0 or recent <= 0:
        return None
    ratio = recent / baseline
    if ratio > 0.6:
        return None
    return _make(
        PatternKind.COMPRESSION, candles, asset, timeframe,
        quality=min(1.0, 0.4 + (0.6 - ratio)),
        detail=f"10-bar range is {ratio:.0%} of the 50-bar average",
        range_ratio=ratio,
    )


def expansion(candles: Sequence[Candle], asset: str, timeframe: Timeframe) -> Detection | None:
    """Range expanding sharply above its recent norm."""
    if len(candles) < 60:
        return None
    recent = _average_range(candles, period=5)
    baseline = _average_range(candles, period=50)
    if baseline <= 0:
        return None
    ratio = recent / baseline
    if ratio < 1.8:
        return None
    return _make(
        PatternKind.EXPANSION, candles, asset, timeframe,
        quality=min(1.0, 0.4 + (ratio - 1.8) * 0.2),
        detail=f"5-bar range is {ratio:.1f}x the 50-bar average",
        range_ratio=ratio,
    )


def volume_anomaly(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """Volume far outside its recent distribution.

    Uses a median-and-MAD threshold rather than a mean-and-sigma one: volume is
    heavily right-skewed, and a single spike inflates a standard deviation enough to
    conceal itself.
    """
    if len(candles) < _VOLUME_LOOKBACK + 2:
        return None
    window = [c.volume for c in candles[-_VOLUME_LOOKBACK - 1 : -1]]
    centre = median(window)
    if centre <= 0:
        return None
    current = candles[-1].volume
    deviations = median([abs(v - centre) for v in window]) * 1.4826
    if deviations <= 0:
        # Perfectly uniform recent volume leaves no scale to measure against, but a
        # large departure from it is still plainly anomalous. Falling through to
        # `return None` here would make the detector blind in exactly the case where
        # the anomaly is most obvious.
        if current < centre * 3.0:
            return None
        z = 4.0 + (current / centre - 3.0)
    else:
        z = (current - centre) / deviations
    if z < 4.0:
        return None
    return _make(
        PatternKind.VOLUME_ANOMALY, candles, asset, timeframe,
        quality=min(1.0, 0.4 + (z - 4.0) * 0.05),
        detail=f"volume {z:.1f} robust sigma above the {_VOLUME_LOOKBACK}-bar median",
        volume_z=z,
    )


# ------------------------------------------------------ accumulation/distribution


def accumulation_distribution(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """A flat price range with volume flowing persistently one way.

    Price going nowhere while volume is consistently absorbed on one side is the
    classic description of accumulation (or its mirror). Requires the range to be
    genuinely quiet, so an ordinary trend is not mislabelled.
    """
    if len(candles) < 40:
        return None
    window = candles[-30:]
    highest = max(c.high for c in window)
    lowest = min(c.low for c in window)
    if lowest <= 0:
        return None
    range_pct = (highest - lowest) / lowest * 100.0
    net_move = (window[-1].close - window[0].close) / window[0].close * 100.0
    if range_pct > 12.0 or abs(net_move) > range_pct * 0.35:
        return None  # trending, not ranging

    up_volume = sum(
        c.volume for i, c in enumerate(window) if i and c.close > window[i - 1].close
    )
    down_volume = sum(
        c.volume for i, c in enumerate(window) if i and c.close < window[i - 1].close
    )
    total = up_volume + down_volume
    if total <= 0:
        return None
    imbalance = (up_volume - down_volume) / total
    if abs(imbalance) < 0.25:
        return None

    kind = (
        PatternKind.ACCUMULATION if imbalance > 0 else PatternKind.DISTRIBUTION
    )
    return _make(
        kind, candles, asset, timeframe,
        quality=min(1.0, 0.4 + abs(imbalance)),
        detail=(
            f"{range_pct:.1f}% range with {abs(imbalance):.0%} volume imbalance "
            f"{'upward' if imbalance > 0 else 'downward'}"
        ),
        range_pct=range_pct, imbalance=imbalance,
    )


# ----------------------------------------------------------------- divergences


def _swing_points(
    candles: Sequence[Candle], left: int = 2, right: int = 2
) -> tuple[list[int], list[int]]:
    """Indices of confirmed swing highs and lows within the window."""
    highs, lows = [], []
    for i in range(left, len(candles) - right):
        window = candles[i - left : i + right + 1]
        # The extreme must be unique in the window: a flat top would otherwise
        # register one "pivot" per bar along it.
        high = candles[i].high
        if high == max(c.high for c in window) and sum(
            1 for c in window if c.high == high
        ) == 1:
            highs.append(i)
        low = candles[i].low
        if low == min(c.low for c in window) and sum(
            1 for c in window if c.low == low
        ) == 1:
            lows.append(i)
    return highs, lows


def _rsi_series(candles: Sequence[Candle], period: int = 14) -> list[float | None]:
    """Wilder RSI over the window — recomputed locally so detectors stay standalone."""
    values: list[float | None] = [None] * len(candles)
    if len(candles) < period + 1:
        return values
    gains = [max(candles[i].close - candles[i - 1].close, 0.0) for i in range(1, len(candles))]
    losses = [max(candles[i - 1].close - candles[i].close, 0.0) for i in range(1, len(candles))]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains) + 1):
        if i > period:
            avg_gain += (gains[i - 1] - avg_gain) / period
            avg_loss += (losses[i - 1] - avg_loss) / period
        values[i] = (
            100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / max(avg_loss, 1e-12))
        )
    return values


def divergence(candles: Sequence[Candle], asset: str, timeframe: Timeframe) -> Detection | None:
    """Price makes a new extreme that momentum does not confirm.

    Only fires on the bar where the second swing is *confirmed*, which lags the swing
    itself — that lag is the honest cost of knowing a pivot was a pivot.
    """
    if len(candles) < 60:
        return None
    window = candles[-60:]
    rsi = _rsi_series(window)
    highs, lows = _swing_points(window)

    # The most recent confirmed swing must be near the end of the window, or the
    # divergence is stale rather than current.
    if len(highs) >= 2 and highs[-1] >= len(window) - 4:
        first, second = highs[-2], highs[-1]
        # Bound to locals so the None checks actually narrow the type; a subscript
        # expression cannot be narrowed by `is not None`.
        first_rsi, second_rsi = rsi[first], rsi[second]
        if (
            first_rsi is not None
            and second_rsi is not None
            and window[second].high > window[first].high
            and second_rsi < first_rsi - 3.0
        ):
            gap = first_rsi - second_rsi
            return _make(
                PatternKind.BEARISH_DIVERGENCE, candles, asset, timeframe,
                quality=min(1.0, 0.4 + gap * 0.03),
                detail=f"higher price high with RSI {gap:.0f} points lower",
                rsi_gap=gap,
            )
    if len(lows) >= 2 and lows[-1] >= len(window) - 4:
        first, second = lows[-2], lows[-1]
        first_rsi, second_rsi = rsi[first], rsi[second]
        if (
            first_rsi is not None
            and second_rsi is not None
            and window[second].low < window[first].low
            and second_rsi > first_rsi + 3.0
        ):
            gap = second_rsi - first_rsi
            return _make(
                PatternKind.BULLISH_DIVERGENCE, candles, asset, timeframe,
                quality=min(1.0, 0.4 + gap * 0.03),
                detail=f"lower price low with RSI {gap:.0f} points higher",
                rsi_gap=gap,
            )
    return None


def momentum_exhaustion(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """An extended run in one direction with momentum rolling over.

    Deliberately *not* "RSI above 70". Overbought markets keep rising for a long time;
    the additional requirement that momentum has already turned is what separates a
    strong trend from a fading one.
    """
    if len(candles) < 40:
        return None
    rsi = _rsi_series(candles[-40:])
    current, previous = rsi[-1], rsi[-4]
    if current is None or previous is None:
        return None

    if current >= 70 and current < previous - 4.0:
        return _make(
            PatternKind.MOMENTUM_EXHAUSTION_UP, candles, asset, timeframe,
            quality=min(1.0, 0.4 + (previous - current) * 0.03),
            detail=f"RSI rolled over from {previous:.0f} to {current:.0f} while overbought",
            rsi=current, rsi_change=current - previous,
        )
    if current <= 30 and current > previous + 4.0:
        return _make(
            PatternKind.MOMENTUM_EXHAUSTION_DOWN, candles, asset, timeframe,
            quality=min(1.0, 0.4 + (current - previous) * 0.03),
            detail=f"RSI turned up from {previous:.0f} to {current:.0f} while oversold",
            rsi=current, rsi_change=current - previous,
        )
    return None


# --------------------------------------------------------- trend and structure


def trend_continuation(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """A pullback inside an established trend that resumes in the trend's direction."""
    if len(candles) < 60:
        return None
    closes = [c.close for c in candles]
    fast = sum(closes[-20:]) / 20
    slow = sum(closes[-50:]) / 50
    bar = candles[-1]

    # Established trend: the averages are separated and price agrees with them.
    separation = (fast - slow) / slow * 100.0 if slow else 0.0
    if abs(separation) < 1.0:
        return None

    pulled_back = min(c.low for c in candles[-4:-1]) < fast if separation > 0 else (
        max(c.high for c in candles[-4:-1]) > fast
    )
    if not pulled_back:
        return None

    if separation > 0 and bar.close > fast and bar.close > candles[-2].close:
        return _make(
            PatternKind.TREND_CONTINUATION_UP, candles, asset, timeframe,
            quality=min(1.0, 0.4 + separation * 0.05),
            detail=f"pullback to the 20-bar mean resumed upward ({separation:+.1f}% MA spread)",
            ma_separation=separation,
        )
    if separation < 0 and bar.close < fast and bar.close < candles[-2].close:
        return _make(
            PatternKind.TREND_CONTINUATION_DOWN, candles, asset, timeframe,
            quality=min(1.0, 0.4 + abs(separation) * 0.05),
            detail=f"pullback to the 20-bar mean resumed downward ({separation:+.1f}% MA spread)",
            ma_separation=separation,
        )
    return None


def structure_break(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> Detection | None:
    """A confirmed swing structure flips from lower-highs to higher-highs, or vice versa."""
    if len(candles) < 80:
        return None
    window = candles[-80:]
    highs, lows = _swing_points(window)
    if len(highs) < 3 or len(lows) < 3:
        return None

    was_down = window[highs[-3]].high > window[highs[-2]].high
    now_up = window[highs[-1]].high > window[highs[-2]].high
    lows_rising = window[lows[-1]].low > window[lows[-2]].low

    if was_down and now_up and lows_rising and highs[-1] >= len(window) - 5:
        return _make(
            PatternKind.STRUCTURE_BREAK_UP, candles, asset, timeframe,
            quality=0.6,
            detail="swing structure flipped from lower highs to higher highs and lows",
        )
    was_up = window[highs[-3]].high < window[highs[-2]].high
    now_down = window[highs[-1]].high < window[highs[-2]].high
    lows_falling = window[lows[-1]].low < window[lows[-2]].low
    if was_up and now_down and lows_falling and highs[-1] >= len(window) - 5:
        return _make(
            PatternKind.STRUCTURE_BREAK_DOWN, candles, asset, timeframe,
            quality=0.6,
            detail="swing structure flipped from higher highs to lower highs and lows",
        )
    return None


#: Every detector, in a fixed order so results are reproducible.
DETECTORS: list[Detector] = [
    breakout,
    fakeout,
    liquidity_sweep,
    compression,
    expansion,
    volume_anomaly,
    accumulation_distribution,
    divergence,
    momentum_exhaustion,
    trend_continuation,
    structure_break,
]


def detect_all(
    candles: Sequence[Candle], asset: str, timeframe: Timeframe
) -> list[Detection]:
    """Run every detector against the final bar of the supplied window.

    The caller passes a slice ending at the bar under test, which is what makes
    look-ahead structurally impossible rather than merely discouraged.
    """
    if not candles or not candles[-1].is_final:
        return []
    detections: list[Detection] = []
    for detector in DETECTORS:
        try:
            found = detector(candles, asset, timeframe)
        except (ValueError, ZeroDivisionError, IndexError):
            # A detector must never take down a scan; a bad window is a skip.
            continue
        if found is not None and not math.isnan(found.close):
            detections.append(found)
    return detections
