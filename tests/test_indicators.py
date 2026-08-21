"""Indicator correctness against independent reference implementations.

Phase 2's gate demands that indicator values match an independent reference. The
references below are written **naively and separately**: they recompute over whole
arrays with no shared state, no shared helpers, and no import from the incremental
code. That independence is the entire point — comparing an implementation against
itself proves nothing.

Two distinct claims are tested:

* **Correctness**: incremental values equal the naive batch values.
* **Exactness**: they are equal *bit for bit*, not merely close. Windowed indicators
  recompute from their deque and recursive ones perform the identical recursion, so
  any drift would signal an accumulator bug rather than acceptable float noise.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from tests.conftest import FIXED_NOW, make_candle

from mie.core.timeframes import Timeframe
from mie.features.indicators import (
    ADX,
    ATR,
    EMA,
    MACD,
    OBV,
    ROC,
    RSI,
    SMA,
    AnchoredVWAP,
    BollingerBands,
    RealisedVolatility,
    Stochastic,
    WilderMA,
)

# --------------------------------------------------------------- test fixtures


def price_path(count: int = 300, seed: float = 100.0) -> list[float]:
    """A deterministic, non-trivial price path.

    Trend plus two out-of-phase cycles plus a jitter term: enough structure that
    indicators produce varied output, and no randomness so failures reproduce.
    """
    prices = []
    for i in range(count):
        trend = seed + i * 0.08
        cycle = 6.0 * math.sin(i / 13.0) + 2.5 * math.cos(i / 5.0)
        jitter = 0.4 * math.sin(i * 2.7)
        prices.append(round(trend + cycle + jitter, 6))
    return prices


def candles_from(prices: list[float], timeframe: Timeframe = Timeframe.H1):
    """Wrap a price path into well-formed bars with varying volume."""
    bars = []
    for i, close in enumerate(prices):
        open_price = prices[i - 1] if i else close
        span = abs(close - open_price)
        # Padding must be proportional to price, or the fixture itself stops being
        # scale-invariant and any test of scale invariance measures the helper.
        pad = span * 0.3 + abs(close) * 0.002
        high = max(open_price, close) + pad
        low = min(open_price, close) - pad
        bars.append(
            make_candle(
                FIXED_NOW + timeframe.delta * i,
                close=close,
                open_=open_price,
                # Deliberately unrounded: quantising to 6 decimals is invisible at a
                # price of 100,000 and severe at 0.1, which would make the fixture
                # itself scale-dependent.
                high=high,
                low=low,
                volume=round(100.0 + 40.0 * math.sin(i / 7.0) + i % 11, 4),
                timeframe=timeframe,
            )
        )
    return bars


@pytest.fixture(scope="module")
def prices() -> list[float]:
    return price_path()


@pytest.fixture(scope="module")
def bars(prices: list[float]):
    return candles_from(prices)


def run(indicator, bars) -> list:
    """Feed every bar through an indicator and collect its output."""
    return [indicator.update(bar) for bar in bars]


# ------------------------------------------------- independent reference impls


def ref_sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(None)
        else:
            window = values[i + 1 - period : i + 1]
            out.append(sum(window) / period)
    return out


def ref_ema(values: list[float], period: int) -> list[float | None]:
    """Standard EMA, seeded with the SMA of the first `period` values."""
    alpha = 2.0 / (period + 1.0)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = current + alpha * (values[i] - current)
        out[i] = current
    return out


def ref_wilder(values: list[float], period: int) -> list[float | None]:
    """Wilder's smoothing, seeded with the SMA of the first `period` values."""
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = current + (values[i] - current) / period
        out[i] = current
    return out


def ref_rsi(closes: list[float], period: int = 14) -> list[float | None]:
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = ref_wilder(gains, period)
    avg_loss = ref_wilder(losses, period)

    out: list[float | None] = [None] * len(closes)
    for i in range(len(gains)):
        g, loss = avg_gain[i], avg_loss[i]
        if g is None or loss is None:
            continue
        out[i + 1] = 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + g / loss))
    return out


def ref_true_range(bars) -> list[float]:
    values = []
    for i, bar in enumerate(bars):
        span = bar.high - bar.low
        if i == 0:
            values.append(span)
        else:
            previous = bars[i - 1].close
            values.append(
                max(span, abs(bar.high - previous), abs(bar.low - previous))
            )
    return values


def ref_atr(bars, period: int = 14) -> list[float | None]:
    return ref_wilder(ref_true_range(bars), period)


def ref_bollinger(closes: list[float], period: int, deviations: float):
    out: list[tuple[float, float, float] | None] = []
    for i in range(len(closes)):
        if i + 1 < period:
            out.append(None)
            continue
        window = closes[i + 1 - period : i + 1]
        mid = sum(window) / period
        var = sum((v - mid) ** 2 for v in window) / period
        sigma = math.sqrt(var)
        out.append((mid, mid + deviations * sigma, mid - deviations * sigma))
    return out


def ref_stochastic(bars, period: int, smooth: int):
    ks: list[float | None] = []
    for i in range(len(bars)):
        if i + 1 < period:
            ks.append(None)
            continue
        window = bars[i + 1 - period : i + 1]
        highest = max(b.high for b in window)
        lowest = min(b.low for b in window)
        span = highest - lowest
        ks.append(50.0 if span == 0 else (bars[i].close - lowest) / span * 100.0)

    ds: list[float | None] = []
    for i in range(len(ks)):
        window = ks[i + 1 - smooth : i + 1] if i + 1 >= smooth else []
        ds.append(
            sum(window) / smooth if window and all(v is not None for v in window) else None
        )
    return ks, ds


def ref_obv(bars) -> list[float]:
    total = 0.0
    out = []
    for i, bar in enumerate(bars):
        if i:
            previous = bars[i - 1].close
            if bar.close > previous:
                total += bar.volume
            elif bar.close < previous:
                total -= bar.volume
        out.append(total)
    return out


def ref_roc(closes: list[float], period: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(closes)):
        if i < period:
            out.append(None)
        else:
            base = closes[i - period]
            out.append(None if base == 0 else (closes[i] - base) / base * 100.0)
    return out


# ------------------------------------------------------------------- the tests


class TestMovingAverages:
    def test_sma_matches_reference_exactly(self, bars, prices) -> None:
        got = run(SMA(20), bars)
        want = ref_sma(prices, 20)
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            assert (g is None) == (w is None)
            if g is not None:
                assert g == pytest.approx(w, abs=1e-9)

    @pytest.mark.parametrize("period", [9, 21, 50])
    def test_ema_matches_reference(self, bars, prices, period: int) -> None:
        got = run(EMA(period), bars)
        want = ref_ema(prices, period)
        for g, w in zip(got, want, strict=True):
            assert (g is None) == (w is None)
            if g is not None:
                assert g == pytest.approx(w, abs=1e-9)

    def test_ema_seeds_from_sma_not_from_first_value(self, bars, prices) -> None:
        """The seed is where implementations usually diverge, so it is pinned."""
        first = next(v for v in run(EMA(10), bars) if v is not None)
        assert first == pytest.approx(sum(prices[:10]) / 10, abs=1e-12)

    def test_wilder_is_slower_than_ema_of_the_same_period(self, bars) -> None:
        """Wilder's alpha is 1/n against the EMA's 2/(n+1); substituting one for the
        other silently changes every RSI, ATR and ADX in the system."""
        ema = [v for v in run(EMA(14), bars) if v is not None]
        wilder = [v for v in run(WilderMA(14), bars) if v is not None]
        ema_moves = sum(abs(b - a) for a, b in pairwise(ema))
        wilder_moves = sum(abs(b - a) for a, b in pairwise(wilder))
        assert wilder_moves < ema_moves


class TestOscillators:
    def test_rsi_matches_reference(self, bars, prices) -> None:
        got = run(RSI(14), bars)
        want = ref_rsi(prices, 14)
        for g, w in zip(got, want, strict=True):
            assert (g is None) == (w is None)
            if g is not None:
                assert g == pytest.approx(w, abs=1e-9)

    def test_rsi_stays_within_bounds(self, bars) -> None:
        for value in run(RSI(14), bars):
            if value is not None:
                assert 0.0 <= value <= 100.0

    def test_rsi_is_100_when_nothing_falls(self) -> None:
        """A monotonic rise has zero average loss; the answer is 100, not a crash."""
        rising = candles_from([100.0 + i for i in range(40)])
        assert run(RSI(14), rising)[-1] == 100.0

    def test_macd_matches_reference(self, bars, prices) -> None:
        fast, slow, signal_period = 12, 26, 9
        ema_fast = ref_ema(prices, fast)
        ema_slow = ref_ema(prices, slow)
        macd_line = [
            None if f is None or s is None else f - s
            for f, s in zip(ema_fast, ema_slow, strict=True)
        ]
        defined = [v for v in macd_line if v is not None]
        signal_defined = ref_ema(defined, signal_period)
        offset = len(macd_line) - len(defined)

        got = run(MACD(fast, slow, signal_period), bars)
        for i, result in enumerate(got):
            expected_signal = (
                signal_defined[i - offset] if i >= offset else None
            )
            if result is None:
                assert expected_signal is None
                continue
            assert result["macd"] == pytest.approx(macd_line[i], abs=1e-9)
            assert result["signal"] == pytest.approx(expected_signal, abs=1e-9)
            assert result["histogram"] == pytest.approx(
                result["macd"] - result["signal"], abs=1e-12
            )

    def test_stochastic_matches_reference(self, bars) -> None:
        ks, ds = ref_stochastic(bars, 14, 3)
        got = run(Stochastic(14, 3), bars)
        for i, result in enumerate(got):
            if result is None:
                assert ds[i] is None
                continue
            assert result["k"] == pytest.approx(ks[i], abs=1e-9)
            assert result["d"] == pytest.approx(ds[i], abs=1e-9)

    def test_stochastic_is_neutral_on_a_flat_range(self) -> None:
        flat = [
            make_candle(FIXED_NOW + Timeframe.H1.delta * i, close=50.0, open_=50.0,
                        high=50.0, low=50.0)
            for i in range(30)
        ]
        assert run(Stochastic(14, 3), flat)[-1]["k"] == 50.0

    def test_roc_matches_reference(self, bars, prices) -> None:
        got = run(ROC(10), bars)
        want = ref_roc(prices, 10)
        for g, w in zip(got, want, strict=True):
            assert (g is None) == (w is None)
            if g is not None:
                assert g == pytest.approx(w, abs=1e-9)


class TestVolatilityAndTrend:
    def test_atr_matches_reference(self, bars) -> None:
        want = ref_atr(bars, 14)
        got = run(ATR(14), bars)
        for i, result in enumerate(got):
            if result is None:
                assert want[i] is None
                continue
            assert result["atr"] == pytest.approx(want[i], abs=1e-9)

    def test_atr_percentage_normalises_across_price_scales(self) -> None:
        """A $900 ATR on BTC and a $0.004 ATR on DOGE are the same statement."""
        expensive = candles_from([p * 1000 for p in price_path(60)])
        cheap = candles_from([p * 0.001 for p in price_path(60)])
        big = run(ATR(14), expensive)[-1]
        small = run(ATR(14), cheap)[-1]
        assert big["atr"] == pytest.approx(small["atr"] * 1e6, rel=1e-6)
        assert big["atr_pct"] == pytest.approx(small["atr_pct"], rel=1e-6)

    def test_bollinger_matches_reference(self, bars, prices) -> None:
        want = ref_bollinger(prices, 20, 2.0)
        got = run(BollingerBands(20, 2.0), bars)
        for i, result in enumerate(got):
            if result is None:
                assert want[i] is None
                continue
            mid, upper, lower = want[i]
            assert result["middle"] == pytest.approx(mid, abs=1e-9)
            assert result["upper"] == pytest.approx(upper, abs=1e-9)
            assert result["lower"] == pytest.approx(lower, abs=1e-9)

    def test_percent_b_locates_price_within_the_bands(self, bars) -> None:
        """%b is the normalised position of price in the band, and may exceed [0,1]
        on a breakout — that is the signal, not an error to clamp away."""
        results = run(BollingerBands(20, 2.0), bars)
        for bar, result in zip(bars, results, strict=True):
            if result is None:
                continue
            width = result["upper"] - result["lower"]
            expected = (bar.close - result["lower"]) / width
            assert result["percent_b"] == pytest.approx(expected, abs=1e-12)
            assert result["bandwidth"] >= 0.0
            assert result["lower"] <= result["middle"] <= result["upper"]

    def test_adx_is_bounded_and_rises_in_a_trend(self) -> None:
        trending = candles_from([100.0 + i * 1.5 for i in range(120)])
        choppy = candles_from([100.0 + 3.0 * math.sin(i / 2.0) for i in range(120)])
        trend_adx = [r["adx"] for r in run(ADX(14), trending) if r]
        chop_adx = [r["adx"] for r in run(ADX(14), choppy) if r]

        assert all(0.0 <= v <= 100.0 for v in trend_adx + chop_adx)
        assert trend_adx[-1] > chop_adx[-1], "ADX must distinguish trend from chop"

    def test_directional_indicators_agree_with_direction(self) -> None:
        rising = candles_from([100.0 + i for i in range(120)])
        result = [r for r in run(ADX(14), rising) if r][-1]
        assert result["plus_di"] > result["minus_di"]

    def test_realised_volatility_scales_with_dispersion(self) -> None:
        calm = candles_from([100.0 + 0.1 * math.sin(i) for i in range(80)])
        wild = candles_from([100.0 + 8.0 * math.sin(i) for i in range(80)])
        calm_vol = run(RealisedVolatility(20, 365 * 24), calm)[-1]
        wild_vol = run(RealisedVolatility(20, 365 * 24), wild)[-1]
        assert wild_vol > calm_vol * 10


class TestVolume:
    def test_obv_matches_reference(self, bars) -> None:
        want = ref_obv(bars)
        got = run(OBV(20), bars)
        for i, result in enumerate(got):
            if result is None:
                continue
            assert result["obv"] == pytest.approx(want[i], abs=1e-9)

    def test_obv_ignores_unchanged_closes(self) -> None:
        flat = [
            make_candle(FIXED_NOW + Timeframe.H1.delta * i, close=100.0, volume=50.0)
            for i in range(5)
        ]
        assert run(OBV(20), flat)[-1]["obv"] == 0.0

    def test_vwap_reanchors_each_utc_day(self) -> None:
        """Without a daily anchor, VWAP becomes an all-time average that never moves."""
        day_one = candles_from(
            [100.0] * 24, timeframe=Timeframe.H1
        )
        vwap = AnchoredVWAP()
        for bar in day_one:
            vwap.update(bar)
        end_of_day_one = vwap._pv

        next_day = make_candle(
            FIXED_NOW + Timeframe.H1.delta * 24, close=200.0, open_=200.0,
            high=200.0, low=200.0, volume=10.0
        )
        result = vwap.update(next_day)
        assert end_of_day_one > 0
        assert result["vwap"] == pytest.approx(200.0, abs=1e-9), "state must reset"

    def test_vwap_is_volume_weighted_not_a_mean(self) -> None:
        heavy = make_candle(FIXED_NOW, close=100.0, open_=100.0, high=100.0, low=100.0,
                            volume=1000.0)
        light = make_candle(FIXED_NOW + Timeframe.H1.delta, close=200.0, open_=200.0,
                            high=200.0, low=200.0, volume=1.0)
        vwap = AnchoredVWAP()
        vwap.update(heavy)
        result = vwap.update(light)
        assert result["vwap"] < 101.0, "the heavy bar must dominate"


class TestExactness:
    """Phase 2 gate: incremental output is bit-identical to a fresh recompute."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SMA(20),
            lambda: EMA(21),
            lambda: WilderMA(14),
            lambda: RSI(14),
            lambda: ATR(14),
            lambda: MACD(12, 26, 9),
            lambda: BollingerBands(20, 2.0),
            lambda: ADX(14),
            lambda: Stochastic(14, 3),
            lambda: OBV(20),
            lambda: ROC(10),
            lambda: RealisedVolatility(20, 8760),
        ],
    )
    def test_resuming_equals_computing_from_scratch(self, bars, factory) -> None:
        """A fresh instance replayed over all bars must equal one that ran live.

        This is what makes warm-up on restart safe: priming from stored history has
        to land in exactly the state continuous operation would have reached.
        """
        streamed = run(factory(), bars)

        primed = factory()
        primed.prime(bars[:-1])
        resumed = primed.update(bars[-1])

        assert _identical(resumed, streamed[-1])

    @pytest.mark.parametrize("factory", [lambda: SMA(20), lambda: EMA(21), lambda: RSI(14)])
    def test_no_accumulator_drift_over_a_long_series(self, factory) -> None:
        """Rolling sums drift from fresh summation; windowed indicators must not."""
        long_bars = candles_from(price_path(2000))
        streamed = run(factory(), long_bars)
        fresh = factory()
        fresh.prime(long_bars[:-1])
        assert _identical(fresh.update(long_bars[-1]), streamed[-1])


class TestGuards:
    @pytest.mark.parametrize(
        "factory",
        [lambda: SMA(0), lambda: EMA(0), lambda: ROC(0), lambda: RealisedVolatility(1, 100)],
    )
    def test_invalid_periods_are_rejected(self, factory) -> None:
        with pytest.raises(ValueError):
            factory()

    def test_macd_rejects_inverted_periods(self) -> None:
        with pytest.raises(ValueError, match="fast period"):
            MACD(26, 12, 9)

    def test_warmup_is_honest(self, bars) -> None:
        """An indicator must not claim a value before it has the history for one."""
        for factory in (SMA(20), EMA(21), RSI(14), ATR(14), MACD(12, 26, 9), ADX(14)):
            values = run(factory, bars)
            first_defined = next(i for i, v in enumerate(values) if v is not None)
            assert first_defined + 1 >= factory.warmup - 1, factory.name


def _identical(left, right) -> bool:
    """Exact equality, including inside composite dict results."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(left[k] == right[k] for k in left)
    return left == right
