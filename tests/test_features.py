"""Feature engine: warm-up, persistence, event wiring, and look-ahead safety.

The most important test in this file is :class:`TestNoLookAhead`. Every other
guarantee in the system rests on features at time *t* depending only on bars that had
closed by *t*; if that fails, every backtest result downstream is fiction and no
amount of careful modelling later recovers it.
"""

from __future__ import annotations

import math

import pytest
from tests.conftest import FIXED_NOW, make_candle
from tests.test_indicators import candles_from, price_path

from mie.config.settings import Settings
from mie.core.events import Event, InProcessEventBus, Topics
from mie.core.timeframes import Timeframe
from mie.features.engine import FEATURE_SET_VERSION, FeatureEngine, FeatureSet, build_indicators
from mie.features.levels import (
    StructureAnalyzer,
    cluster_levels,
    fibonacci_levels,
    find_swings,
    volume_profile,
)
from mie.storage.db import Database
from mie.storage.repositories import FeatureRepository, OHLCVRepository


@pytest.fixture
def engine(database: Database, settings: Settings) -> FeatureEngine:
    return FeatureEngine(database, settings)


def feature_set(timeframe: Timeframe = Timeframe.H1, structure: bool = True) -> FeatureSet:
    return FeatureSet(
        asset="BTC",
        timeframe=timeframe,
        source="fake",
        indicators=build_indicators(timeframe),
        structure=StructureAnalyzer() if structure else None,
    )


class TestFeatureSet:
    def test_produces_a_full_vector_once_warm(self) -> None:
        fs = feature_set()
        bars = candles_from(price_path(400))
        values: dict[str, float] = {}
        for bar in bars:
            values = fs.update(bar)

        assert fs.is_warm
        for key in ("sma_20", "rsi_14", "macd_12_26_9.macd", "atr_14.atr", "bb_20.percent_b"):
            assert key in values, key
        assert all(isinstance(v, float) for v in values.values())

    def test_composite_indicators_are_namespaced(self) -> None:
        """`macd.signal` must never collide with some other indicator's `signal`."""
        fs = feature_set()
        values = {}
        for bar in candles_from(price_path(400)):
            values = fs.update(bar)
        assert "macd_12_26_9.signal" in values
        assert "stoch_14.k" in values
        assert "signal" not in values

    def test_cold_set_reports_partial_values_without_claiming_warmth(self) -> None:
        fs = feature_set()
        for bar in candles_from(price_path(30)):
            values = fs.update(bar)
        assert not fs.is_warm, "200-period SMA cannot be ready after 30 bars"
        assert "sma_20" in values
        assert "sma_200" not in values

    def test_provisional_candles_are_refused(self) -> None:
        """A forming bar reaching a recursive indicator is unrecoverable corruption."""
        fs = feature_set(structure=False)
        with pytest.raises(ValueError, match="provisional"):
            fs.update(make_candle(FIXED_NOW, close=100.0, is_final=False))

    def test_out_of_order_candles_are_refused(self) -> None:
        """Indicator state cannot be rewound, so replaying an old bar must not be
        quietly folded in."""
        fs = feature_set(structure=False)
        fs.update(make_candle(FIXED_NOW + Timeframe.H1.delta, close=100.0))
        with pytest.raises(ValueError, match="out-of-order"):
            fs.update(make_candle(FIXED_NOW, close=101.0))

    def test_repeated_candle_is_refused(self) -> None:
        fs = feature_set(structure=False)
        bar = make_candle(FIXED_NOW, close=100.0)
        fs.update(bar)
        with pytest.raises(ValueError, match="out-of-order"):
            fs.update(bar)


class TestNoLookAhead:
    """Phase 2 gate: no feature at time t may depend on any bar closing after t."""

    def test_future_bars_cannot_change_a_past_feature_vector(self) -> None:
        """The definitive test.

        Two series share the first 250 bars and diverge violently afterwards. Every
        feature vector up to the divergence must be identical between them. If any
        indicator peeked forward — a centred window, a full-series normalisation, a
        look-ahead pivot — this catches it.
        """
        shared = price_path(250)
        calm = shared + [shared[-1] + 0.05 * i for i in range(60)]
        crash = shared + [shared[-1] * (0.5 ** (i / 10.0)) for i in range(60)]

        calm_vectors = _vectors(candles_from(calm))
        crash_vectors = _vectors(candles_from(crash))

        assert len(calm_vectors) >= 250
        for i in range(250):
            assert calm_vectors[i].keys() == crash_vectors[i].keys(), f"bar {i}"
            for key, value in calm_vectors[i].items():
                assert value == crash_vectors[i][key], f"bar {i} feature {key} leaked the future"

    def test_truncating_the_series_does_not_change_earlier_features(self) -> None:
        """Computing over 300 bars must give bar 200 the same values as stopping at 201."""
        bars = candles_from(price_path(300))
        full = _vectors(bars)
        truncated = _vectors(bars[:201])
        assert full[200] == truncated[200]

    def test_structure_only_confirms_pivots_with_completed_windows(self) -> None:
        """A pivot needs `right` bars after it; the newest bars cannot be pivots yet."""
        bars = candles_from(price_path(100))
        swings = find_swings(bars, left=2, right=2)
        assert swings
        assert max(s.index for s in swings) <= len(bars) - 3


class TestStructure:
    def test_finds_obvious_swings(self) -> None:
        prices = [10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 13, 12, 11, 12, 13]
        swings = find_swings(candles_from([float(p) for p in prices]), 2, 2)
        assert any(s.kind == "high" for s in swings)
        assert any(s.kind == "low" for s in swings)

    def test_plateaus_do_not_produce_duplicate_pivots(self) -> None:
        """A genuinely flat top is one level, not one 'pivot' per bar along it.

        Built directly rather than through `candles_from`, because that helper pads
        high/low proportionally and so would not produce identical highs.
        """
        highs = [10.0, 11.0, 12.0, 12.0, 12.0, 11.0, 10.0]
        bars = [
            make_candle(
                FIXED_NOW + Timeframe.H1.delta * i,
                close=high - 0.5,
                open_=high - 0.5,
                high=high,
                low=high - 1.0,
            )
            for i, high in enumerate(highs)
        ]
        swings = find_swings(bars, 2, 2)
        # Three bars share the maximum, so none of them is an unambiguous pivot.
        assert [s for s in swings if s.kind == "high"] == []

    def test_a_monotonic_line_has_no_pivots(self) -> None:
        """Every bar exceeds the last, so no bar is ever a local extreme. Reporting
        'range' here is honest: there is no confirmed structure to describe."""
        analyzer = StructureAnalyzer(lookback=200, refresh_every=1)
        result = None
        for bar in candles_from([100.0 + i for i in range(60)]):
            result = analyzer.update(bar)
        assert result is not None
        assert result.swings == []
        assert result.trend == "range"

    def test_clustering_merges_nearby_levels(self) -> None:
        """Price turns in a zone, not at an exact number."""
        bars = candles_from([100.0 + 5.0 * (i % 7 < 3) for i in range(120)])
        swings = find_swings(bars, 2, 2)
        levels = cluster_levels(swings, tolerance_pct=1.0, min_touches=2)
        assert levels
        assert all(lvl.touches >= 2 for lvl in levels)

    def test_single_touch_levels_are_excluded_by_default(self) -> None:
        swings = find_swings(candles_from(price_path(120)), 2, 2)
        assert all(lvl.touches >= 2 for lvl in cluster_levels(swings, min_touches=2))

    def test_volume_profile_finds_the_busiest_price(self) -> None:
        bars = list(candles_from([100.0] * 40))
        bars += list(candles_from([130.0] * 5))
        profile = volume_profile(bars, bins=20)
        assert profile is not None
        assert profile["value_area_low"] <= profile["poc"] <= profile["value_area_high"]
        assert profile["profile_low"] <= profile["poc"] <= profile["profile_high"]

    def test_volume_profile_needs_a_price_range(self) -> None:
        assert volume_profile([], bins=20) is None

    def test_fibonacci_direction_matters(self) -> None:
        """Measured from the wrong end the numbers look plausible and mean nothing."""
        up = fibonacci_levels(200.0, 100.0, uptrend=True)
        down = fibonacci_levels(200.0, 100.0, uptrend=False)
        assert up["fib_0.618"] == pytest.approx(200.0 - 100.0 * 0.618)
        assert down["fib_0.618"] == pytest.approx(100.0 + 100.0 * 0.618)
        assert up["fib_0.5"] == down["fib_0.5"], "the midpoint is direction-agnostic"

    def test_rising_zigzag_is_an_uptrend(self) -> None:
        """A real uptrend pulls back; the pullbacks are what create the pivots that
        make higher highs and higher lows observable at all."""
        prices = [100.0 + i * 0.6 + 4.0 * math.sin(i / 3.0) for i in range(150)]
        analyzer = StructureAnalyzer(lookback=200, refresh_every=1)
        result = None
        for bar in candles_from(prices):
            result = analyzer.update(bar)
        assert result is not None
        assert result.trend == "up"

    def test_falling_zigzag_is_a_downtrend(self) -> None:
        prices = [200.0 - i * 0.6 + 4.0 * math.sin(i / 3.0) for i in range(150)]
        analyzer = StructureAnalyzer(lookback=200, refresh_every=1)
        result = None
        for bar in candles_from(prices):
            result = analyzer.update(bar)
        assert result is not None
        assert result.trend == "down"

    def test_broadening_formation_is_not_called_a_trend(self) -> None:
        """Higher highs *with* lower lows is a broadening formation. Calling it an
        uptrend would be wrong in the most expensive direction, so both sides must
        agree before a trend is declared."""
        from mie.features.levels import Swing, _classify_trend

        now = FIXED_NOW
        highs = [Swing("high", 100.0, now, 0), Swing("high", 110.0, now, 4)]
        lows = [Swing("low", 90.0, now, 2), Swing("low", 80.0, now, 6)]
        assert _classify_trend(highs, lows) == "range"

    def test_structure_reports_surrounding_levels(self) -> None:
        analyzer = StructureAnalyzer(lookback=200, refresh_every=1)
        result = None
        for bar in candles_from(price_path(200)):
            result = analyzer.update(bar)
        assert result is not None
        features = result.as_features(price=result.swings[-1].price if result.swings else 100.0)
        assert "structure_trend" in features
        assert "fib_0.618" in features


class TestPersistence:
    async def test_backfill_writes_one_vector_per_warm_bar(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        bars = candles_from(price_path(400))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(bars)

        written = await engine.backfill("BTC", Timeframe.H1, "fake")
        assert written > 100

        async with database.session() as session:
            stored = await FeatureRepository(session).count("BTC", Timeframe.H1)
        assert stored == written

    async def test_stored_vector_round_trips(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        bars = candles_from(price_path(400))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(bars)
        await engine.backfill("BTC", Timeframe.H1, "fake")

        async with database.session() as session:
            latest = await FeatureRepository(session).latest("BTC", Timeframe.H1)
        assert latest is not None
        assert latest.version == FEATURE_SET_VERSION
        assert latest.open_time == bars[-1].open_time
        assert latest.payload["rsi_14"] == pytest.approx(
            _vectors(bars)[-1]["rsi_14"], abs=1e-9
        )

    async def test_recomputation_is_idempotent(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        bars = candles_from(price_path(300))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(bars)

        first = await engine.backfill("BTC", Timeframe.H1, "fake")
        second = await engine.backfill("BTC", Timeframe.H1, "fake")
        async with database.session() as session:
            stored = await FeatureRepository(session).count("BTC", Timeframe.H1)
        assert first == second == stored

    async def test_features_from_two_sources_stay_separate(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        """Two venues are two series; merging them would corrupt every recursive
        indicator during a failover."""
        binance = candles_from(price_path(300))
        kraken = [c.model_copy(update={"source": "kraken"}) for c in binance]
        async with database.session() as session:
            repo = OHLCVRepository(session)
            await repo.upsert_candles(binance)
            await repo.upsert_candles(kraken)

        await engine.backfill("BTC", Timeframe.H1, "fake")
        await engine.backfill("BTC", Timeframe.H1, "kraken")

        async with database.session() as session:
            repo = FeatureRepository(session)
            assert await repo.count("BTC", Timeframe.H1, source="fake") > 0
            assert await repo.count("BTC", Timeframe.H1, source="kraken") > 0
            assert await repo.count("BTC", Timeframe.H1) == (
                await repo.count("BTC", Timeframe.H1, source="fake")
                + await repo.count("BTC", Timeframe.H1, source="kraken")
            )


class TestWarmup:
    async def test_warmup_reaches_the_same_state_as_running_continuously(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        """Restarting must not change the numbers, or every restart is a discontinuity
        in the feature history."""
        bars = candles_from(price_path(400))
        async with database.session() as session:
            await OHLCVRepository(session).upsert_candles(bars[:-1])

        primed = await engine.warmup("BTC", Timeframe.H1, "fake", extra_bars=200)
        assert primed is not None and primed.is_warm

        resumed = await engine.handle(bars[-1], persist=False)
        continuous = _vectors(bars)[-1]

        assert resumed is not None
        for key, value in continuous.items():
            assert resumed[key] == pytest.approx(value, abs=1e-9), key

    async def test_warmup_without_history_is_not_an_error(
        self, engine: FeatureEngine
    ) -> None:
        assert await engine.warmup("BTC", Timeframe.H1, "fake") is None


class TestEventWiring:
    async def test_engine_consumes_candle_closed_events(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        bus = InProcessEventBus()
        engine.bus = bus
        engine.subscribe()

        for bar in candles_from(price_path(400)):
            await bus.publish(Event(topic=Topics.CANDLE_CLOSED, payload=bar))

        assert engine.processed > 100
        async with database.session() as session:
            assert await FeatureRepository(session).count("BTC", Timeframe.H1) > 100

    async def test_batched_payloads_are_handled(
        self, engine: FeatureEngine, database: Database
    ) -> None:
        """Backfill publishes a list of candles; live publishes one at a time."""
        bus = InProcessEventBus()
        engine.bus = bus
        engine.subscribe()
        await bus.publish(
            Event(topic=Topics.CANDLE_CLOSED, payload=candles_from(price_path(400)))
        )
        assert engine.processed > 100

    async def test_provisional_candles_are_ignored_not_fatal(
        self, engine: FeatureEngine
    ) -> None:
        result = await engine.handle(
            make_candle(FIXED_NOW, close=100.0, is_final=False), persist=False
        )
        assert result is None
        assert engine.skipped == 1

    async def test_duplicate_delivery_is_skipped_quietly(
        self, engine: FeatureEngine
    ) -> None:
        """Overlapping polls re-deliver bars constantly; that is normal, not an error."""
        bar = make_candle(FIXED_NOW, close=100.0)
        assert await engine.handle(bar, persist=False) is not None
        assert await engine.handle(bar, persist=False) is None
        assert engine.skipped == 1


def _vectors(bars) -> list[dict[str, float]]:
    """Feature vectors for every bar, computed the same way the engine does."""
    fs = feature_set()
    return [fs.update(bar) for bar in bars]
