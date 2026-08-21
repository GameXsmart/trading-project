"""Multi-timeframe market state.

Phase 3's gate has two halves, and both are tested here:

1. hand-constructed historical scenarios (clear uptrend, chop, capitulation,
   recovery) are classified correctly;
2. a bullish daily with a bearish 15m yields "pullback within uptrend" — **never** a
   flat neutral average.

The second is the one that justifies the whole module. Averaging is the obvious
implementation and it destroys precisely the reading that matters most.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW
from tests.test_indicators import candles_from

from mie.config.settings import Settings
from mie.core.timeframes import Timeframe
from mie.features.engine import FeatureEngine, FeatureSet, build_indicators
from mie.features.levels import StructureAnalyzer
from mie.state.classifier import TimeframeClassifier
from mie.state.engine import StateEngine
from mie.state.hierarchy import HierarchyAnalyzer, split_hierarchy
from mie.state.types import Alignment, Direction, Regime, TimeframeState
from mie.storage.db import Database
from mie.storage.repositories import MarketStateRepository, OHLCVRepository

# ------------------------------------------------------------------- scenarios


def scenario(kind: str, count: int = 320) -> list[float]:
    """Deterministic price paths for hand-labelled market conditions."""
    if kind == "uptrend":
        # Steady advance with shallow pullbacks — a textbook trend.
        return [100.0 + i * 0.55 + 3.0 * math.sin(i / 9.0) for i in range(count)]
    if kind == "downtrend":
        return [300.0 - i * 0.55 + 3.0 * math.sin(i / 9.0) for i in range(count)]
    if kind == "chop":
        # No net progress, only oscillation.
        return [100.0 + 4.0 * math.sin(i / 6.0) + 1.5 * math.cos(i / 2.5) for i in range(count)]
    if kind == "capitulation":
        # A long grind lower ending in an accelerating, high-volatility flush.
        base = [200.0 - i * 0.3 for i in range(count - 60)]
        crash = [base[-1] * (0.97**j) * (1 + 0.02 * math.sin(j)) for j in range(60)]
        return base + crash
    if kind == "recovery":
        # A violent decline that has bottomed and is snapping back hard.
        fall = [200.0 * (0.985**i) for i in range(count - 80)]
        rise = [fall[-1] * (1.02**j) * (1 + 0.015 * math.sin(j)) for j in range(80)]
        return fall + rise
    raise ValueError(f"unknown scenario {kind}")


def features_for(prices: list[float], timeframe: Timeframe = Timeframe.H1) -> dict[str, float]:
    """Feature vector at the end of a price path."""
    fs = FeatureSet(
        asset="BTC",
        timeframe=timeframe,
        source="fake",
        indicators=build_indicators(timeframe),
        structure=StructureAnalyzer(),
    )
    values: dict[str, float] = {}
    for bar in candles_from(prices, timeframe=timeframe):
        values = fs.update(bar)
    return values


def classify(kind: str, timeframe: Timeframe = Timeframe.H1) -> TimeframeState:
    return TimeframeClassifier().classify(
        asset="BTC",
        timeframe=timeframe,
        features=features_for(scenario(kind), timeframe),
        as_of=FIXED_NOW,
    )


def state(
    timeframe: Timeframe,
    direction: Direction,
    strength: float = 0.7,
    confidence: float = 0.8,
    volatility: float | None = 50.0,
) -> TimeframeState:
    """A hand-built timeframe state, for exercising the hierarchy directly."""
    return TimeframeState(
        asset="BTC",
        timeframe=timeframe,
        as_of=FIXED_NOW,
        direction=direction,
        strength=strength,
        confidence=confidence,
        score=direction.score,
        volatility_pct=volatility,
        close=100.0,
    )


# ------------------------------------------------------------------ unit tests


class TestDirection:
    def test_scores_map_symmetrically(self) -> None:
        assert Direction.STRONG_UP.score == -Direction.STRONG_DOWN.score
        assert Direction.NEUTRAL.score == 0.0

    def test_from_score_has_a_wide_neutral_band(self) -> None:
        """Markets are usually going nowhere; a classifier that refuses to say so is
        a random number generator with opinions."""
        assert Direction.from_score(0.05) is Direction.NEUTRAL
        assert Direction.from_score(-0.05) is Direction.NEUTRAL
        assert Direction.from_score(0.9) is Direction.STRONG_UP
        assert Direction.from_score(-0.9) is Direction.STRONG_DOWN

    def test_sign_ignores_magnitude(self) -> None:
        assert Direction.WEAK_UP.sign == Direction.STRONG_UP.sign == 1
        assert Direction.NEUTRAL.sign == 0


class TestClassifierScenarios:
    """Phase 3 gate, part 1: hand-labelled scenarios classify correctly."""

    def test_clear_uptrend_reads_bullish(self) -> None:
        result = classify("uptrend")
        assert result.direction.is_bullish, result.summary()
        assert result.confidence > 0.4
        assert result.evidence, "a classification must be able to explain itself"

    def test_clear_downtrend_reads_bearish(self) -> None:
        result = classify("downtrend")
        assert result.direction.is_bearish, result.summary()
        assert result.confidence > 0.4

    def test_chop_is_not_given_a_confident_direction(self) -> None:
        """The important failure mode: inventing a trend in a market that has none."""
        result = classify("chop")
        assert abs(result.score) < 0.35, result.summary()
        assert result.strength < 0.6

    def test_capitulation_is_strongly_bearish_and_volatile(self) -> None:
        result = classify("capitulation")
        assert result.direction.is_bearish, result.summary()
        assert result.volatility_pct is not None and result.volatility_pct > 100

    def test_recovery_is_bullish_after_a_crash(self) -> None:
        result = classify("recovery")
        assert result.direction.is_bullish, result.summary()

    def test_evidence_and_counter_evidence_are_separated(self) -> None:
        """Requirement §16: every reading shows what argues against it too."""
        result = classify("uptrend")
        assert all(e.contribution > 0 for e in result.evidence)
        assert all(e.contribution < 0 for e in result.counter_evidence)


class TestClassifierConfidence:
    def test_missing_features_lower_confidence(self) -> None:
        """Absent data must reduce confidence, not silently vote neutral."""
        full = TimeframeClassifier().classify(
            "BTC", Timeframe.H1, features_for(scenario("uptrend")), FIXED_NOW
        )
        sparse = TimeframeClassifier().classify(
            "BTC", Timeframe.H1, {"close": 100.0, "rsi_14": 70.0}, FIXED_NOW
        )
        assert sparse.confidence < full.confidence

    def test_no_features_produces_no_opinion(self) -> None:
        result = TimeframeClassifier().classify("BTC", Timeframe.H1, {}, FIXED_NOW)
        assert result.direction is Direction.NEUTRAL
        assert result.confidence == 0.0
        assert not result.is_usable

    def test_degraded_data_quality_reduces_confidence(self) -> None:
        """The Phase 1 trust score reaching an actual output — requirement §20."""
        features = features_for(scenario("uptrend"))
        clean = TimeframeClassifier().classify(
            "BTC", Timeframe.H1, features, FIXED_NOW, data_quality=1.0
        )
        degraded = TimeframeClassifier().classify(
            "BTC", Timeframe.H1, features, FIXED_NOW, data_quality=0.3
        )
        assert degraded.direction is clean.direction, "quality must not change the reading"
        assert degraded.confidence == pytest.approx(clean.confidence * 0.3, abs=1e-4)

    def test_contradictory_signals_lower_confidence(self) -> None:
        """MACD up and RSI down should read 'unclear', not 'confidently neutral'."""
        coherent = TimeframeClassifier().classify(
            "BTC",
            Timeframe.H1,
            {"close": 110.0, "ema_21": 100.0, "sma_50": 98.0, "sma_200": 95.0,
             "rsi_14": 65.0, "roc_10": 4.0, "structure_trend": 1.0},
            FIXED_NOW,
        )
        mixed = TimeframeClassifier().classify(
            "BTC",
            Timeframe.H1,
            {"close": 110.0, "ema_21": 100.0, "sma_50": 98.0, "sma_200": 95.0,
             "rsi_14": 30.0, "roc_10": -4.0, "structure_trend": -1.0},
            FIXED_NOW,
        )
        assert mixed.confidence < coherent.confidence


class TestHierarchy:
    """Phase 3 gate, part 2: conflict is information, not noise."""

    def test_bullish_daily_with_bearish_15m_is_a_pullback(self) -> None:
        """The headline case. Averaging these would produce 'neutral', which is the
        one description that fits neither timeframe."""
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.UP, strength=0.75, confidence=0.85),
                state(Timeframe.H4, Direction.UP, strength=0.7, confidence=0.8),
                state(Timeframe.H1, Direction.WEAK_UP, strength=0.5, confidence=0.6),
                state(Timeframe.M15, Direction.DOWN, strength=0.6, confidence=0.7),
            ],
        )
        assert result.alignment is Alignment.PULLBACK_IN_UPTREND
        assert result.bias.is_bullish, "the daily trend must still dominate the bias"
        assert result.bias is not Direction.NEUTRAL, "must never average to neutral"
        assert "pullback" in result.interpretation.lower()
        assert result.conflicts, "the disagreement must be surfaced, not smoothed over"

    def test_bearish_daily_with_bullish_15m_is_a_counter_trend_rally(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.DOWN, strength=0.75, confidence=0.85),
                state(Timeframe.H4, Direction.DOWN, strength=0.7, confidence=0.8),
                state(Timeframe.M15, Direction.UP, strength=0.6, confidence=0.7),
            ],
        )
        assert result.alignment is Alignment.RALLY_IN_DOWNTREND
        assert result.bias.is_bearish
        assert "counter-trend" in result.interpretation.lower()

    def test_a_fading_higher_trend_reads_as_a_possible_reversal(self) -> None:
        """A strong trend absorbs a counter-move; a weak one may not, and the
        difference is what separates a dip from a turn."""
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.WEAK_UP, strength=0.2, confidence=0.4),
                state(Timeframe.H4, Direction.WEAK_UP, strength=0.25, confidence=0.4),
                state(Timeframe.M15, Direction.DOWN, strength=0.7, confidence=0.8),
            ],
        )
        assert result.alignment is Alignment.POSSIBLE_REVERSAL
        assert result.alignment.is_conflicted
        assert "unconfirmed" in result.interpretation.lower()

    def test_full_agreement_is_reported_as_aligned(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.STRONG_UP),
                state(Timeframe.H4, Direction.UP),
                state(Timeframe.H1, Direction.UP),
                state(Timeframe.M15, Direction.UP),
            ],
        )
        assert result.alignment is Alignment.ALIGNED_BULLISH
        assert result.agreement == 1.0
        assert result.is_actionable

    def test_all_neutral_is_rangebound_not_conflicted(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [state(tf, Direction.NEUTRAL) for tf in (Timeframe.D1, Timeframe.H4, Timeframe.M15)],
        )
        assert result.alignment is Alignment.RANGEBOUND
        assert result.bias is Direction.NEUTRAL

    def test_higher_timeframes_carry_more_weight(self) -> None:
        """A daily trend is not repealed by a fifteen-minute wobble."""
        daily_led = HierarchyAnalyzer().analyse(
            "BTC",
            [state(Timeframe.D1, Direction.STRONG_UP), state(Timeframe.M15, Direction.DOWN)],
        )
        micro_led = HierarchyAnalyzer().analyse(
            "BTC",
            [state(Timeframe.D1, Direction.DOWN), state(Timeframe.M15, Direction.STRONG_UP)],
        )
        assert daily_led.bias.is_bullish
        assert micro_led.bias.is_bearish

    def test_conflict_reduces_confidence(self) -> None:
        aligned = HierarchyAnalyzer().analyse(
            "BTC", [state(Timeframe.D1, Direction.UP), state(Timeframe.M15, Direction.UP)]
        )
        conflicted = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.WEAK_UP, strength=0.2, confidence=0.4),
                state(Timeframe.M15, Direction.DOWN),
            ],
        )
        assert conflicted.confidence < aligned.confidence

    def test_unusable_states_produce_insufficient_evidence(self) -> None:
        """Saying nothing is a valid output; guessing is not."""
        result = HierarchyAnalyzer().analyse(
            "BTC", [state(Timeframe.D1, Direction.UP, confidence=0.05)]
        )
        assert result.confidence == 0.0
        assert "insufficient evidence" in result.interpretation.lower()
        assert not result.is_actionable

    def test_degraded_data_quality_reduces_state_confidence(self) -> None:
        states = [state(Timeframe.D1, Direction.UP), state(Timeframe.H4, Direction.UP)]
        clean = HierarchyAnalyzer().analyse("BTC", states, data_quality=1.0)
        degraded = HierarchyAnalyzer().analyse("BTC", states, data_quality=0.25)
        assert degraded.confidence < clean.confidence
        assert degraded.bias is clean.bias

    def test_split_puts_structure_above_tactics(self) -> None:
        higher, lower = split_hierarchy(
            [state(tf, Direction.UP) for tf in (Timeframe.D1, Timeframe.H1, Timeframe.M15, Timeframe.M5)]
        )
        assert [s.timeframe for s in higher] == [Timeframe.D1, Timeframe.H1]
        assert [s.timeframe for s in lower] == [Timeframe.M15, Timeframe.M5]

    def test_all_states_are_retained_even_when_unusable(self) -> None:
        """The levels are stored in full; the explanation panel needs them."""
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.UP),
                state(Timeframe.M15, Direction.DOWN, confidence=0.05),
            ],
        )
        assert len(result.timeframes) == 2


class TestRegime:
    def test_high_volatility_downside_is_capitulation(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.STRONG_DOWN, strength=0.9, volatility=180.0),
                state(Timeframe.H4, Direction.STRONG_DOWN, strength=0.85, volatility=190.0),
            ],
        )
        assert result.regime is Regime.CAPITULATION

    def test_quiet_directionless_market_is_low_volatility(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.NEUTRAL, strength=0.1, volatility=20.0),
                state(Timeframe.H4, Direction.NEUTRAL, strength=0.1, volatility=18.0),
            ],
        )
        assert result.regime is Regime.LOW_VOLATILITY

    def test_volatility_outranks_direction(self) -> None:
        """A violent market is a different environment regardless of direction, and
        models calibrated in calm conditions do not transfer into it."""
        calm = HierarchyAnalyzer().analyse(
            "BTC", [state(Timeframe.D1, Direction.STRONG_UP, volatility=40.0)]
        )
        violent = HierarchyAnalyzer().analyse(
            "BTC", [state(Timeframe.D1, Direction.STRONG_UP, volatility=200.0)]
        )
        assert calm.regime is Regime.STRONG_BULL
        assert violent.regime is Regime.HIGH_VOLATILITY

    def test_trend_regimes_track_bias(self) -> None:
        for direction, expected in (
            (Direction.STRONG_UP, Regime.STRONG_BULL),
            (Direction.DOWN, Regime.BEAR),
            (Direction.NEUTRAL, Regime.NEUTRAL),
        ):
            result = HierarchyAnalyzer().analyse(
                "BTC",
                [state(Timeframe.D1, direction, volatility=60.0),
                 state(Timeframe.H4, direction, volatility=60.0)],
            )
            assert result.regime is expected, f"{direction} -> {result.regime}"


class TestStateEngine:
    @pytest.fixture
    def engine(self, database: Database, settings: Settings) -> StateEngine:
        return StateEngine(
            database, settings, source="fake",
            timeframes=[Timeframe.D1, Timeframe.H1],
        )

    async def _seed(self, database: Database, settings: Settings, kind: str) -> None:
        features = FeatureEngine(database, settings)
        for timeframe in (Timeframe.D1, Timeframe.H1):
            bars = candles_from(scenario(kind), timeframe=timeframe)
            async with database.session() as session:
                await OHLCVRepository(session).upsert_candles(bars)
            await features.backfill("BTC", timeframe, "fake")

    async def test_builds_state_from_stored_features(
        self, engine: StateEngine, database: Database, settings: Settings
    ) -> None:
        await self._seed(database, settings, "uptrend")
        result = await engine.build("BTC")

        assert result.timeframes, "both timeframes should have produced a reading"
        assert result.bias.is_bullish, result.summary()
        assert result.interpretation

    async def test_state_persists_with_all_levels(
        self, engine: StateEngine, database: Database, settings: Settings
    ) -> None:
        await self._seed(database, settings, "uptrend")
        built = await engine.build("BTC", persist=True)

        async with database.session() as session:
            stored = await MarketStateRepository(session).latest("BTC")
        assert stored is not None
        assert stored.bias == str(built.bias)
        assert stored.regime == str(built.regime)
        assert set(stored.levels) == {str(s.timeframe) for s in built.timeframes}
        assert stored.levels["1h"]["evidence"], "per-level evidence must survive storage"

    async def test_missing_features_yield_insufficient_evidence(
        self, engine: StateEngine
    ) -> None:
        result = await engine.build("BTC")
        assert result.confidence == 0.0
        assert "insufficient evidence" in result.interpretation.lower()

    async def test_as_of_excludes_bars_that_had_not_closed(
        self, engine: StateEngine, database: Database, settings: Settings
    ) -> None:
        """Rebuilding a past state must not use bars that were still forming then —
        otherwise every historical study is contaminated by hindsight."""
        await self._seed(database, settings, "uptrend")
        full = await engine.build("BTC")
        assert full.timeframes

        cutoff = min(s.as_of for s in full.timeframes if s.is_usable) - timedelta(days=5)
        historical = await engine.build("BTC", as_of=cutoff)
        for level in historical.timeframes:
            assert level.timeframe.close_time(level.as_of) <= cutoff


class TestConfidenceCeiling:
    """The system must never claim certainty about a market."""

    def test_classifier_confidence_never_reaches_one(self) -> None:
        """A hand-weighted rule set is not an oracle. Clean trending data makes every
        signal agree, which without a ceiling produces a confidence of exactly 1.00 —
        unearned certainty fed straight into Phase 7."""
        for kind in ("uptrend", "downtrend", "recovery", "capitulation"):
            result = classify(kind)
            assert result.confidence < 1.0, f"{kind} reported certainty"
            assert result.confidence <= 0.90 + 1e-9

    def test_state_confidence_never_reaches_one(self) -> None:
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.STRONG_UP, strength=1.0, confidence=0.9),
                state(Timeframe.H4, Direction.STRONG_UP, strength=1.0, confidence=0.9),
                state(Timeframe.H1, Direction.STRONG_UP, strength=1.0, confidence=0.9),
            ],
        )
        assert result.agreement == 1.0
        assert result.confidence < 1.0

    def test_the_ceiling_does_not_flatten_relative_confidence(self) -> None:
        """Capping must preserve ordering, or it would destroy the signal it protects."""
        clear = classify("uptrend")
        murky = classify("chop")
        assert clear.confidence > murky.confidence


class TestRelativeHierarchySplit:
    """The structural/tactical split is positional, not anchored to a fixed timeframe.

    An absolute hinge silently disables pullback detection whenever every requested
    timeframe falls on one side of it — a failure that hand-built unit tests cannot
    see, because they choose timeframes that straddle the hinge by construction.
    """

    def test_all_slow_timeframes_still_produce_a_tactical_group(self) -> None:
        higher, lower = split_hierarchy(
            [state(tf, Direction.UP) for tf in (Timeframe.D1, Timeframe.H4, Timeframe.H1)]
        )
        assert [s.timeframe for s in higher] == [Timeframe.D1, Timeframe.H4]
        assert [s.timeframe for s in lower] == [Timeframe.H1]

    def test_all_fast_timeframes_still_produce_a_structural_group(self) -> None:
        higher, lower = split_hierarchy(
            [state(tf, Direction.UP) for tf in (Timeframe.M15, Timeframe.M5, Timeframe.M1)]
        )
        assert [s.timeframe for s in higher] == [Timeframe.M15, Timeframe.M5]
        assert [s.timeframe for s in lower] == [Timeframe.M1]

    def test_pullback_is_detectable_on_slow_timeframe_sets(self) -> None:
        """The case the absolute hinge broke: 1d/4h/1h with the 1h correcting."""
        result = HierarchyAnalyzer().analyse(
            "BTC",
            [
                state(Timeframe.D1, Direction.UP, strength=0.8, confidence=0.85),
                state(Timeframe.H4, Direction.UP, strength=0.75, confidence=0.8),
                state(Timeframe.H1, Direction.DOWN, strength=0.6, confidence=0.7),
            ],
        )
        assert result.alignment is Alignment.PULLBACK_IN_UPTREND
        assert result.bias.is_bullish

    def test_a_single_timeframe_has_no_tactical_group(self) -> None:
        higher, lower = split_hierarchy([state(Timeframe.D1, Direction.UP)])
        assert len(higher) == 1 and lower == []
