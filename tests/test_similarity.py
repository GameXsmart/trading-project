"""Historical similarity search and sequence mining.

Both techniques share a failure mode: they will *always* return something. k nearest
neighbours exist in any dataset, and a long enough history contains every sequence.
The tests here are mostly about the guards against that — the distance ceiling, the
embargo, the minimum sample, and the multiple-comparison correction — because without
them both modules would produce confident output from noise.
"""

from __future__ import annotations

import math
from datetime import timedelta

from tests.conftest import FIXED_NOW

from mie.core.timeframes import Timeframe
from mie.patterns.sequences import SequenceMiner
from mie.patterns.similarity import SimilarityEngine
from mie.patterns.types import PATTERN_DIRECTIONS, Detection, PatternKind

# ---------------------------------------------------------------------- helpers


def feature_vector(rsi: float, adx: float = 25.0, vol: float = 50.0) -> dict[str, float]:
    """A complete comparison vector, parameterised on the dimensions under test."""
    return {
        "rsi_14": rsi,
        "bb_20.percent_b": rsi / 100.0,
        "bb_20.bandwidth": 5.0,
        "adx_14.adx": adx,
        "adx_14.plus_di": 25.0,
        "adx_14.minus_di": 20.0,
        "atr_14.atr_pct": 1.0,
        "roc_10": (rsi - 50.0) / 10.0,
        "stoch_14.k": rsi,
        "realised_vol_20": vol,
        "vwap.vwap_distance_pct": 0.5,
        "structure_trend": 1.0,
    }


def build_history(
    count: int, rsi_of, close_of
) -> tuple[list[tuple], list[float]]:
    """Parallel (timestamp, features) and closes series."""
    history = [
        (FIXED_NOW + timedelta(hours=i), feature_vector(rsi_of(i))) for i in range(count)
    ]
    closes = [close_of(i) for i in range(count)]
    return history, closes


def detection(kind: PatternKind, at) -> Detection:
    return Detection(
        kind=kind,
        asset="BTC",
        timeframe=Timeframe.H1,
        at=at,
        direction=PATTERN_DIRECTIONS[kind],
        close=100.0,
    )


# ------------------------------------------------------------------ similarity


class TestSimilarityGuards:
    """The guards that stop the search from inventing analogues."""

    def test_dissimilar_history_yields_insufficient_evidence(self) -> None:
        """The important case. k nearest neighbours always exist; that does not make
        them comparable, and returning them anyway is how this technique lies."""
        # Query sits at RSI 95; all history sits near RSI 30-35.
        history, closes = build_history(
            600, rsi_of=lambda i: 30.0 + (i % 5), close_of=lambda i: 100.0 + i * 0.1
        )
        history[-1] = (history[-1][0], feature_vector(95.0))

        result = SimilarityEngine().search(
            history, closes, query_index=len(history) - 1, horizon=12,
            asset="BTC", timeframe=Timeframe.H1,
        )
        assert not result.has_evidence
        assert result.rejected_as_dissimilar > 0
        assert "insufficient evidence" in result.summary()

    def test_similar_history_produces_analogues(self) -> None:
        history, closes = build_history(
            600,
            rsi_of=lambda i: 50.0 + 10.0 * math.sin(i / 9.0),
            close_of=lambda i: 100.0 + 10.0 * math.sin(i / 9.0),
        )
        result = SimilarityEngine().search(
            history, closes, query_index=len(history) - 1, horizon=12,
            asset="BTC", timeframe=Timeframe.H1,
        )
        assert result.has_evidence
        assert result.estimate is not None
        assert len(result.analogues) >= 20

    def test_analogues_are_embargoed_from_the_query(self) -> None:
        """A neighbour whose forward window overlaps the query's present would be
        reporting the future as evidence about it."""
        history, closes = build_history(
            600,
            rsi_of=lambda i: 50.0 + 10.0 * math.sin(i / 9.0),
            close_of=lambda i: 100.0 + 10.0 * math.sin(i / 9.0),
        )
        horizon = 12
        query_index = len(history) - 1
        result = SimilarityEngine().search(
            history, closes, query_index=query_index, horizon=horizon,
            asset="BTC", timeframe=Timeframe.H1,
        )
        cutoff = history[query_index - horizon][0]
        assert all(a.at < cutoff for a in result.analogues)

    def test_short_history_returns_nothing_rather_than_guessing(self) -> None:
        history, closes = build_history(
            30, rsi_of=lambda i: 50.0, close_of=lambda i: 100.0
        )
        result = SimilarityEngine().search(
            history, closes, query_index=29, horizon=12, asset="BTC", timeframe=Timeframe.H1
        )
        assert not result.has_evidence
        assert result.analogues == []

    def test_missing_features_exclude_a_candidate(self) -> None:
        history, closes = build_history(
            400,
            rsi_of=lambda i: 50.0 + 5.0 * math.sin(i / 7.0),
            close_of=lambda i: 100.0 + i * 0.02,
        )
        # Strip a feature from half the history; those bars cannot be compared.
        for i in range(0, 200):
            stripped = dict(history[i][1])
            del stripped["rsi_14"]
            history[i] = (history[i][0], stripped)

        result = SimilarityEngine().search(
            history, closes, query_index=len(history) - 1, horizon=12,
            asset="BTC", timeframe=Timeframe.H1,
        )
        assert result.searched < 200

    def test_comparison_is_scale_free(self) -> None:
        """Two eras at completely different price levels must still be comparable —
        the search is about behaviour, not about price."""
        cheap_history, cheap_closes = build_history(
            600,
            rsi_of=lambda i: 50.0 + 10.0 * math.sin(i / 9.0),
            close_of=lambda i: 1.0 + 0.1 * math.sin(i / 9.0),
        )
        rich_history, rich_closes = build_history(
            600,
            rsi_of=lambda i: 50.0 + 10.0 * math.sin(i / 9.0),
            close_of=lambda i: 90000.0 + 9000.0 * math.sin(i / 9.0),
        )
        engine = SimilarityEngine()
        cheap = engine.search(
            cheap_history, cheap_closes, len(cheap_history) - 1, 12, "DOGE", Timeframe.H1
        )
        rich = engine.search(
            rich_history, rich_closes, len(rich_history) - 1, 12, "BTC", Timeframe.H1
        )
        assert len(cheap.analogues) == len(rich.analogues)

    def test_result_reports_baseline_comparison_not_a_bare_rate(self) -> None:
        history, closes = build_history(
            800,
            rsi_of=lambda i: 50.0 + 10.0 * math.sin(i / 9.0),
            close_of=lambda i: 100.0 + 10.0 * math.sin(i / 9.0) + i * 0.01,
        )
        result = SimilarityEngine().search(
            history, closes, len(history) - 1, 12, "BTC", Timeframe.H1
        )
        assert result.estimate is not None
        assert 0.0 <= result.estimate.baseline <= 1.0
        assert "vs baseline" in result.summary()


# -------------------------------------------------------------------- sequences


class TestSequenceMining:
    def test_chains_below_the_occurrence_floor_are_not_reported(self) -> None:
        """A chain seen a handful of times cannot support any claim."""
        base = FIXED_NOW
        detections = [
            detection(PatternKind.BREAKOUT_UP, base),
            detection(PatternKind.EXPANSION, base + timedelta(hours=2)),
        ]
        index_of = {d.at: i * 2 for i, d in enumerate(detections)}
        closes = [100.0 + i * 0.1 for i in range(200)]
        assert SequenceMiner().mine(detections, closes, index_of, "BTC", Timeframe.H1) == []

    def test_a_repeated_chain_is_enumerated_and_tested(self) -> None:
        base = FIXED_NOW
        detections, index_of = [], {}
        # 60 repetitions of breakout -> expansion, three bars apart.
        for repeat in range(60):
            start = repeat * 20
            first = base + timedelta(hours=start)
            second = base + timedelta(hours=start + 3)
            detections += [
                detection(PatternKind.BREAKOUT_UP, first),
                detection(PatternKind.EXPANSION, second),
            ]
            index_of[first] = start
            index_of[second] = start + 3

        closes = [100.0 + i * 0.05 for i in range(1400)]
        chains = SequenceMiner(horizons=(12,)).mine(
            detections, closes, index_of, "BTC", Timeframe.H1
        )
        labels = {c.label for c in chains}
        assert "breakout_up -> expansion" in labels
        for chain in chains:
            assert chain.estimate.trials > 0
            assert chain.occurrences >= 25

    def test_distant_events_are_not_treated_as_one_episode(self) -> None:
        """Two events a hundred bars apart both happened; they are not a chain."""
        base = FIXED_NOW
        detections, index_of = [], {}
        for repeat in range(60):
            start = repeat * 200
            first = base + timedelta(hours=start)
            second = base + timedelta(hours=start + 100)
            detections += [
                detection(PatternKind.BREAKOUT_UP, first),
                detection(PatternKind.EXPANSION, second),
            ]
            index_of[first] = start
            index_of[second] = start + 100

        closes = [100.0 + i * 0.05 for i in range(13000)]
        chains = SequenceMiner(horizons=(12,), max_gap_bars=12).mine(
            detections, closes, index_of, "BTC", Timeframe.H1
        )
        assert "breakout_up -> expansion" not in {c.label for c in chains}

    def test_mining_applies_multiple_comparison_correction(self) -> None:
        """Pairs and triples across an alphabet generate thousands of hypotheses;
        uncorrected, the output would be chains assembled entirely from noise."""
        base = FIXED_NOW
        kinds = list(PatternKind)[:6]
        detections, index_of = [], {}
        for step in range(600):
            kind = kinds[step % len(kinds)]
            at = base + timedelta(hours=step * 2)
            detections.append(detection(kind, at))
            index_of[at] = step * 2

        # A pure sawtooth: no chain can carry information about it.
        closes = [100.0 + (i % 7) * 0.3 for i in range(1400)]
        chains = SequenceMiner(horizons=(12,)).mine(
            detections, closes, index_of, "BTC", Timeframe.H1
        )
        assert not [c for c in chains if c.is_informative]


class TestTransitionMatrix:
    def test_transition_probabilities_carry_intervals(self) -> None:
        states = ["bull", "neutral"] * 40 + ["bull", "bear"] * 20
        matrix = SequenceMiner().transition_matrix(states)
        assert matrix.transitions
        for transition in matrix.transitions:
            assert transition.low <= transition.probability <= transition.high
            assert "[" in transition.summary()

    def test_self_transitions_are_excluded(self) -> None:
        """States persist for long stretches; including self-transitions produces a
        matrix whose every row says 'most likely, no change'."""
        states = ["bull"] * 100 + ["bear"] * 100
        matrix = SequenceMiner().transition_matrix(states, min_observations=1)
        assert all(t.source != t.target for t in matrix.transitions)

    def test_rare_source_states_are_not_reported(self) -> None:
        states = ["bull", "bear"] * 50 + ["capitulation", "recovery"]
        matrix = SequenceMiner().transition_matrix(states, min_observations=20)
        assert "capitulation" not in {t.source for t in matrix.transitions}

    def test_most_likely_successor_is_identified(self) -> None:
        states = (["bull", "neutral"] * 30) + (["bull", "bear"] * 5)
        matrix = SequenceMiner().transition_matrix(states, min_observations=10)
        nxt = matrix.most_likely_after("bull")
        assert nxt is not None
        assert nxt.target == "neutral"

    def test_empty_input_is_handled(self) -> None:
        assert SequenceMiner().transition_matrix([]).transitions == []
