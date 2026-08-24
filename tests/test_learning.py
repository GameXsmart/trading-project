"""The self-evaluation loop, and whether it actually learns anything.

Phase 9's gate is the sharpest in the project because §14 names the failure directly:
storing predictions is not learning. Two tests carry the weight.

* An injected model whose skill degrades in one regime must be down-weighted **in that
  regime and only that regime**. Skill is not a scalar property of a model, and a loop
  that reacts to a trend follower failing in chop by lowering its weight everywhere has
  learned something false.
* Recalibration must **measurably improve** calibration on data the curve was not
  fitted on — otherwise "recalibrated" is a log line, not a change.

Everything else here defends the bookkeeping the two gates rest on: append-only writes,
hash verification, resolution from final candles only, and slicing that refuses to
print a number it does not have the evidence for.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW
from tests.test_ensemble import _draw, _overstated, _truth_for, _uniform
from tests.test_models import HORIZON, HOUR, candles, drifting

from mie.learning.loop import LearningLoop, LearningReport, OutcomeResolver, _bar_covering
from mie.learning.metrics import slice_outcomes
from mie.learning.records import (
    PredictionRecord,
    ResolvedOutcome,
    content_hash,
    prediction_id,
    volatility_bucket,
)
from mie.learning.weights import WeightKey, WeightLearner
from mie.models.types import Distribution, Outcome, Prediction, PredictionEvidence
from mie.storage.repositories import PredictionRepository

# --------------------------------------------------------------------- helpers


def _prediction(
    *,
    model_id: str = "m",
    index: int = 0,
    distribution: Distribution | None = None,
    confidence: float = 0.5,
    regime: str = "trend",
    price: float = 100.0,
    threshold: float = 0.5,
) -> Prediction:
    return Prediction(
        model_id=model_id,
        asset="BTC",
        timeframe=HOUR,
        horizon=HORIZON,
        as_of=FIXED_NOW + timedelta(hours=index),
        distribution=distribution or Distribution.from_edge(0.2),
        confidence=confidence,
        regime=regime,
        move_threshold_pct=threshold,
        reference_price=price,
        evidence=[PredictionEvidence(label="because")],
    )


def _record(**kwargs) -> PredictionRecord:
    return PredictionRecord.of(_prediction(**kwargs), volatility="normal")


def _outcome(
    *,
    model_id: str,
    index: int,
    regime: str,
    brier: float,
    asset: str = "BTC",
    correct: bool = False,
) -> ResolvedOutcome:
    """A resolved outcome with a chosen Brier score.

    Constructed directly rather than scored from a distribution, so a test can specify
    exactly the skill profile it wants to inject and the learner's behaviour is the
    only thing under examination.
    """
    as_of = FIXED_NOW + timedelta(hours=index)
    return ResolvedOutcome(
        prediction_id=prediction_id(model_id, asset, str(HOUR), HORIZON.bars, as_of),
        model_id=model_id,
        asset=asset,
        timeframe=HOUR,
        horizon_bars=HORIZON.bars,
        regime=regime,
        volatility_bucket="normal",
        as_of=as_of,
        resolved_at=as_of + HORIZON.duration,
        realised_direction=Outcome.UP,
        realised_move_pct=1.0,
        exit_price=101.0,
        brier=brier,
        log_loss=0.9,
        correct=correct,
        probability_of_truth=0.4,
    )


#: Baseline Brier used throughout. Climatology on a balanced three-class problem sits
#: near 0.66, so this is the realistic bar rather than a flattering one.
_BASE = 0.66


def _paired(
    model_id: str,
    regime: str,
    count: int,
    model_brier: float,
    start: int = 0,
    jitter: float = 0.02,
) -> list[ResolvedOutcome]:
    """A run of outcomes for one model plus the baseline it is scored against.

    Jitter is deterministic and small: identical Brier scores at every point would give
    the paired test zero variance, which is not how any real measurement behaves and
    would let a degenerate case pass.
    """
    out: list[ResolvedOutcome] = []
    for i in range(count):
        index = start + i
        wobble = (_uniform(index, f"{model_id}{regime}") - 0.5) * jitter
        out.append(_outcome(model_id=model_id, index=index, regime=regime,
                            brier=model_brier + wobble))
        out.append(_outcome(model_id="baseline_climatology", index=index, regime=regime,
                            brier=_BASE + wobble))
    return out


# ------------------------------------------------------------------- records


class TestPredictionRecord:
    def test_the_id_is_derived_so_a_rerun_cannot_duplicate_the_sample(self) -> None:
        first = _record(index=5)
        second = _record(index=5)
        assert first.prediction_id == second.prediction_id
        assert _record(index=6).prediction_id != first.prediction_id

    def test_a_record_verifies_against_its_own_hash(self) -> None:
        assert _record().verify()

    def test_tampering_with_the_distribution_breaks_the_hash(self) -> None:
        """A drifted stored forecast would be indistinguishable from a better model."""
        record = _record()
        record.distribution = Distribution(up=0.9, flat=0.05, down=0.05)
        assert not record.verify()

    def test_tampering_with_the_confidence_breaks_the_hash(self) -> None:
        record = _record()
        record.confidence = 0.99
        assert not record.verify()

    def test_annotations_outside_the_claim_do_not_break_the_hash(self) -> None:
        """A check that cries wolf is a check that gets switched off."""
        record = _record()
        record.evidence = {"anything": "at all"}
        record.created_at = FIXED_NOW + timedelta(days=9)
        assert record.verify()

    def test_the_hash_covers_the_threshold_the_outcome_is_scored_against(self) -> None:
        a = content_hash(
            model_id="m", model_version="1", asset="BTC", timeframe="1h",
            horizon_bars=12, as_of=FIXED_NOW, up=0.4, flat=0.3, down=0.3,
            confidence=0.5, move_threshold_pct=0.5,
        )
        b = content_hash(
            model_id="m", model_version="1", asset="BTC", timeframe="1h",
            horizon_bars=12, as_of=FIXED_NOW, up=0.4, flat=0.3, down=0.3,
            confidence=0.5, move_threshold_pct=0.9,
        )
        assert a != b

    def test_a_prediction_is_not_due_before_its_horizon_elapses(self) -> None:
        record = _record()
        assert not record.is_due(record.resolves_at - timedelta(hours=1))
        assert not record.is_due(record.resolves_at)
        assert record.is_due(record.resolves_at + timedelta(hours=2))

    def test_scoring_uses_the_stored_threshold(self) -> None:
        """Recomputing it would score the forecast against a different question."""
        record = _record(price=100.0, threshold=2.0)
        outcome = ResolvedOutcome.score(record, 101.0, FIXED_NOW)
        assert outcome is not None
        # +1% against a 2% threshold is FLAT, even though it is clearly upward.
        assert outcome.realised_direction is Outcome.FLAT

    def test_scoring_refuses_a_degenerate_price(self) -> None:
        assert ResolvedOutcome.score(_record(price=0.0), 100.0, FIXED_NOW) is None


class TestVolatilityBucket:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.05, "very_low"), (0.2, "low"), (0.4, "normal"), (0.8, "high"), (2.0, "very_high")],
    )
    def test_buckets_split_the_observed_range(self, value: float, expected: str) -> None:
        assert volatility_bucket(value) == expected


# ------------------------------------------------------------------ resolver


class TestOutcomeResolver:
    @staticmethod
    def _series():
        return candles(drifting(400))

    def test_a_prediction_resolves_from_the_bar_covering_its_horizon(self) -> None:
        series = self._series()
        entry = series[100]
        record = PredictionRecord.of(
            _prediction(index=0, price=entry.close).model_copy(
                update={"as_of": HOUR.close_time(entry.open_time)}
            )
        )
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": series}, now=FIXED_NOW + timedelta(days=400)
        )
        assert not pending
        assert len(resolved) == 1
        assert resolved[0].exit_price == series[112].close

    def test_an_unripe_prediction_stays_pending(self) -> None:
        record = _record()
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": self._series()}, now=record.as_of
        )
        assert resolved == []
        assert pending == [record.prediction_id]

    def test_forming_bars_are_never_used_to_resolve(self) -> None:
        """Reading an incomplete candle is the same look-ahead error in reverse."""
        series = self._series()
        final = [c.model_copy(update={"is_final": i < 150}) for i, c in enumerate(series)]
        entry = series[200]
        record = PredictionRecord.of(
            _prediction(price=entry.close).model_copy(
                update={"as_of": HOUR.close_time(entry.open_time)}
            )
        )
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": final}, now=FIXED_NOW + timedelta(days=400)
        )
        assert resolved == []
        assert pending == [record.prediction_id]

    def test_a_corrupted_record_is_refused_not_scored(self) -> None:
        record = _record()
        record.confidence = 0.99
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": self._series()}, now=FIXED_NOW + timedelta(days=400)
        )
        assert resolved == []
        assert pending == []

    def test_already_resolved_records_are_skipped(self) -> None:
        record = _record()
        record.resolved = True
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": self._series()}, now=FIXED_NOW + timedelta(days=400)
        )
        assert resolved == [] and pending == []

    def test_bar_covering_never_reaches_past_the_moment(self) -> None:
        series = self._series()
        chosen = _bar_covering(series, series[50].close_time)
        assert chosen is series[50]
        assert _bar_covering(series, series[0].open_time - timedelta(hours=1)) is None

    def test_bar_covering_refuses_to_reach_backwards_across_a_gap(self) -> None:
        """Found by the forming-bar test: 'last bar at or before' is not enough.

        With the resolving bar missing, that rule quietly returns whatever was
        available and the outcome is scored against a price from an arbitrary distance
        in the past. It would look resolved and be fiction.
        """
        series = self._series()
        far = series[100].close_time + timedelta(hours=40)
        assert _bar_covering(series[:101], far) is series[100]
        assert _bar_covering(series[:101], far, timedelta(hours=1)) is None

    def test_a_prediction_whose_resolving_bar_is_missing_stays_pending(self) -> None:
        series = self._series()
        entry = series[100]
        record = PredictionRecord.of(
            _prediction(price=entry.close).model_copy(
                update={"as_of": HOUR.close_time(entry.open_time)}
            )
        )
        # Everything from the entry onward is absent from the feed.
        resolved, pending = OutcomeResolver().resolve(
            [record], {"BTC": series[:101]}, now=FIXED_NOW + timedelta(days=400)
        )
        assert resolved == []
        assert pending == [record.prediction_id]


# ------------------------------------------------------------------- metrics


class TestSlicedMetrics:
    @staticmethod
    def _outcomes(count: int = 120):
        out = []
        for i in range(count):
            out.append(
                _outcome(
                    model_id="m",
                    index=i,
                    regime="trend" if i % 2 else "chop",
                    brier=0.5 if i % 2 else 0.8,
                )
            )
        return out

    def test_every_dimension_is_sliced(self) -> None:
        table = slice_outcomes(self._outcomes())
        dimensions = {s.dimension for s in table.slices}
        assert dimensions == {"overall", "asset", "timeframe", "horizon", "regime", "volatility"}

    def test_regimes_are_kept_apart(self) -> None:
        """The whole point: an average over regimes destroys the finding."""
        table = slice_outcomes(self._outcomes())
        by_regime = {s.value: s.brier for s in table.for_dimension("regime")}
        assert by_regime["trend"] < by_regime["chop"]

    def test_a_thin_slice_reports_insufficient_evidence(self) -> None:
        table = slice_outcomes(self._outcomes(20))
        assert all(not s.has_evidence for s in table.slices)
        assert "insufficient evidence" in table.slices[0].verdict

    def test_class_balance_is_reported_so_brier_is_interpretable(self) -> None:
        table = slice_outcomes(self._outcomes())
        overall = table.for_dimension("overall")[0]
        assert sum(overall.class_balance.values()) == pytest.approx(1.0)

    def test_an_empty_run_produces_an_empty_table(self) -> None:
        assert slice_outcomes([]).slices == []


# ------------------------------------------------------------------- weights


class TestWeightLearner:
    def test_a_skilled_model_earns_a_weight(self) -> None:
        outcomes = _paired("good", "trend", 200, model_brier=0.55)
        updates = WeightLearner().learn(outcomes)
        weights = {u.key.regime: u.weight for u in updates if u.key.model_id == "good"}
        assert weights["trend"] > 0
        assert weights["all"] > 0

    def test_an_unskilled_model_earns_nothing(self) -> None:
        outcomes = _paired("flat", "trend", 200, model_brier=_BASE)
        updates = WeightLearner().learn(outcomes)
        assert all(u.weight == 0.0 for u in updates)
        assert any("not significant" in u.gated_out or "floor" in u.gated_out for u in updates)

    def test_a_worse_model_earns_nothing(self) -> None:
        """One-sided: significantly worse is not a pass."""
        outcomes = _paired("bad", "trend", 200, model_brier=0.80)
        updates = WeightLearner().learn(outcomes)
        assert all(u.weight == 0.0 for u in updates)

    def test_too_few_outcomes_earn_nothing(self) -> None:
        outcomes = _paired("good", "trend", 15, model_brier=0.40)
        updates = WeightLearner().learn(outcomes)
        assert all(u.weight == 0.0 for u in updates)
        assert any("resolved outcomes" in u.gated_out for u in updates)

    def test_degradation_in_one_regime_is_confined_to_that_regime(self) -> None:
        """Phase 9's gate, stated verbatim in the requirements.

        The model keeps working in `trend` throughout and stops working in `chop`
        halfway. The loop must lower the `chop` weight and leave `trend` alone: skill
        is not a scalar property of a model, and reacting everywhere would be learning
        something false.
        """
        learner = WeightLearner(recent_window=150)
        early = [
            *_paired("swinger", "trend", 150, model_brier=0.55, start=0),
            *_paired("swinger", "chop", 150, model_brier=0.55, start=1000),
        ]
        before = {u.key: u.weight for u in learner.learn(early)}
        trend_before = before[WeightKey("swinger", "BTC", str(HOUR), HORIZON.bars, "trend")]
        chop_before = before[WeightKey("swinger", "BTC", str(HOUR), HORIZON.bars, "chop")]
        assert trend_before > 0
        assert chop_before > 0

        # Same model, later period: trend unchanged, chop no better than the baseline.
        later = [
            *early,
            *_paired("swinger", "trend", 150, model_brier=0.55, start=2000),
            *_paired("swinger", "chop", 150, model_brier=_BASE, start=3000),
        ]
        after = {u.key: u.weight for u in learner.learn(later, previous=before)}
        trend_after = after[WeightKey("swinger", "BTC", str(HOUR), HORIZON.bars, "trend")]
        chop_after = after[WeightKey("swinger", "BTC", str(HOUR), HORIZON.bars, "chop")]

        assert chop_after == 0.0, "degraded regime must lose its weight"
        assert trend_after > 0, "the healthy regime must keep its weight"
        assert trend_after == pytest.approx(trend_before, abs=0.02)

    def test_another_models_weight_is_untouched_by_a_neighbours_collapse(self) -> None:
        learner = WeightLearner(recent_window=150)
        stable = _paired("stable", "trend", 150, model_brier=0.55, start=0)
        before = {u.key: u.weight for u in learner.learn(stable)}
        with_collapse = [*stable, *_paired("collapsed", "trend", 150, model_brier=0.9, start=500)]
        after = {u.key: u.weight for u in learner.learn(with_collapse, previous=before)}
        stable_key = WeightKey("stable", "BTC", str(HOUR), HORIZON.bars, "trend")
        assert after[stable_key] > 0
        assert after[WeightKey("collapsed", "BTC", str(HOUR), HORIZON.bars, "trend")] == 0.0

    def test_more_evidence_retains_more_of_the_measured_skill(self) -> None:
        """Sample-size shrinkage: a slice that just cleared the gate cannot dominate."""
        small = WeightLearner().learn(_paired("g", "trend", 50, model_brier=0.55))
        large = WeightLearner().learn(_paired("g", "trend", 400, model_brier=0.55))
        small_weight = next(u.weight for u in small if u.key.regime == "trend")
        large_weight = next(u.weight for u in large if u.key.regime == "trend")
        assert 0 < small_weight < large_weight

    def test_weights_among_qualifiers_are_blended_toward_equal(self) -> None:
        """Shrinkage toward equal weights — but only among models that qualified."""
        outcomes = [
            *_paired("strong", "trend", 200, model_brier=0.50),
            *_paired("weaker", "trend", 200, model_brier=0.60),
        ]
        blended = {
            u.key.model_id: u.weight
            for u in WeightLearner(equal_blend=0.5).learn(outcomes)
            if u.key.regime == "trend"
        }
        unblended = {
            u.key.model_id: u.weight
            for u in WeightLearner(equal_blend=0.0).learn(outcomes)
            if u.key.regime == "trend"
        }
        assert blended["strong"] < unblended["strong"]
        assert blended["weaker"] > unblended["weaker"]

    def test_a_model_that_never_qualified_stays_at_zero_not_an_equal_share(self) -> None:
        """The deliberate deviation: equal weights would hand out unearned influence."""
        outcomes = [
            *_paired("strong", "trend", 200, model_brier=0.50),
            *_paired("useless", "trend", 200, model_brier=_BASE),
        ]
        weights = {
            u.key.model_id: u.weight
            for u in WeightLearner(equal_blend=0.5).learn(outcomes)
            if u.key.regime == "trend"
        }
        assert weights["useless"] == 0.0
        assert weights["strong"] > 0

    def test_without_baseline_outcomes_nothing_is_learned(self) -> None:
        lonely = [_outcome(model_id="m", index=i, regime="trend", brier=0.2) for i in range(200)]
        assert WeightLearner().learn(lonely) == []

    def test_an_update_reports_why_it_was_gated_out(self) -> None:
        updates = WeightLearner().learn(_paired("g", "trend", 15, model_brier=0.4))
        assert all(u.gated_out for u in updates)
        assert all("no weight" in u.summary() for u in updates)


# ---------------------------------------------------------------------- loop


class TestLearningLoop:
    @staticmethod
    def _miscalibrated(count: int = 900) -> tuple[list[PredictionRecord], list[ResolvedOutcome]]:
        """A model overstating its edge by a factor of three, with matching outcomes."""
        levels = (0.55, 0.65, 0.75, 0.85)
        records, outcomes = [], []
        for i in range(count):
            level = levels[i % len(levels)]
            stated = _overstated(level)
            actual = _draw(i, _truth_for(level, 0.35), "loop")
            record = PredictionRecord.of(
                _prediction(model_id="overconfident", index=i, distribution=stated,
                            confidence=0.6, regime="trend"),
                volatility="normal",
            )
            records.append(record)
            scored = ResolvedOutcome.score(record, 101.0, record.resolves_at)
            assert scored is not None
            outcomes.append(
                replace(
                    scored,
                    realised_direction=actual,
                    brier=sum(
                        (stated.probability(o) - (1.0 if o is actual else 0.0)) ** 2
                        for o in Outcome
                    ),
                    probability_of_truth=stated.probability(actual),
                )
            )
        return records, outcomes

    def test_recalibration_measurably_improves_on_held_out_data(self) -> None:
        """Phase 9's second gate."""
        records, outcomes = self._miscalibrated()
        report = LearningLoop().run(records, {}, existing_outcomes=outcomes)
        adopted = report.adopted_calibrations
        assert adopted, "a systematically miscalibrated model should be correctable"
        assert all(c.improvement > 0 for c in adopted)
        assert any(c.ece_after < c.ece_before for c in adopted)
        assert report.learned

    def test_too_little_evidence_is_not_reported_as_having_learned_nothing(self) -> None:
        """Three states, not two: a waiting room is not a finding."""
        records, outcomes = self._miscalibrated(20)
        report = LearningLoop().run(records, {}, existing_outcomes=outcomes)
        assert not report.had_enough_evidence
        assert not report.learned
        assert "nothing to learn from yet" in report.verdict

    def test_enough_evidence_and_no_change_is_reported_as_learned_nothing(self) -> None:
        """The honest finding, and the one this repo keeps producing."""
        outcomes = _paired("honest", "trend", 200, model_brier=_BASE)
        report = LearningLoop().run([], {}, existing_outcomes=outcomes)
        assert report.had_enough_evidence
        assert not report.learned
        assert "learned nothing" in report.verdict

    def test_a_weight_change_counts_as_learning(self) -> None:
        outcomes = _paired("good", "trend", 200, model_brier=0.55)
        report = LearningLoop().run([], {}, existing_outcomes=outcomes)
        assert report.weight_changes
        assert report.learned
        assert "learned:" in report.verdict

    def test_metrics_are_computed_even_when_nothing_is_learned(self) -> None:
        """Measuring is free; changing weights on thin evidence is not."""
        records, outcomes = self._miscalibrated(40)
        report = LearningLoop().run(records, {}, existing_outcomes=outcomes)
        assert report.metrics.slices
        assert not report.weights.updates

    def test_prior_outcomes_are_carried_forward(self) -> None:
        """A loop that forgot between runs would never grow past one horizon."""
        outcomes = _paired("good", "trend", 200, model_brier=0.55)
        report = LearningLoop().run([], {}, existing_outcomes=outcomes)
        assert report.total_outcomes == len(outcomes)
        assert report.resolved == 0

    def test_corrupted_records_are_counted_and_refused(self) -> None:
        records, _ = self._miscalibrated(5)
        records[0].confidence = 0.99
        report = LearningLoop().run(records, {"BTC": candles(drifting(400))})
        assert report.corrupted == 1

    def test_the_report_renders(self) -> None:
        outcomes = _paired("good", "trend", 200, model_brier=0.55)
        text = LearningLoop().run([], {}, existing_outcomes=outcomes).report()
        assert "VERDICT" in text
        assert "Learning loop" in text

    def test_an_empty_run_says_so_rather_than_crashing(self) -> None:
        report = LearningLoop().run([], {})
        assert isinstance(report, LearningReport)
        assert report.total_outcomes == 0
        assert not report.learned


# ------------------------------------------------------------------ storage


class TestPredictionRepository:
    async def test_predictions_are_append_only(self, database) -> None:
        """Re-running the same point must not duplicate or revise the sample."""
        async with database.session() as session:
            repo = PredictionRepository(session)
            record = _record(index=1)
            await repo.append([record])
            await session.commit()

            revised = _record(index=1)
            revised.distribution = Distribution(up=0.99, flat=0.005, down=0.005)
            await repo.append([revised])
            await session.commit()

            counts = await repo.counts()
            assert counts["predictions"] == 1
            stored = (await repo.unresolved())[0]
            assert stored.prob_up == pytest.approx(record.distribution.up)

    async def test_outcomes_mark_their_prediction_resolved(self, database) -> None:
        async with database.session() as session:
            repo = PredictionRepository(session)
            record = _record(index=2)
            await repo.append([record])
            outcome = ResolvedOutcome.score(record, 105.0, record.resolves_at)
            assert outcome is not None
            await repo.record_outcomes([outcome])
            await session.commit()

            counts = await repo.counts()
            assert counts == {"predictions": 1, "resolved": 1, "pending": 0, "outcomes": 1}
            assert await repo.unresolved() == []

    async def test_due_returns_only_ripe_predictions(self, database) -> None:
        async with database.session() as session:
            repo = PredictionRepository(session)
            await repo.append([_record(index=3)])
            await session.commit()
            assert await repo.due(now=FIXED_NOW) == []
            assert len(await repo.due(now=FIXED_NOW + timedelta(days=2))) == 1

    async def test_records_returns_resolved_predictions_too(self, database) -> None:
        """Recalibration needs them, and loading only unresolved rows fails silently.

        Found by running the loop twice: the first pass resolved the whole backlog, and
        the second had no records left to pair outcomes against, so calibration
        quietly stopped happening rather than reporting that it could not.
        """
        async with database.session() as session:
            repo = PredictionRepository(session)
            record = _record(index=7)
            await repo.append([record])
            outcome = ResolvedOutcome.score(record, 105.0, record.resolves_at)
            assert outcome is not None
            await repo.record_outcomes([outcome])
            await session.commit()

            assert await repo.unresolved() == []
            assert len(await repo.records()) == 1

    async def test_weights_are_upserted_with_their_previous_value(self, database) -> None:
        async with database.session() as session:
            repo = PredictionRepository(session)
            updates = WeightLearner().learn(_paired("good", "trend", 200, model_brier=0.55))
            await repo.upsert_weights(updates)
            await session.commit()
            rows = await repo.weights()
            assert rows
            assert any(r.weight > 0 for r in rows)
            assert all(r.samples > 0 for r in rows)
