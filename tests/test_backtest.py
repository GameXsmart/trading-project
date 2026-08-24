"""Walk-forward folds, leakage detection and survivorship handling.

The load-bearing tests here are the ones that prove the leakage detector *works*.
Phase 8's gate is "a deliberately leaky model is caught by the harness", and a detector
that has only ever been run against clean code has not been shown to detect anything.
So a leaky pipeline is built on purpose and the probe is required to catch it — and,
just as importantly, a clean pipeline is required to come back clean, since a detector
that flags everything is equally useless.

The probe's blind spot is tested too. It cannot see a model reading the future through
a channel outside the context it was handed, so the implausible-skill screen covers
that case, and both the catch and the boundary between them are asserted rather than
described.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW
from tests.test_models import HORIZON, HOUR, _OracleModel, candles, drifting

from mie.backtest.harness import WalkForwardHarness
from mie.backtest.leakage import (
    LeakageProbe,
    Verdict,
    corrupt_after,
    corrupt_before,
    implausible_skill,
)
from mie.backtest.universe import AssetListing, HistoricalUniverse
from mie.backtest.windows import DataWindow, Fold, FoldScheme, generate_folds
from mie.models.base import PredictionContext, Predictor
from mie.models.baselines import ClimatologyBaseline, PersistenceBaseline, UniformBaseline
from mie.models.evaluation import WalkForwardEvaluator
from mie.models.predictors import ALL_MODELS
from mie.models.runner import ContextSource, build_contexts
from mie.models.types import Distribution, Horizon, Outcome, Prediction

# --------------------------------------------------------------------- fixtures


class _PriceModel(Predictor):
    """A model whose view depends only on recent closes — responsive, and honest."""

    model_id = "price"
    warmup_bars = 20

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"price"})

    def predict(self, ctx: PredictionContext) -> Prediction:
        closes = [c.close for c in ctx.candles[-20:] if c.close > 0]
        if len(closes) < 20 or closes[0] <= 0:
            return self.abstain(ctx, "insufficient history")
        edge = max(-1.0, min(1.0, (closes[-1] - closes[0]) / closes[0] * 20.0))
        return self.build(ctx, Distribution.from_edge(edge), confidence=0.5)


class _ConfidenceOnlyModel(_PriceModel):
    """Emits a fixed distribution but varies its *confidence* with the last close.

    A model can leak through confidence alone — reading the future to decide how sure
    to be, without moving a single probability. The probe compares confidence for
    exactly this reason.
    """

    model_id = "confidence_only"

    def predict(self, ctx: PredictionContext) -> Prediction:
        if not ctx.candles:
            return self.abstain(ctx, "no data")
        last = ctx.candles[-1].close
        return self.build(
            ctx,
            distribution=Distribution(up=0.4, flat=0.35, down=0.25),
            confidence=min(1.0, abs(last) % 1.0),
        )


class _InertModel(Predictor):
    """Always abstains. The probe must call this INCONCLUSIVE, never CLEAN."""

    model_id = "inert"
    warmup_bars = 1

    def inputs_used(self) -> frozenset[str]:
        return frozenset()

    def predict(self, ctx: PredictionContext) -> Prediction:
        return self.abstain(ctx, "never has a view")


class _AnotherInertModel(_InertModel):
    """A second model with nothing to say, to prove the redundancy check fires."""

    model_id = "inert_too"


class _LeakySource(ContextSource):
    """A context source carrying the exact bug the probe exists to find.

    It hands the model the *whole* series instead of the prefix ending at ``as_of``.
    This is not a contrived failure: forgetting to slice, or filtering one field and
    not another, is the realistic shape of a look-ahead bug in this codebase, and it
    looks entirely correct when read.
    """

    def context_at(self, index: int, horizon: Horizon) -> PredictionContext | None:
        context = super().context_at(index, horizon)
        if context is not None:
            context.candles = list(self.candles)
        return context


def _source(bars: int = 900, leaky: bool = False) -> ContextSource:
    series = candles(drifting(bars))
    cls = _LeakySource if leaky else ContextSource
    return cls("BTC", HOUR, series)


def _probe() -> LeakageProbe:
    return LeakageProbe(max_points=8)


# ------------------------------------------------------------------ corruption


class TestCorruption:
    def test_corrupting_the_future_leaves_the_past_untouched(self) -> None:
        series = candles(drifting(200))
        boundary = series[100].close_time
        corrupted = corrupt_after(series, boundary)
        assert [c.close for c in corrupted[:101]] == [c.close for c in series[:101]]
        assert [c.close for c in corrupted[101:]] != [c.close for c in series[101:]]

    def test_corrupting_the_past_leaves_the_future_untouched(self) -> None:
        series = candles(drifting(200))
        boundary = series[100].close_time
        corrupted = corrupt_before(series, boundary)
        assert [c.close for c in corrupted[101:]] == [c.close for c in series[101:]]
        assert [c.close for c in corrupted[:101]] != [c.close for c in series[:101]]

    def test_corrupted_bars_are_still_structurally_valid(self) -> None:
        """Otherwise the probe would be testing the validator, not the model."""
        series = candles(drifting(200))
        for bar in corrupt_after(series, series[0].close_time):
            assert bar.high >= bar.low
            assert bar.high >= max(bar.open, bar.close)
            assert bar.low <= min(bar.open, bar.close)
            assert bar.open > 0 and bar.close > 0

    def test_corruption_actually_changes_direction(self) -> None:
        """Scaling alone would not move a model that keys on direction."""
        series = candles(drifting(200))
        corrupted = corrupt_after(series, series[0].close_time)
        signs = [
            (a.close > a.open) != (b.close > b.open)
            for a, b in zip(series[1:], corrupted[1:], strict=True)
            if a.close != a.open
        ]
        assert sum(signs) > len(signs) * 0.9


# ----------------------------------------------------------------- the probe


class TestLeakageProbe:
    def test_a_deliberately_leaky_pipeline_is_caught(self) -> None:
        """Phase 8's gate, stated verbatim in the requirements."""
        report = _probe().probe(_PriceModel(), _source(leaky=True), HORIZON)
        assert report.verdict is Verdict.LEAKING
        assert report.leaking_points
        assert report.worst_response > 0.01
        assert "LEAKING" in report.summary()

    def test_the_same_model_on_a_clean_pipeline_is_clean(self) -> None:
        """Without this, the test above would pass on a detector that flags everything."""
        report = _probe().probe(_PriceModel(), _source(), HORIZON)
        assert report.verdict is Verdict.CLEAN
        assert not report.leaking_points
        assert report.responsive_points == report.tested

    def test_a_leak_through_confidence_alone_is_caught(self) -> None:
        """No probability moves, but the model still read the future."""
        report = _probe().probe(_ConfidenceOnlyModel(), _source(leaky=True), HORIZON)
        assert report.verdict is Verdict.LEAKING

    def test_a_model_that_never_responds_is_inconclusive_not_clean(self) -> None:
        """The control. A detector that laundered this into 'clean' would be worthless."""
        report = _probe().probe(_InertModel(), _source(), HORIZON)
        assert report.verdict is Verdict.INCONCLUSIVE
        assert report.responsive_points == 0
        assert "INCONCLUSIVE" in report.summary()

    def test_leaking_outranks_inconclusive(self) -> None:
        """One contaminated point invalidates the run; it is not a rate to tolerate.

        This model reads only the last bar, so corrupting the past moves nothing and
        every point is unresponsive — the shape that would otherwise be reported as
        INCONCLUSIVE. It leaks, so it is reported as leaking.
        """
        report = _probe().probe(_ConfidenceOnlyModel(), _source(leaky=True), HORIZON)
        assert report.responsive_points == 0
        assert report.verdict is Verdict.LEAKING

    def test_the_corrupted_arms_keep_the_source_behaviour(self) -> None:
        """Otherwise the probe compares a leaky context against a correct one.

        The rebuilt sources must reproduce whatever the original does, or the measured
        difference is an artefact of the rebuild rather than evidence about the model.
        """
        probe = _probe()
        source = _source(leaky=True)
        context = source.context_at(500, HORIZON)
        assert context is not None
        rebuilt = probe._rebuild(source, context.as_of, corrupt_before)
        assert type(rebuilt) is type(source)
        rebuilt_context = rebuilt.context_at(500, HORIZON)
        assert rebuilt_context is not None
        assert len(rebuilt_context.candles) == len(context.candles)

    def test_the_uniform_baseline_is_inconclusive(self) -> None:
        report = _probe().probe(UniformBaseline(), _source(), HORIZON)
        assert report.verdict is Verdict.INCONCLUSIVE

    @pytest.mark.parametrize("baseline", [ClimatologyBaseline, PersistenceBaseline])
    def test_the_baselines_do_not_leak(self, baseline: type[Predictor]) -> None:
        report = _probe().probe(baseline(), _source(), HORIZON)
        assert report.verdict is not Verdict.LEAKING

    def test_no_shipped_model_leaks(self) -> None:
        """The audit that matters: the real pipeline, probed rather than trusted."""
        source = _source()
        for factory in ALL_MODELS:
            report = _probe().probe(factory(), source, HORIZON)
            assert report.verdict is not Verdict.LEAKING, report.summary()

    def test_probe_points_are_spread_across_history(self) -> None:
        report = _probe().probe(_PriceModel(), _source(), HORIZON)
        moments = [p.as_of for p in report.points]
        assert moments == sorted(moments)
        assert len(set(moments)) == len(moments)
        assert (moments[-1] - moments[0]) > timedelta(hours=100)


# ---------------------------------------------------------- implausible skill


class TestSkillScreen:
    @staticmethod
    def _contexts():
        return build_contexts(
            ContextSource("BTC", HOUR, candles(drifting(1400))), HORIZON, warmup=450
        )

    def test_an_oracle_is_flagged_as_suspicious(self) -> None:
        """The backstop for leaks perturbation structurally cannot see."""
        pairs = self._contexts()
        truth = {c.as_of: Outcome.classify(r, c.threshold_pct) for c, r in pairs}
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [_OracleModel(truth)], pairs
        )
        screens = implausible_skill(report.scores)
        assert screens
        assert screens[0].suspicious
        assert screens[0].verdict is Verdict.SUSPICIOUS
        assert "SUSPICIOUS" in screens[0].summary()

    def test_a_perturbation_probe_cannot_see_the_oracle(self) -> None:
        """States the boundary between the two mechanisms, rather than implying none.

        The oracle reads a dict it was handed at construction, not the context, so
        corrupting the source moves nothing. This is exactly why the screen exists.
        """
        pairs = self._contexts()
        truth = {c.as_of: Outcome.classify(r, c.threshold_pct) for c, r in pairs}
        report = _probe().probe(
            _OracleModel(truth), ContextSource("BTC", HOUR, candles(drifting(1400))), HORIZON
        )
        assert report.verdict is not Verdict.LEAKING

    def test_real_models_are_not_flagged(self) -> None:
        """A screen that fired on ordinary results would be a nuisance, not a check."""
        pairs = self._contexts()
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [m() for m in ALL_MODELS], pairs
        )
        assert not [s for s in implausible_skill(report.scores) if s.suspicious]

    def test_only_the_overall_slice_is_screened(self) -> None:
        """Regime slices are small enough to post wild numbers by chance."""
        pairs = self._contexts()
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [m() for m in ALL_MODELS], pairs
        )
        assert len(implausible_skill(report.scores)) == len(ALL_MODELS)


# ---------------------------------------------------------------------- folds


class TestFolds:
    @staticmethod
    def _series(bars: int = 1600):
        return candles(drifting(bars))

    def test_folds_are_chronological_and_do_not_overlap(self) -> None:
        folds = generate_folds(self._series(), horizon_bars=12, folds=4)
        assert len(folds) == 4
        for fold in folds:
            assert not fold.train.overlaps(fold.test)
            assert fold.train.end_index <= fold.test.start_index
        starts = [f.test.start_index for f in folds]
        assert starts == sorted(starts)

    def test_the_gap_covers_the_horizon(self) -> None:
        """Purging: a training label reaching h bars forward contaminates h bars."""
        folds = generate_folds(self._series(), horizon_bars=24, folds=3)
        assert folds
        for fold in folds:
            assert fold.purge_bars == 24
            assert fold.gap_bars >= fold.purge_bars
            assert not fold.leaks()

    def test_an_embargo_is_added_on_top_of_the_purge(self) -> None:
        folds = generate_folds(self._series(), horizon_bars=24, folds=3)
        assert all(f.embargo_bars > 0 for f in folds)
        assert all(f.gap_bars > f.purge_bars for f in folds)

    def test_a_fold_with_too_small_a_gap_is_reported_as_leaking(self) -> None:
        """The invariant is checked, not assumed."""
        series = self._series(600)
        bad = Fold(
            index=0,
            train=DataWindow.of(series, 0, 300),
            test=DataWindow.of(series, 305, 600),
            purge_bars=24,
            embargo_bars=0,
            scheme=FoldScheme.EXPANDING,
        )
        assert bad.gap_bars == 5
        assert bad.leaks()

    def test_overlapping_windows_are_reported_as_leaking(self) -> None:
        series = self._series(600)
        bad = Fold(
            index=0,
            train=DataWindow.of(series, 0, 400),
            test=DataWindow.of(series, 350, 600),
            purge_bars=0,
            embargo_bars=0,
            scheme=FoldScheme.EXPANDING,
        )
        assert bad.leaks()

    def test_expanding_training_windows_grow(self) -> None:
        folds = generate_folds(self._series(), horizon_bars=12, folds=4)
        sizes = [f.train.bars for f in folds]
        assert sizes == sorted(sizes)
        assert all(f.train.start_index == 0 for f in folds)

    def test_rolling_training_windows_move(self) -> None:
        folds = generate_folds(
            self._series(), horizon_bars=12, folds=4, scheme=FoldScheme.ROLLING
        )
        starts = [f.train.start_index for f in folds]
        assert starts == sorted(starts)
        assert starts[-1] > starts[0]

    def test_windows_record_their_exact_range(self) -> None:
        """Leakage has to be auditable after the fact, not re-derived from the code."""
        series = self._series()
        folds = generate_folds(series, horizon_bars=12, folds=3)
        for fold in folds:
            assert fold.train.start_time == series[fold.train.start_index].open_time
            assert fold.train.end_time == series[fold.train.end_index - 1].close_time
            assert fold.test.bars == fold.test.end_index - fold.test.start_index

    def test_the_last_fold_reaches_the_end_of_history(self) -> None:
        series = self._series()
        folds = generate_folds(series, horizon_bars=12, folds=3)
        assert folds[-1].test.end_index == len(series)

    def test_too_little_history_produces_no_folds(self) -> None:
        """Refusing is correct; degenerate folds would produce numbers nobody can read."""
        assert generate_folds(self._series(420), horizon_bars=12, folds=5) == []
        assert generate_folds(self._series(1600), horizon_bars=12, folds=0) == []


# ------------------------------------------------------------------ universe


class TestHistoricalUniverse:
    @staticmethod
    def _universe() -> HistoricalUniverse:
        universe = HistoricalUniverse.from_symbols(["BTC", "ETH"])
        universe.add(
            AssetListing(
                symbol="DEAD",
                listed_at=FIXED_NOW - timedelta(days=400),
                delisted_at=FIXED_NOW - timedelta(days=100),
                reason="delisted from every tracked venue",
            )
        )
        universe.add(AssetListing(symbol="NEW", listed_at=FIXED_NOW - timedelta(days=30)))
        return universe

    def test_a_delisted_asset_is_active_before_it_died(self) -> None:
        universe = self._universe()
        assert "DEAD" in universe.active_at(FIXED_NOW - timedelta(days=200))
        assert "DEAD" not in universe.active_at(FIXED_NOW)

    def test_an_asset_is_not_active_before_it_listed(self) -> None:
        universe = self._universe()
        assert "NEW" not in universe.active_at(FIXED_NOW - timedelta(days=200))
        assert "NEW" in universe.active_at(FIXED_NOW)

    def test_the_survivor_universe_silently_drops_the_dead(self) -> None:
        """The bias, quantified rather than described."""
        gap = self._universe().survivorship_gap(FIXED_NOW - timedelta(days=200))
        assert gap.missing == ("DEAD",)
        assert gap.spurious == ("NEW",)
        assert gap.bias_fraction == pytest.approx(1 / 3, abs=1e-4)
        assert not gap.is_clean
        assert "DEAD" in gap.summary()

    def test_a_universe_with_no_delistings_reports_a_clean_gap(self) -> None:
        """The current state of this repo, asserted so it stops being true loudly."""
        universe = HistoricalUniverse.from_symbols(["BTC", "ETH", "SOL"])
        gap = universe.survivorship_gap(FIXED_NOW)
        assert gap.is_clean
        assert gap.bias_fraction == 0.0
        assert universe.delisted() == ()

    def test_adding_a_symbol_twice_replaces_rather_than_duplicates(self) -> None:
        universe = HistoricalUniverse.from_symbols(["BTC"])
        universe.add(AssetListing(symbol="BTC", delisted_at=FIXED_NOW))
        assert len(universe.listings) == 1
        assert universe.survivors() == ()

    def test_an_audit_covers_every_fold_boundary(self) -> None:
        universe = self._universe()
        moments = [FIXED_NOW - timedelta(days=d) for d in (300, 200, 50)]
        gaps = universe.audit(moments)
        assert [g.is_clean for g in gaps] == [False, False, False]


# ------------------------------------------------------------------- harness


class TestHarness:
    @staticmethod
    def _harness(**kwargs) -> WalkForwardHarness:
        return WalkForwardHarness(
            folds=3, warmup_bars=400, probe=LeakageProbe(max_points=5), **kwargs
        )

    def test_a_leaky_pipeline_excludes_every_model_from_the_results(self) -> None:
        """The harness refuses to report numbers it cannot trust."""
        report = self._harness().run(
            [_PriceModel(), _ConfidenceOnlyModel()], _source(1400, leaky=True), HORIZON
        )
        assert sorted(report.excluded) == ["confidence_only", "price"]
        assert report.passing_any_fold() == set()

    def test_a_clean_pipeline_keeps_its_models(self) -> None:
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        assert report.excluded == []
        assert report.folds
        assert report.total_test_points > 0

    def test_an_oracle_is_excluded_by_the_skill_screen(self) -> None:
        """Caught by the second mechanism, since the first structurally cannot see it."""
        source = ContextSource("BTC", HOUR, candles(drifting(1400)))
        pairs = build_contexts(source, HORIZON, warmup=400, stride=1)
        truth = {c.as_of: Outcome.classify(r, c.threshold_pct) for c, r in pairs}
        report = self._harness().run([_OracleModel(truth), _PriceModel()], source, HORIZON)
        assert "oracle" in report.excluded
        assert "price" not in report.excluded

    def test_results_are_reported_per_fold_and_per_regime(self) -> None:
        """A single blended accuracy number is not accepted — the Phase 8 gate."""
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        assert len(report.folds) >= 2
        regimes = {s.regime for f in report.folds for s in f.scores}
        assert regimes - {"all"}

    def test_every_fold_records_the_window_it_was_fitted_on(self) -> None:
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        for result in report.folds:
            assert result.train_window.bars > 0
            assert result.test_window.bars > 0
            assert result.train_window.end_index <= result.test_window.start_index
            assert not result.fold.leaks()

    def test_passing_every_fold_is_stricter_than_passing_one(self) -> None:
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        assert report.passing_in_every_fold() <= report.passing_any_fold()

    def test_stability_reports_the_spread_not_just_the_mean(self) -> None:
        """A model that swings between folds has found an era, not an edge."""
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        stability = report.stability()
        assert "price" in stability
        average, spread = stability["price"]
        assert spread >= 0.0
        assert isinstance(average, float)

    def test_too_little_history_produces_an_empty_report_not_a_crash(self) -> None:
        report = self._harness().run([_PriceModel()], _source(420), HORIZON)
        assert report.folds == []
        assert report.total_test_points == 0

    def test_models_that_all_abstain_are_flagged_as_not_independent(self) -> None:
        """Four abstaining models are one result reported four times."""
        quiet = [_InertModel(), _AnotherInertModel()]
        report = self._harness().run([*quiet, _PriceModel()], _source(1400), HORIZON)
        groups = report.identical_series()
        assert groups
        assert any(set(g) == {"inert", "inert_too"} for g in groups)

    def test_distinct_models_are_not_flagged_as_identical(self) -> None:
        report = self._harness().run([_PriceModel()], _source(1400), HORIZON)
        assert report.identical_series() == []

    def test_the_first_fold_trains_on_predictable_bars(self) -> None:
        """Fold zero must not train on a window where every model abstains."""
        report = self._harness().run([_PriceModel()], _source(1600), HORIZON)
        assert report.folds
        first = report.folds[0]
        assert first.train_points > 0
        assert first.train_window.end_index > 400

    def test_the_probe_can_be_disabled_but_defaults_on(self) -> None:
        report = self._harness(run_probe=False).run(
            [_PriceModel()], _source(1400, leaky=True), HORIZON
        )
        assert report.leakage == {}
        assert "price" not in report.excluded
