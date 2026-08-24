"""The walk-forward harness: fit on one window, predict on the next, never overlap.

Phase 6 answered "does this model have skill?" over one long sweep. That question has
a hidden assumption — that whatever the models learned was learned before the point it
was applied to. For the eight rule-based predictors that is trivially true, because
they fit nothing. But Phase 7 introduced two things that *are* fitted: calibration
curves and skill weights. Those were derived from the same run they were then used in,
which is fine for measuring the machinery and useless as an estimate of live
performance.

This module fixes that. Each fold fits calibration and skill weights on its training
window *only*, then applies them to a test window that begins after a purge and an
embargo. The fitted artefacts carry the instant they were fitted through, and Phase 7's
guard raises if one is ever applied to a point at or before it — so the separation is
enforced by the same mechanism that already caught a leak in my own sweep script,
rather than by this module promising to be careful.

Two refusals are built in, because a harness that reports numbers for unsound inputs is
worse than one that reports nothing:

* **A fold whose own construction is unsound is never run.** `Fold.leaks()` checks that
  the gap covers the purge and the windows do not overlap.
* **A model caught leaking is excluded from the results**, not annotated in them. Its
  scores would be meaningless, and a meaningless number placed next to meaningful ones
  will eventually be read as if it were meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean

from mie.backtest.leakage import (
    LeakageProbe,
    LeakageReport,
    SkillScreen,
    Verdict,
    implausible_skill,
)
from mie.backtest.windows import DataWindow, Fold, FoldScheme, generate_folds
from mie.core.logging import get_logger
from mie.ensemble.calibration import CalibrationLibrary
from mie.ensemble.meta import EnsembleModel, SkillWeights
from mie.models.base import PredictionContext, Predictor
from mie.models.baselines import ClimatologyBaseline
from mie.models.evaluation import ModelScore, WalkForwardEvaluator
from mie.models.runner import ContextSource
from mie.models.types import Horizon

log = get_logger(__name__)

__all__ = ["BacktestReport", "FoldResult", "WalkForwardHarness"]


@dataclass(slots=True)
class FoldResult:
    """What one fold produced, including exactly what it was allowed to see."""

    fold: Fold
    train_points: int
    test_points: int
    #: Scores on the test window, sliced by regime, against the baseline.
    scores: list[ModelScore] = field(default_factory=list)
    #: Models that earned a non-zero weight *from the training window alone*.
    trained_weights: SkillWeights = field(default_factory=SkillWeights)
    #: Calibration records fitted on the training window that proved usable.
    usable_calibrations: int = 0
    fitted_calibrations: int = 0
    #: Ensemble behaviour on the test window.
    ensemble_published: int = 0
    #: Points where a train-fitted calibration could not legitimately be applied.
    calibration_not_applicable: int = 0

    @property
    def train_window(self) -> DataWindow:
        return self.fold.train

    @property
    def test_window(self) -> DataWindow:
        return self.fold.test

    def passing(self) -> set[str]:
        return {s.model_id for s in self.scores if s.beats_baseline}

    def overall(self) -> list[ModelScore]:
        return [s for s in self.scores if s.regime == "all"]

    def summary(self) -> str:
        passing = sorted(self.passing())
        return (
            f"{self.fold.summary()} | train pts {self.train_points} "
            f"test pts {self.test_points} | weights "
            f"{len(self.trained_weights.skilled_models())} | calib "
            f"{self.usable_calibrations}/{self.fitted_calibrations} | "
            f"published {self.ensemble_published} | pass: {', '.join(passing) or 'none'}"
        )


@dataclass(slots=True)
class BacktestReport:
    """Every fold, plus the leakage verdicts that decide whose numbers count."""

    asset: str
    horizon: Horizon
    scheme: FoldScheme
    folds: list[FoldResult] = field(default_factory=list)
    leakage: dict[str, LeakageReport] = field(default_factory=dict)
    #: Models excluded from the results because the probe caught them leaking.
    excluded: list[str] = field(default_factory=list)
    #: Models the probe could not test, because their output never moved.
    untestable: list[str] = field(default_factory=list)
    #: Skill screens run on the measured results, catching leaks the probe cannot see.
    screens: list[SkillScreen] = field(default_factory=list)

    @property
    def suspicious(self) -> list[str]:
        return sorted({s.model_id for s in self.screens if s.suspicious})

    @property
    def total_test_points(self) -> int:
        return sum(f.test_points for f in self.folds)

    def passing_in_every_fold(self) -> set[str]:
        """Models that beat the baseline in *every* fold.

        The right question for a walk-forward run. Passing one fold out of five is what
        a model with no skill does roughly one time in five, and reporting the union of
        per-fold winners would turn that into a list of successes.
        """
        if not self.folds:
            return set()
        sets = [f.passing() for f in self.folds]
        return set.intersection(*sets) if sets else set()

    def passing_any_fold(self) -> set[str]:
        return {m for f in self.folds for m in f.passing()}

    def identical_series(self) -> list[tuple[str, ...]]:
        """Groups of models whose per-fold skill is bit-identical.

        Two models declared independent that produce the same number in every fold are
        not two opinions. In practice this catches models that abstain: several
        emitting a uniform distribution score identically, and their apparent agreement
        inside the ensemble would be an artefact of all of them having nothing to say.
        Surfaced explicitly because a table of per-fold skill only reveals it if
        somebody happens to read across the rows and notice.
        """
        grouped: dict[tuple[float, ...], list[str]] = defaultdict(list)
        for model_id, values in self.skill_by_model().items():
            grouped[tuple(values)].append(model_id)
        return [tuple(sorted(names)) for names in grouped.values() if len(names) > 1]

    def skill_by_model(self) -> dict[str, list[float]]:
        """Overall skill per model, one entry per fold, in fold order."""
        series: dict[str, list[float]] = defaultdict(list)
        for result in self.folds:
            for score in result.overall():
                series[score.model_id].append(score.skill)
        return dict(series)

    def stability(self) -> dict[str, tuple[float, float]]:
        """Mean and spread of each model's per-fold skill.

        A model whose skill swings from +0.05 to −0.05 across folds has not found
        anything; it has found one era. The spread is the part worth looking at.
        """
        out: dict[str, tuple[float, float]] = {}
        for model_id, values in self.skill_by_model().items():
            if not values:
                continue
            average = mean(values)
            spread = max(values) - min(values)
            out[model_id] = (round(average, 5), round(spread, 5))
        return out

    def report(self) -> str:
        lines = [
            f"Walk-forward backtest: {self.asset} {self.horizon} ({self.scheme})",
            "=" * 78,
        ]
        for result in self.folds:
            lines.append("  " + result.summary())
        lines.append("")
        for _, leak in sorted(self.leakage.items()):
            lines.append("  " + leak.summary())
        lines.append("")
        every = sorted(self.passing_in_every_fold())
        any_fold = sorted(self.passing_any_fold())
        lines.append(f"pass in EVERY fold ({len(every)}): {', '.join(every) or 'none'}")
        lines.append(f"pass in any fold  ({len(any_fold)}): {', '.join(any_fold) or 'none'}")
        for group in self.identical_series():
            lines.append(
                f"identical in every fold: {', '.join(group)} "
                f"- not {len(group)} independent results"
            )
        for screen in self.screens:
            if screen.suspicious:
                lines.append(screen.summary())
        if self.untestable:
            lines.append(f"untestable by perturbation: {', '.join(sorted(self.untestable))}")
        if self.excluded:
            lines.append(f"EXCLUDED for leakage: {', '.join(sorted(self.excluded))}")
        return "\n".join(lines)


class WalkForwardHarness:
    """Runs models across chronological folds and reports per fold and per regime."""

    def __init__(
        self,
        baseline: Predictor | None = None,
        folds: int = 5,
        warmup_bars: int = 400,
        scheme: FoldScheme = FoldScheme.EXPANDING,
        probe: LeakageProbe | None = None,
        run_probe: bool = True,
    ) -> None:
        self.baseline = baseline or ClimatologyBaseline()
        self.fold_count = folds
        self.warmup_bars = warmup_bars
        self.scheme = scheme
        self.probe = probe or LeakageProbe()
        self.run_probe = run_probe

    def run(
        self,
        models: Sequence[Predictor],
        source: ContextSource,
        horizon: Horizon,
        stride: int | None = None,
    ) -> BacktestReport:
        report = BacktestReport(asset=source.asset, horizon=horizon, scheme=self.scheme)
        step = stride or horizon.bars

        candidates = list(models)
        if self.run_probe:
            candidates = self._screen(candidates, source, horizon, report)

        folds = generate_folds(
            source.candles,
            horizon_bars=horizon.bars,
            folds=self.fold_count,
            warmup_bars=self.warmup_bars,
            scheme=self.scheme,
        )
        if not folds:
            log.warning(
                "no_usable_folds",
                asset=source.asset,
                bars=len(source.candles),
                horizon=horizon.bars,
            )
            return report

        for fold in folds:
            report.folds.append(self._run_fold(fold, candidates, source, horizon, step))

        # The screen runs on measured results, so it can only be applied after the
        # folds have been scored — which is also why it is a second line of defence
        # rather than a replacement for the probe.
        report.screens = implausible_skill(
            [score for result in report.folds for score in result.scores]
        )
        for model_id in report.suspicious:
            if model_id not in report.excluded:
                report.excluded.append(model_id)
        return report

    # ------------------------------------------------------------------ internals

    def _screen(
        self,
        models: Sequence[Predictor],
        source: ContextSource,
        horizon: Horizon,
        report: BacktestReport,
    ) -> list[Predictor]:
        """Probe every model and drop the ones caught reading the future."""
        kept: list[Predictor] = []
        for model in models:
            result = self.probe.probe(model, source, horizon)
            report.leakage[model.model_id] = result
            if result.verdict is Verdict.LEAKING:
                report.excluded.append(model.model_id)
                continue
            if result.verdict is Verdict.INCONCLUSIVE:
                # Kept, but recorded: its clean verdict is unearned, and a reader
                # should know the probe had nothing to work with.
                report.untestable.append(model.model_id)
            kept.append(model)
        return kept

    def _points(
        self,
        source: ContextSource,
        horizon: Horizon,
        window: DataWindow,
        step: int,
    ) -> list[tuple[PredictionContext, float]]:
        """Evaluation points whose *prediction instant* lies inside ``window``.

        The context still draws on all history before the point — that is what a live
        system would have. What the window controls is where predictions are made and
        scored, which is what the purge and embargo are protecting.
        """
        candles = source.candles
        last = min(window.end_index, len(candles) - horizon.bars)
        # Never below the warmup: a "prediction" made before a model has enough
        # history is an abstention, and a training window full of abstentions
        # produces a calibration fitted on nothing.
        first = max(window.start_index, self.warmup_bars)
        pairs: list[tuple[PredictionContext, float]] = []
        for index in range(first, last, max(1, step)):
            context = source.context_at(index, horizon)
            if context is None:
                continue
            entry = candles[index].close
            if entry <= 0:
                continue
            realised = (candles[index + horizon.bars].close - entry) / entry * 100.0
            pairs.append((context, realised))
        return pairs

    def _run_fold(
        self,
        fold: Fold,
        models: Sequence[Predictor],
        source: ContextSource,
        horizon: Horizon,
        step: int,
    ) -> FoldResult:
        train_points = self._points(source, horizon, fold.train, step)
        test_points = self._points(source, horizon, fold.test, step)

        result = FoldResult(
            fold=fold, train_points=len(train_points), test_points=len(test_points)
        )
        if not train_points or not test_points:
            return result

        evaluator = WalkForwardEvaluator(self.baseline)

        # --- fit on the training window only -------------------------------
        train_report = evaluator.evaluate(models, train_points)
        result.trained_weights = SkillWeights.from_report(train_report)
        library = CalibrationLibrary()
        fitted = library.fit([e for group in train_report.scored.values() for e in group])
        result.fitted_calibrations = len(fitted)
        result.usable_calibrations = len(library.usable())

        # --- apply to the test window --------------------------------------
        test_report = evaluator.evaluate(models, test_points)
        result.scores = test_report.scores

        ensemble = EnsembleModel(list(models), result.trained_weights, library)
        for context, _ in test_points:
            outcome = ensemble.predict_detailed(context)
            if outcome.published:
                result.ensemble_published += 1
            if outcome.factors.calibration == 0.0 and result.usable_calibrations:
                result.calibration_not_applicable += 1

        log.info(
            "fold_complete",
            fold=fold.index,
            train_bars=fold.train.bars,
            test_bars=fold.test.bars,
            gap=fold.gap_bars,
            weights=len(result.trained_weights.skilled_models()),
            published=result.ensemble_published,
        )
        return result
