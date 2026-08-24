"""Detecting look-ahead bias by experiment rather than by inspection.

Every other defence in this repository against look-ahead is *structural*: the forming
bar is stored as `is_final = false`, contexts are built from `candles[:i + 1]`, models
receive a context rather than a database handle. Structural defences are the right
first line, but they share a weakness — they are arguments, and an argument can be
wrong in a way that reading the code will not reveal. A feature history that forgot to
filter by `as_of`, a peer series copied whole, a model holding a reference to the full
frame: each of those defeats the structure while looking correct.

So this module tests the claim instead of asserting it.

**The probe.** Take a prediction point. Build the context normally. Then build a second
context from history in which *everything strictly after the prediction instant has
been replaced with something wildly different* — prices tripled, direction reversed —
and run the model again. A model that cannot see the future must produce a bit-identical
prediction, because from its side nothing changed. If the two predictions differ, the
model read data it should not have, and the difference is proof rather than suspicion.

**The control, which matters as much.** A model that ignores its inputs entirely — one
that abstains, or returns a constant — also passes the future test, trivially and
uninformatively. So the probe also perturbs the *past* and requires the prediction to
change. If it does not, the verdict is `INCONCLUSIVE`, not `CLEAN`. A leakage detector
that reports "clean" for a model it cannot actually test is worse than no detector,
because it launders ignorance into assurance.

Three verdicts, therefore: `LEAKING`, `CLEAN`, `INCONCLUSIVE`.

**What the probe cannot see, stated up front.** Perturbation tests the pipeline —
source to context to prediction. A model that smuggles in its own handle on the full
price series, rather than reading the future through the context it was given, is
reading data the probe never touched, and will pass unmoved. That limitation is
structural and cannot be patched by perturbing harder.

So there is a second, independent screen for exactly that class:
:func:`implausible_skill`. On hourly crypto, a Brier skill above roughly 0.25 against
climatology is not a discovery, it is a bug — the honest ceiling for genuine skill on
this data is perhaps a tenth of that. The screen is a heuristic and says so; unlike the
perturbation probe it cannot prove a leak, only refuse to accept a number that no
correct model could produce. Together the two cover the realistic cases: the probe
proves pipeline leaks, the screen catches the ones it structurally cannot reach.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from mie.core.logging import get_logger
from mie.core.types import Candle
from mie.models.base import PredictionContext, Predictor
from mie.models.runner import ContextSource
from mie.models.types import Horizon, Prediction

log = get_logger(__name__)

__all__ = [
    "LeakageProbe",
    "LeakageReport",
    "PointVerdict",
    "SkillScreen",
    "Verdict",
    "corrupt_after",
    "corrupt_before",
    "implausible_skill",
]

#: Probability mass difference below which two predictions count as identical. Not
#: zero, because a model may legitimately involve floating-point summation whose order
#: is not guaranteed; well below any difference a real leak would produce.
_IDENTICAL_TOLERANCE = 1e-9


class Verdict(StrEnum):
    """What the probe concluded about one model."""

    CLEAN = "clean"
    LEAKING = "leaking"
    #: The model's output did not respond to its own inputs, so the future test proves
    #: nothing about it.
    INCONCLUSIVE = "inconclusive"
    #: Skill so far beyond what this data can support that a leak the probe cannot see
    #: is the likeliest explanation.
    SUSPICIOUS = "suspicious"


#: Brier skill against climatology above which a result on hourly crypto is treated as
#: evidence of a bug rather than of skill. Set an order of magnitude above anything
#: genuinely plausible here — the best measured across 2,032 slices was +0.053, on 68
#: points, and did not reach significance — so this flags impossibility, not merely a
#: good result.
_IMPLAUSIBLE_SKILL = 0.25


def _scale(candle: Candle, factor: float, flip: bool) -> Candle:
    """Replace a bar with an implausible one, preserving structural validity.

    The corruption has to survive the same validation real data does — `high >= low`,
    positive prices — or the probe would be testing the validator rather than the
    model. Both a large scale change and a direction flip are applied, since a model
    keying on direction alone would be unmoved by scaling and vice versa.
    """
    open_ = candle.open * factor
    close = candle.close * factor
    if flip:
        # Reflect the bar's move around its open, so an up bar becomes a down bar of
        # the same size.
        close = open_ * 2 - close
        if close <= 0:
            close = open_ * 0.5
    high = max(open_, close) * 1.01
    low = min(open_, close) * 0.99
    return candle.model_copy(
        update={
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": candle.volume * factor,
            "quote_volume": None if candle.quote_volume is None else candle.quote_volume * factor,
        }
    )


def corrupt_after(
    candles: Sequence[Candle], boundary: datetime, factor: float = 3.0, flip: bool = True
) -> list[Candle]:
    """Replace every bar closing strictly after ``boundary`` with an implausible one."""
    return [
        _scale(c, factor, flip) if c.close_time > boundary else c
        for c in candles
    ]


def corrupt_before(
    candles: Sequence[Candle], boundary: datetime, factor: float = 3.0, flip: bool = True
) -> list[Candle]:
    """Replace every bar closing at or before ``boundary`` with an implausible one.

    The control condition. A model that survives this unchanged is not reading its
    inputs, and the future test tells us nothing about it.
    """
    return [
        _scale(c, factor, flip) if c.close_time <= boundary else c
        for c in candles
    ]


@dataclass(frozen=True, slots=True)
class PointVerdict:
    """The probe's result at one prediction point."""

    as_of: datetime
    #: Total absolute probability change when the future was corrupted. Must be zero.
    future_response: float
    #: Total absolute probability change when the past was corrupted. Must not be zero.
    past_response: float
    baseline_abstained: bool

    @property
    def leaked(self) -> bool:
        return self.future_response > _IDENTICAL_TOLERANCE

    @property
    def responsive(self) -> bool:
        return self.past_response > _IDENTICAL_TOLERANCE

    def __str__(self) -> str:  # pragma: no cover - display affordance
        return (
            f"{self.as_of:%Y-%m-%d %H:%M} future={self.future_response:.2e} "
            f"past={self.past_response:.2e} "
            f"{'LEAK' if self.leaked else 'ok' if self.responsive else 'inert'}"
        )


@dataclass(slots=True)
class LeakageReport:
    """What the probe found for one model."""

    model_id: str
    points: list[PointVerdict] = field(default_factory=list)

    @property
    def tested(self) -> int:
        return len(self.points)

    @property
    def leaking_points(self) -> list[PointVerdict]:
        return [p for p in self.points if p.leaked]

    @property
    def responsive_points(self) -> int:
        return sum(1 for p in self.points if p.responsive)

    @property
    def verdict(self) -> Verdict:
        """Leaking beats inconclusive beats clean.

        Order matters: a model that leaks at one point out of a thousand is leaking,
        full stop. Look-ahead is not a rate to be tolerated — one contaminated
        prediction means the evaluation of every prediction is in question.
        """
        if self.leaking_points:
            return Verdict.LEAKING
        if self.responsive_points == 0:
            return Verdict.INCONCLUSIVE
        return Verdict.CLEAN

    @property
    def worst_response(self) -> float:
        return max((p.future_response for p in self.points), default=0.0)

    def summary(self) -> str:
        verdict = self.verdict
        if verdict is Verdict.LEAKING:
            return (
                f"{self.model_id}: LEAKING - {len(self.leaking_points)} of {self.tested} "
                f"points changed when only the future changed "
                f"(worst {self.worst_response:.4f})"
            )
        if verdict is Verdict.INCONCLUSIVE:
            return (
                f"{self.model_id}: INCONCLUSIVE - output never responded to its own "
                f"inputs across {self.tested} points, so the future test proves nothing"
            )
        return (
            f"{self.model_id}: clean - {self.tested} points, "
            f"{self.responsive_points} responsive to past data, none to future data"
        )


class LeakageProbe:
    """Runs the corruption experiment over a model and a slice of history."""

    def __init__(
        self,
        factor: float = 3.0,
        flip: bool = True,
        max_points: int = 25,
    ) -> None:
        self.factor = factor
        self.flip = flip
        self.max_points = max_points

    def probe(
        self,
        model: Predictor,
        source: ContextSource,
        horizon: Horizon,
        indices: Sequence[int] | None = None,
    ) -> LeakageReport:
        """Probe ``model`` at the given bar indices, or at evenly spaced points."""
        report = LeakageReport(model_id=model.model_id)
        chosen = list(indices) if indices is not None else self._spread(source, horizon)

        for index in chosen:
            context = source.context_at(index, horizon)
            if context is None:
                continue
            baseline = _safe_predict(model, context)
            if baseline is None:
                continue

            future = self._rebuild(source, context.as_of, corrupt_after)
            past = self._rebuild(source, context.as_of, corrupt_before)

            future_context = future.context_at(index, horizon)
            past_context = past.context_at(index, horizon)
            if future_context is None or past_context is None:
                continue

            future_prediction = _safe_predict(model, future_context)
            past_prediction = _safe_predict(model, past_context)
            if future_prediction is None or past_prediction is None:
                continue

            report.points.append(
                PointVerdict(
                    as_of=context.as_of,
                    future_response=_distance(baseline, future_prediction),
                    past_response=_distance(baseline, past_prediction),
                    baseline_abstained=baseline.confidence <= 0.0,
                )
            )

        if report.verdict is Verdict.LEAKING:
            log.error(
                "leakage_detected",
                model=model.model_id,
                points=len(report.leaking_points),
                worst=report.worst_response,
            )
        return report

    # ------------------------------------------------------------------ internals

    def _spread(self, source: ContextSource, horizon: Horizon) -> list[int]:
        """Evenly spaced prediction points across the usable range.

        Spread rather than consecutive: a leak that only manifests in one regime would
        be missed by twenty-five adjacent bars, and the probe is expensive enough that
        testing every point is not an option.
        """
        warmup = 400
        last = len(source.candles) - horizon.bars - 1
        if last <= warmup:
            warmup = max(1, len(source.candles) // 2)
            last = len(source.candles) - 1
        if last <= warmup:
            return []
        count = min(self.max_points, last - warmup)
        if count <= 0:
            return []
        step = (last - warmup) / count
        return [int(warmup + i * step) for i in range(count)]

    def _rebuild(
        self,
        source: ContextSource,
        boundary: datetime,
        corrupt: Callable[..., list[Candle]],
    ) -> ContextSource:
        """A copy of the source with candles corrupted on one side of ``boundary``.

        Peers are corrupted too. A cross-asset model reading an uncorrupted peer series
        would otherwise slip through the future test, and cross-asset data is one of
        the easier places for a filter to be forgotten.

        The rebuilt source keeps the *original's type*. Constructing a plain
        :class:`ContextSource` here would silently discard whatever context-building
        behaviour a subclass has — including, in the case that matters, the leak being
        looked for. The probe would then compare a leaky context against a correct one
        and report a difference that says nothing about the model. Caught by a test
        asserting the control condition, which is the whole reason the control exists.
        """
        return type(source)(
            asset=source.asset,
            timeframe=source.timeframe,
            candles=corrupt(source.candles, boundary, self.factor, self.flip),
            feature_history=_corrupt_features(source.feature_history, boundary, corrupt),
            peers={
                name: corrupt(series, boundary, self.factor, self.flip)
                for name, series in source.peers.items()
            },
            funding=_corrupt_series(source.funding, boundary, corrupt),
            open_interest=_corrupt_series(source.open_interest, boundary, corrupt),
            news=source.news,
            data_quality=source.data_quality,
        )


def _corrupt_features(
    history: Sequence[tuple[datetime, Mapping[str, float]]],
    boundary: datetime,
    corrupt: Callable[..., list[Candle]],
) -> list[tuple[datetime, Mapping[str, float]]]:
    """Scale numeric feature values on the corrupted side of the boundary."""
    after = corrupt is corrupt_after
    out: list[tuple[datetime, Mapping[str, float]]] = []
    for moment, values in history:
        target = moment > boundary if after else moment <= boundary
        if not target:
            out.append((moment, values))
            continue
        out.append((moment, {k: -v * 3.0 for k, v in values.items()}))
    return out


def _corrupt_series(
    series: Sequence[tuple[datetime, float]],
    boundary: datetime,
    corrupt: Callable[..., list[Candle]],
) -> list[tuple[datetime, float]]:
    after = corrupt is corrupt_after
    return [
        (moment, -value * 3.0 if (moment > boundary if after else moment <= boundary) else value)
        for moment, value in series
    ]


def _distance(left: Prediction, right: Prediction) -> float:
    """Total absolute difference between two predictions' probability mass.

    Confidence is compared too. A model could in principle leak through its confidence
    without moving a single probability — reading the future to decide how sure it is
    — and that is still look-ahead.
    """
    return (
        abs(left.distribution.up - right.distribution.up)
        + abs(left.distribution.flat - right.distribution.flat)
        + abs(left.distribution.down - right.distribution.down)
        + abs(left.confidence - right.confidence)
    )


def _safe_predict(model: Predictor, context: PredictionContext) -> Prediction | None:
    try:
        return model.predict(context)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("probe_predict_failed", model=model.model_id, error=str(exc)[:200])
        return None


@dataclass(frozen=True, slots=True)
class SkillScreen:
    """The result of screening one model's measured skill for impossibility."""

    model_id: str
    best_skill: float
    threshold: float
    points: int

    @property
    def suspicious(self) -> bool:
        return self.best_skill > self.threshold

    @property
    def verdict(self) -> Verdict:
        return Verdict.SUSPICIOUS if self.suspicious else Verdict.CLEAN

    def summary(self) -> str:
        if not self.suspicious:
            return f"{self.model_id}: skill {self.best_skill:+.4f}, within plausible range"
        return (
            f"{self.model_id}: SUSPICIOUS - skill {self.best_skill:+.4f} over "
            f"{self.points} points exceeds {self.threshold:.2f}, which no correct model "
            f"achieves on this data. Treat as a leak the perturbation probe cannot see."
        )


def implausible_skill(
    scores: Sequence[object], threshold: float = _IMPLAUSIBLE_SKILL
) -> list[SkillScreen]:
    """Screen measured skill for results too good to be real.

    The backstop for the one thing perturbation cannot reach: a model reading the
    future through a channel outside the context it was handed. This cannot *prove*
    a leak — a genuinely revolutionary model would also trip it — but on this data,
    where 2,032 measured slices produced a best of +0.053 and nothing significant, a
    figure four times larger is a bug until shown otherwise.

    Accepts anything exposing ``model_id``, ``skill``, ``regime`` and ``predictions``,
    which in practice is :class:`~mie.models.evaluation.ModelScore`.
    """
    best: dict[str, tuple[float, int]] = {}
    for score in scores:
        if getattr(score, "regime", None) != "all":
            continue
        model_id = getattr(score, "model_id", "")
        skill = float(getattr(score, "skill", 0.0))
        points = int(getattr(score, "predictions", 0))
        current = best.get(model_id)
        if current is None or skill > current[0]:
            best[model_id] = (skill, points)

    screens = [
        SkillScreen(model_id=model_id, best_skill=skill, threshold=threshold, points=points)
        for model_id, (skill, points) in sorted(best.items())
    ]
    for screen in screens:
        if screen.suspicious:
            log.error(
                "implausible_skill",
                model=screen.model_id,
                skill=screen.best_skill,
                threshold=threshold,
            )
    return screens
