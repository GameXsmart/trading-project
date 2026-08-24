"""Reweighting: the part that has to actually change behaviour, or it is theatre.

§14 is blunt about this. Storing predictions is not learning. Computing metrics is not
learning. The loop earns the word only if the numbers it produces change what the
system does next, and only in the places the evidence supports.

So the contract here is narrow and checkable: given resolved outcomes, produce a weight
per (model, asset, timeframe, horizon, regime), and make every change traceable to the
sample that caused it. Two properties follow from that, and both are gates:

* **A model that degrades in one regime is down-weighted in that regime and nowhere
  else.** Skill is not a scalar property of a model. A trend follower that stops
  working in chop has not become worse at trends, and a loop that reacts by lowering
  its weight everywhere has learned something false.
* **Recency without noise-chasing.** Only the most recent outcomes per slice count, so
  degradation is visible rather than diluted by ancient history. But a short window is
  a noisy window, so the measured skill is shrunk toward zero by sample size before it
  becomes a weight.

**A deliberate deviation from the original design.** The specification asks for
shrinkage toward *equal* weights. Applied literally that hands influence to models that
have demonstrated none — with eight models and no skill anywhere, equal weights means
every model gets 12.5% of a vote it has not earned. So shrinkage happens in two stages:
a model must first clear the gate (enough samples, skill above a floor, significance
surviving correction) to receive any weight at all, and only *among those that clear
it* are relative weights blended toward equal. Below the gate the weight is zero, not
small. The prior is "this model has shown nothing", because on this data that is what
almost every model has shown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mie.core.logging import get_logger
from mie.learning.records import ResolvedOutcome
from mie.models.evaluation import _paired_p_value
from mie.patterns.statistics import benjamini_hochberg

log = get_logger(__name__)

__all__ = ["WeightKey", "WeightLearner", "WeightUpdate"]

#: Resolved outcomes needed in a slice before it can move a weight at all.
_MIN_SAMPLES = 40

#: Only the most recent outcomes per slice are counted, so that a model which has
#: stopped working is not propped up by how well it did a year ago.
_RECENT_WINDOW = 400

#: Sample-size shrinkage constant: measured skill is multiplied by n / (n + k). At the
#: gate minimum of 40 samples a slice keeps about 29% of its measured skill; by 400 it
#: keeps 80%. Chosen so that a slice which just cleared the sample gate cannot
#: immediately dominate one with ten times the evidence.
_SHRINKAGE_K = 100

#: How far relative weights among qualifying models are pulled toward equal. Zero would
#: let a marginally better model dominate on a difference that is within noise; one
#: would discard the measurement entirely.
_EQUAL_BLEND = 0.3

#: Skill below this is treated as zero even when significant, matching the Phase 6 and
#: Phase 7 floors. A statistically detectable edge of 0.002 Brier is not a usable one.
_MIN_USABLE_SKILL = 0.01


@dataclass(frozen=True, slots=True)
class WeightKey:
    """The scope a weight applies to."""

    model_id: str
    asset: str
    timeframe: str
    horizon_bars: int
    regime: str

    def label(self) -> str:
        return f"{self.model_id}/{self.asset}/{self.timeframe}+{self.horizon_bars}/{self.regime}"

    def __str__(self) -> str:  # pragma: no cover
        return self.label()


@dataclass(slots=True)
class WeightUpdate:
    """One weight, what it was, and the evidence that moved it."""

    key: WeightKey
    raw_skill: float
    weight: float
    previous_weight: float
    samples: int
    p_value: float
    significant: bool
    baseline_brier: float
    model_brier: float
    computed_at: datetime | None = None

    @property
    def delta(self) -> float:
        return round(self.weight - self.previous_weight, 6)

    @property
    def changed(self) -> bool:
        return abs(self.delta) > 1e-9

    @property
    def direction(self) -> str:
        if not self.changed:
            return "unchanged"
        return "up" if self.delta > 0 else "down"

    @property
    def gated_out(self) -> str:
        """Why this slice earned no weight, if it earned none."""
        if self.weight > 0:
            return ""
        if self.samples < _MIN_SAMPLES:
            return f"only {self.samples} resolved outcomes"
        if self.raw_skill <= _MIN_USABLE_SKILL:
            return f"skill {self.raw_skill:+.4f} at or below the usable floor"
        if not self.significant:
            return f"skill {self.raw_skill:+.4f} not significant (p={self.p_value:.3f})"
        return "shrunk to zero"

    def summary(self) -> str:
        if self.weight > 0:
            return (
                f"{self.key.label()}: weight {self.previous_weight:.4f} -> "
                f"{self.weight:.4f} ({self.delta:+.4f}) "
                f"on {self.samples} outcomes, skill {self.raw_skill:+.4f}"
            )
        return (
            f"{self.key.label()}: no weight ({self.gated_out})"
            + (f", was {self.previous_weight:.4f}" if self.previous_weight > 0 else "")
        )


class WeightLearner:
    """Turns resolved outcomes into weights, and records what moved."""

    def __init__(
        self,
        baseline_model_id: str = "baseline_climatology",
        min_samples: int = _MIN_SAMPLES,
        recent_window: int = _RECENT_WINDOW,
        shrinkage_k: int = _SHRINKAGE_K,
        equal_blend: float = _EQUAL_BLEND,
        false_discovery_rate: float = 0.05,
    ) -> None:
        self.baseline_model_id = baseline_model_id
        self.min_samples = min_samples
        self.recent_window = recent_window
        self.shrinkage_k = shrinkage_k
        self.equal_blend = max(0.0, min(1.0, equal_blend))
        self.false_discovery_rate = false_discovery_rate

    def learn(
        self,
        outcomes: Sequence[ResolvedOutcome],
        previous: Mapping[WeightKey, float] | None = None,
    ) -> list[WeightUpdate]:
        """Compute weights from resolved outcomes, paired against the baseline.

        The baseline is not a formula here — it is a stored forecaster whose predictions
        were resolved by the same code on the same points. Comparing against a baseline
        computed some other way would be comparing against a different question.
        """
        prior = dict(previous or {})
        baseline_index = {
            (o.asset, o.timeframe, o.horizon_bars, o.as_of): o
            for o in outcomes
            if o.model_id == self.baseline_model_id
        }
        if not baseline_index:
            log.warning("no_baseline_outcomes", baseline=self.baseline_model_id)
            return []

        grouped: dict[WeightKey, list[tuple[ResolvedOutcome, ResolvedOutcome]]] = defaultdict(list)
        for outcome in outcomes:
            if outcome.model_id == self.baseline_model_id:
                continue
            reference = baseline_index.get(
                (outcome.asset, outcome.timeframe, outcome.horizon_bars, outcome.as_of)
            )
            if reference is None:
                # No baseline forecast at this point, so there is nothing to compare
                # against. Dropped rather than scored against a different sample.
                continue
            timeframe = str(outcome.timeframe)
            for regime in ("all", outcome.regime):
                grouped[
                    WeightKey(
                        model_id=outcome.model_id,
                        asset=outcome.asset,
                        timeframe=timeframe,
                        horizon_bars=outcome.horizon_bars,
                        regime=regime,
                    )
                ].append((outcome, reference))

        updates = [
            self._score(key, pairs, prior.get(key, 0.0)) for key, pairs in grouped.items()
        ]
        if not updates:
            return []

        # Correct across every slice at once. Per-slice correction would be no
        # correction: the false positives come from the size of the sweep, and this
        # sweep grows with every asset, horizon and regime added.
        flags = benjamini_hochberg([u.p_value for u in updates], self.false_discovery_rate)
        for update, significant in zip(updates, flags, strict=True):
            update.significant = significant

        self._apply_weights(updates)
        return updates

    # ------------------------------------------------------------------ internals

    def _score(
        self,
        key: WeightKey,
        pairs: Sequence[tuple[ResolvedOutcome, ResolvedOutcome]],
        previous: float,
    ) -> WeightUpdate:
        recent = sorted(pairs, key=lambda pair: pair[0].as_of)[-self.recent_window :]
        model_brier = sum(m.brier for m, _ in recent) / len(recent)
        base_brier = sum(b.brier for _, b in recent) / len(recent)
        differences = [b.brier - m.brier for m, b in recent]
        skill = 1.0 - model_brier / base_brier if base_brier > 0 else 0.0
        return WeightUpdate(
            key=key,
            raw_skill=round(skill, 6),
            weight=0.0,
            previous_weight=previous,
            samples=len(recent),
            p_value=round(_paired_p_value(differences), 6),
            significant=False,
            baseline_brier=round(base_brier, 6),
            model_brier=round(model_brier, 6),
            computed_at=max(m.resolved_at for m, _ in recent),
        )

    def _apply_weights(self, updates: Sequence[WeightUpdate]) -> None:
        """Gate, shrink by sample size, then blend qualifiers toward equal."""
        for update in updates:
            qualifies = (
                update.samples >= self.min_samples
                and update.raw_skill > _MIN_USABLE_SKILL
                and update.significant
            )
            if not qualifies:
                update.weight = 0.0
                continue
            update.weight = round(
                update.raw_skill * update.samples / (update.samples + self.shrinkage_k), 6
            )

        # Blend within each scope, never across scopes. Pulling a BTC weight toward an
        # ETH weight would be averaging two different measurements.
        scopes: dict[tuple[str, str, int, str], list[WeightUpdate]] = defaultdict(list)
        for update in updates:
            if update.weight > 0:
                key = update.key
                scopes[(key.asset, key.timeframe, key.horizon_bars, key.regime)].append(update)

        for group in scopes.values():
            if len(group) < 2 or self.equal_blend <= 0:
                continue
            average = sum(u.weight for u in group) / len(group)
            for update in group:
                update.weight = round(
                    update.weight * (1 - self.equal_blend) + average * self.equal_blend, 6
                )


@dataclass(slots=True)
class WeightTable:
    """The current weights, and the changes that produced them."""

    updates: list[WeightUpdate] = field(default_factory=list)

    def active(self) -> list[WeightUpdate]:
        return [u for u in self.updates if u.weight > 0]

    def changes(self) -> list[WeightUpdate]:
        return [u for u in self.updates if u.changed]

    def as_mapping(self) -> dict[WeightKey, float]:
        return {u.key: u.weight for u in self.updates}

    def report(self) -> str:  # pragma: no cover - display affordance
        lines = ["Weight table", "=" * 78]
        changed = self.changes()
        if not changed:
            lines.append("  nothing changed: no slice crossed the evidence gate")
        else:
            lines.extend("  " + u.summary() for u in sorted(changed, key=lambda u: -abs(u.delta)))
        lines.append("")
        lines.append(f"active weights: {len(self.active())} of {len(self.updates)} slices")
        return "\n".join(lines)
