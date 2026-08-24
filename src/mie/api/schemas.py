"""The response contract — and the place Phase 10's gate is enforced.

The gate reads: *no screen can display a directional call without its confidence and
invalidation conditions visible in the same view.* Enforced in a stylesheet or a
template, that is a convention, and conventions are one refactor away from being false.
So it is enforced here instead, in the type system.

There are exactly two shapes a prediction response can take:

* :class:`DirectionalCall` — carries a direction, and **cannot be constructed** without
  a non-zero confidence, a confidence decomposition, and at least one invalidation
  condition. A caller that tries gets a validation error, not a half-populated object.
* :class:`InsufficientEvidence` — carries no direction at all, and requires at least
  one reason. It is a first-class result, not an error state: on this data it is what
  the system returns almost always, and a UI that renders it as a blank panel or a
  spinner would be misrepresenting a finding as a failure.

They are a discriminated union, so no response can be partly one and partly the other,
and no client can receive a direction without receiving the things that qualify it. The
UI cannot violate the gate because the API cannot express a violation.

Every directional payload also carries ``is_guaranteed: false`` as a literal field.
Belt and braces, deliberately: §21 requires prediction and guarantee to be
unmistakable, and a field that is always present and always false is harder to overlook
than a field that is simply absent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mie.core.timeframes import utcnow

__all__ = [
    "AssetSummary",
    "CalibrationBin",
    "ConfidenceBreakdown",
    "DirectionalCall",
    "EvidenceItem",
    "GateCondition",
    "InsufficientEvidence",
    "ModelPerformance",
    "NewsItem",
    "PredictionResponse",
    "QualitySummary",
    "StateView",
    "SystemStatus",
    "TimeframeState",
]

#: Confidence below which the system will not publish a direction at all. Matches the
#: Phase 7 publication floor; duplicated as a validator here so the contract cannot
#: emit something the ensemble would have suppressed.
_PUBLISH_FLOOR = 0.35


class EvidenceItem(BaseModel):
    """One reason for or against, as shown to a reader."""

    model_config = ConfigDict(frozen=True)

    label: str
    detail: str = ""
    #: Signed contribution on the direction axis, for rendering as a bar.
    contribution: float = 0.0


class ConfidenceBreakdown(BaseModel):
    """Why the confidence is what it is.

    Sent with every directional call so a reader can answer "why only 40%?" without
    asking. A confidence score whose derivation cannot be inspected is an assertion,
    and the UI is required to be able to show the derivation.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    skill: float
    calibration: float
    agreement: float
    data_quality: float
    sample: float
    regime_familiarity: float
    limiting_factor: str
    notes: list[str] = Field(default_factory=list)


class InsufficientEvidence(BaseModel):
    """No directional call, and the specific conditions that were not met.

    A first-class result. On the data this system has measured it is the overwhelmingly
    common one, and it carries information — which condition failed — that a blank
    panel would throw away.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["insufficient_evidence"] = "insufficient_evidence"
    asset: str
    timeframe: str
    horizon: str
    as_of: datetime
    #: Why nothing is being published. Never empty.
    reasons: list[str]
    #: Reported even though nothing is published, so a reader can see how far short it
    #: fell rather than only that it did.
    confidence: ConfidenceBreakdown | None = None
    regime: str = "unknown"
    data_quality: float = 1.0
    #: What the panel would have said, if it had said anything. Shown as context, never
    #: as a call — which is why it lives on the *insufficient* shape and carries no
    #: probabilities.
    panel_summary: str = ""

    @model_validator(mode="after")
    def _require_a_reason(self) -> InsufficientEvidence:
        if not [r for r in self.reasons if r.strip()]:
            raise ValueError(
                "insufficient evidence must say why; an empty reason list renders as a "
                "blank panel and tells a reader nothing"
            )
        return self


class DirectionalCall(BaseModel):
    """A published directional view, with everything that qualifies it attached.

    Cannot be constructed without confidence and invalidation conditions. That is the
    Phase 10 gate, implemented where it cannot be bypassed rather than where it can.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["directional_call"] = "directional_call"
    asset: str
    timeframe: str
    horizon: str
    as_of: datetime
    resolves_at: datetime

    direction: Literal["up", "flat", "down"]
    probability_up: float
    probability_flat: float
    probability_down: float
    #: How far from an even up/down split, in [-1, 1].
    directional_edge: float

    confidence: float
    confidence_breakdown: ConfidenceBreakdown
    #: Concrete conditions that would falsify this call. Never empty.
    invalidation: list[str]

    evidence: list[EvidenceItem] = Field(default_factory=list)
    counter_evidence: list[EvidenceItem] = Field(default_factory=list)
    expected_move_pct: float = 0.0
    expected_volatility_pct: float = 0.0
    move_threshold_pct: float = 0.0
    regime: str = "unknown"
    data_quality: float = 1.0
    is_super_prediction: bool = False
    #: Always false. Present rather than absent so the distinction §21 requires is
    #: impossible to overlook in a payload.
    is_guaranteed: Literal[False] = False

    @model_validator(mode="after")
    def _enforce_the_display_gate(self) -> DirectionalCall:
        if not [c for c in self.invalidation if c.strip()]:
            raise ValueError(
                "a directional call requires at least one invalidation condition: a "
                "forecast that cannot be wrong cannot be evaluated, and the UI is "
                "required to render the conditions alongside the direction"
            )
        if self.confidence < _PUBLISH_FLOOR:
            raise ValueError(
                f"confidence {self.confidence:.2f} is below the publication floor "
                f"{_PUBLISH_FLOOR}; below it the correct response is "
                f"InsufficientEvidence, not a quiet directional call"
            )
        if abs(self.confidence - self.confidence_breakdown.value) > 1e-6:
            raise ValueError(
                "published confidence must equal its own decomposition, or the "
                "breakdown shown to a reader explains a different number"
            )
        total = self.probability_up + self.probability_flat + self.probability_down
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        return self

    def headline(self) -> str:
        """One line carrying direction, probability, confidence and falsifiability.

        Used by the terminal renderer and by the dashboard's collapsed view. Written as
        a single string so the four cannot drift into separate templates and then into
        separate screens.
        """
        return (
            f"{self.asset} {self.horizon}: {self.direction.upper()} "
            f"{self.probability_up:.0%}/{self.probability_flat:.0%}/"
            f"{self.probability_down:.0%} | confidence {self.confidence:.0%} "
            f"| invalidated by: {self.invalidation[0]}"
        )


#: The only two things a prediction endpoint may return.
PredictionResponse = Annotated[
    DirectionalCall | InsufficientEvidence, Field(discriminator="kind")
]


class TimeframeState(BaseModel):
    """Market state on one timeframe."""

    model_config = ConfigDict(frozen=True)

    timeframe: str
    direction: str
    strength: float
    confidence: float
    volatility: str = "unknown"
    momentum: float = 0.0


class StateView(BaseModel):
    """The hierarchical multi-timeframe state for one asset."""

    model_config = ConfigDict(frozen=True)

    asset: str
    as_of: datetime
    alignment: str
    bias_score: float
    agreement: float
    regime: str
    timeframes: list[TimeframeState] = Field(default_factory=list)
    #: Present when the hierarchy disagrees with itself, which is information rather
    #: than a problem to be smoothed away.
    conflict: str = ""


class AssetSummary(BaseModel):
    """One row of the asset grid."""

    model_config = ConfigDict(frozen=True)

    asset: str
    price: float = 0.0
    change_24h_pct: float = 0.0
    volatility_pct: float = 0.0
    regime: str = "unknown"
    data_quality: float = 1.0
    bars_stored: int = 0
    last_bar: datetime | None = None


class ModelPerformance(BaseModel):
    """How one model has actually done, sliced.

    ``uniform_brier`` is carried alongside so a reader can see at a glance that a model
    scoring 0.6667 has done exactly as well as saying nothing.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    dimension: str
    value: str
    outcomes: int
    brier: float
    accuracy: float
    log_loss: float
    weight: float = 0.0
    has_evidence: bool = True
    uniform_brier: float = 2 / 3

    @property
    def beats_saying_nothing(self) -> bool:
        return self.brier < self.uniform_brier


class CalibrationBin(BaseModel):
    """One bucket of a reliability diagram."""

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    count: int
    stated: float
    observed: float
    interval_low: float
    interval_high: float
    consistent: bool


class NewsItem(BaseModel):
    """One deduplicated story."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str = ""
    published_at: datetime
    category: str = "other"
    sentiment: str = "neutral"
    importance: float = 0.0
    coverage: int = 1
    assets: list[str] = Field(default_factory=list)


class QualitySummary(BaseModel):
    """Trust in one stored series."""

    model_config = ConfigDict(frozen=True)

    source: str
    asset: str
    timeframe: str
    score: float
    events_24h: int = 0
    stale_seconds: float = 0.0


class GateCondition(BaseModel):
    """One super-prediction condition, with its numbers."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str


class SystemStatus(BaseModel):
    """What the system is, and what it currently claims.

    ``publishes_predictions`` is deliberately part of the status payload. A dashboard
    showing an empty predictions panel is ambiguous between "still loading", "broken"
    and "the system has measured that it has nothing to say" — and only the third is
    true here.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"] = "ok"
    version: str = "0.1.0"
    served_at: datetime = Field(default_factory=utcnow)
    assets_tracked: int = 0
    bars_stored: int = 0
    predictions_recorded: int = 0
    outcomes_resolved: int = 0
    models_with_weight: int = 0
    publishes_predictions: bool = False
    headline: str = ""
    #: Never absent. The system is analytical and does not execute trades; the UI is
    #: required to say so, so the API states it in every status response.
    executes_trades: Literal[False] = False
