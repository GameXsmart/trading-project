"""What an alert is, and what it is allowed to claim.

An alert is a claim that something is worth a human's attention *right now*. That makes
it the most expensive output the system produces: it interrupts. Everything else in
this repository can be ignored at no cost, so it can afford to be verbose. An alert
cannot.

Two constraints follow, and both are enforced by the types rather than by convention.

**An alert may not say more than the system knows.** After nine phases of measurement,
what survived is volatility clustering — volume spikes and range compression genuinely
precede larger-than-usual movement — and structural facts like regime changes and data
degradation. What did *not* survive is direction. So :class:`AlertKind` is split into
kinds that are `directional` and kinds that are not, and a directional alert carries the
same requirement the API does: it cannot be built without a confidence and an
invalidation condition. On current data no directional alert can fire at all, because
the ensemble publishes nothing — and that is the correct behaviour, not a gap.

**Every alert carries a deduplication key.** Not as an afterthought for the delivery
layer to compute, but as part of the alert's identity, because "is this the same thing
I was told twenty minutes ago" is the question that decides whether an alerting system
is usable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum

from mie.core.timeframes import utcnow

__all__ = ["Alert", "AlertKind", "Severity"]


class Severity(IntEnum):
    """How hard this alert is allowed to push.

    Ordered so budgets can compare them. ``CRITICAL`` is reserved for things that
    invalidate the system's own output — a data feed collapsing, a published call being
    falsified — rather than for market moves, however large. A market move is news; a
    broken feed means everything else on the screen is suspect.
    """

    INFO = 10
    NOTABLE = 20
    IMPORTANT = 30
    CRITICAL = 40

    @property
    def label(self) -> str:
        return self.name.lower()


class AlertKind(StrEnum):
    """The rules this system is willing to alert on."""

    # --- survived measurement: volatility and structure, never direction ---
    VOLUME_ANOMALY = "volume_anomaly"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_COMPRESSION = "volatility_compression"
    REGIME_CHANGE = "regime_change"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDATION_SPIKE = "liquidation_spike"
    MAJOR_NEWS = "major_news"
    # --- about the system's own trustworthiness ---
    DATA_QUALITY = "data_quality"
    MODEL_DISAGREEMENT = "model_disagreement"
    PREDICTION_INVALIDATED = "prediction_invalidated"
    #: A summary of what the rate budget held back. Its own kind rather than borrowed
    #: from another: filing "I suppressed 40 things" under data quality would make a
    #: routine housekeeping notice indistinguishable from a broken feed in any count,
    #: chart or filter built on top.
    SUPPRESSION_DIGEST = "suppression_digest"
    # --- directional: cannot fire without confidence and invalidation ---
    STRONG_PREDICTION = "strong_prediction"
    SUPER_PREDICTION = "super_prediction"

    @property
    def is_directional(self) -> bool:
        """Whether this kind asserts which way the market will go.

        The distinction is load-bearing: directional kinds carry the same burden of
        proof as a published prediction, and nothing else does. A volume anomaly says
        "expect a bigger move than usual" and stays silent about the sign, which is
        exactly what the measurements support.
        """
        return self in {AlertKind.STRONG_PREDICTION, AlertKind.SUPER_PREDICTION}

    @property
    def default_severity(self) -> Severity:
        return {
            AlertKind.VOLUME_ANOMALY: Severity.NOTABLE,
            AlertKind.VOLATILITY_EXPANSION: Severity.NOTABLE,
            AlertKind.VOLATILITY_COMPRESSION: Severity.INFO,
            AlertKind.REGIME_CHANGE: Severity.IMPORTANT,
            AlertKind.CORRELATION_BREAKDOWN: Severity.IMPORTANT,
            AlertKind.LIQUIDATION_SPIKE: Severity.IMPORTANT,
            AlertKind.MAJOR_NEWS: Severity.NOTABLE,
            AlertKind.DATA_QUALITY: Severity.CRITICAL,
            AlertKind.MODEL_DISAGREEMENT: Severity.INFO,
            AlertKind.PREDICTION_INVALIDATED: Severity.CRITICAL,
            AlertKind.SUPPRESSION_DIGEST: Severity.INFO,
            AlertKind.STRONG_PREDICTION: Severity.IMPORTANT,
            AlertKind.SUPER_PREDICTION: Severity.CRITICAL,
        }[self]


@dataclass(slots=True)
class Alert:
    """One thing worth interrupting a human for."""

    kind: AlertKind
    asset: str
    title: str
    detail: str = ""
    severity: Severity | None = None
    at: datetime = field(default_factory=utcnow)
    timeframe: str = ""
    #: Structured context for a channel that can render it. Never required to
    #: understand the alert — a plain-text channel must remain fully informative.
    context: dict[str, object] = field(default_factory=dict)
    #: Required for directional kinds. See :meth:`__post_init__`.
    confidence: float | None = None
    invalidation: list[str] = field(default_factory=list)
    #: Set when this alert is a digest of suppressed ones rather than an event.
    is_digest: bool = False

    def __post_init__(self) -> None:
        if self.severity is None:
            self.severity = self.kind.default_severity
        self.asset = self.asset.upper()
        if not self.kind.is_directional:
            return
        if self.confidence is None or self.confidence <= 0:
            raise ValueError(
                f"{self.kind.value} is a directional alert and requires a confidence; "
                f"an interruption that asserts a direction without one is exactly what "
                f"the rest of this system is built to avoid"
            )
        if not [c for c in self.invalidation if c.strip()]:
            raise ValueError(
                f"{self.kind.value} requires at least one invalidation condition: a "
                f"directional claim a reader cannot check is not worth waking them for"
            )

    @property
    def level(self) -> Severity:
        """The severity, known to be set.

        ``severity`` is optional at construction so a caller can accept the kind's
        default, and ``__post_init__`` always fills it. This property is what callers
        should read: it carries that guarantee in the type rather than making every
        call site assert it.
        """
        assert self.severity is not None  # set unconditionally in __post_init__
        return self.severity

    @property
    def dedup_key(self) -> str:
        """Identity for suppression: same kind, same asset, same timeframe, same claim.

        The title is included but the timestamp is not. Two volume anomalies on BTC 1h
        twenty minutes apart are the same news; the same alert re-derived on the next
        tick should not read as a second event.
        """
        payload = f"{self.kind.value}|{self.asset}|{self.timeframe}|{self.title}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def scope(self) -> tuple[str, str]:
        """The cooldown scope: one kind, one asset."""
        return (self.kind.value, self.asset)

    def render(self) -> str:
        """Plain text carrying everything a reader needs.

        Deliberately the single rendering path. A channel that builds its own message
        can omit the confidence or the invalidation without anyone noticing, and the
        Phase 10 lesson was that a display rule survives only where it cannot be
        bypassed.
        """
        head = f"[{self.level.label.upper()}] {self.asset}"
        if self.timeframe:
            head += f" {self.timeframe}"
        lines = [f"{head}: {self.title}"]
        if self.detail:
            lines.append(self.detail)
        if self.kind.is_directional:
            lines.append(f"confidence {self.confidence:.0%} — this is a probabilistic")
            lines.append("scenario, not a guaranteed outcome")
            lines.append("invalidated if: " + "; ".join(self.invalidation))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.render()
