"""Delivery. Where alerts leave the machine, and what that obliges.

Three rules govern everything here.

**No secrets in code or config files.** §23. Every destination — a Discord webhook, a
Telegram bot token, an SMTP password — is read from the environment and nowhere else. A
channel whose target is not configured is *disabled*, not silently broken: it reports
that it is off, so "no alerts arrived" can be distinguished from "no alerts were sent".

**One rendering path.** Channels format the envelope, never the content. Every one of
them sends :meth:`Alert.render`, which is the single place that guarantees a directional
alert carries its confidence and its invalidation conditions. A channel that built its
own message could drop them, and nobody would notice until it mattered.

**A failing channel never blocks the others.** Delivery is best-effort per channel and
failures are recorded rather than raised. An alert that reached three of four
destinations is a better outcome than an exception that reached none.

Nothing here is enabled by default except the console and a local file. Sending on
someone's behalf — to a chat server, to an inbox — is an outward-facing action, and the
system requires it to be switched on deliberately rather than inheriting it from a
default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from mie.alerts.types import Alert, Severity
from mie.core.logging import get_logger
from mie.core.timeframes import utcnow

log = get_logger(__name__)

__all__ = [
    "Channel",
    "ConsoleChannel",
    "DeliveryResult",
    "DiscordChannel",
    "FileChannel",
    "TelegramChannel",
    "WebhookChannel",
    "channels_from_env",
]

#: Seconds before a delivery attempt is abandoned. Short: an alert that arrives a
#: minute late has already lost most of its value, and a hung request would delay
#: every other channel behind it.
_TIMEOUT = 8.0


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """What happened when one channel tried to send one alert."""

    channel: str
    delivered: bool
    error: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.channel}: {'ok' if self.delivered else self.error or 'failed'}"


class Channel(Protocol):
    """Somewhere an alert can go."""

    name: str

    @property
    def enabled(self) -> bool:
        """Whether this channel has everything it needs to deliver."""
        ...

    async def send(self, alert: Alert) -> DeliveryResult: ...


@dataclass(slots=True)
class ConsoleChannel:
    """Prints to the log. Always available, and the default.

    Deliberately the fallback: a system whose only delivery path is an external service
    is a system that goes silent when that service does, at precisely the moment it is
    most likely to have something to say.
    """

    name: str = "console"
    minimum: Severity = Severity.INFO

    @property
    def enabled(self) -> bool:
        return True

    async def send(self, alert: Alert) -> DeliveryResult:
        if alert.level < self.minimum:
            return DeliveryResult(self.name, False, "below channel minimum")
        log.info(
            "alert",
            kind=alert.kind.value,
            asset=alert.asset,
            severity=alert.level.label,
            title=alert.title,
        )
        return DeliveryResult(self.name, True)


@dataclass(slots=True)
class FileChannel:
    """Appends JSON lines to a file.

    The transport behind the browser and desktop surfaces: a local feed anything can
    tail without needing a network round trip or a third-party account. Append-only, so
    the alert history is a record rather than a view.
    """

    path: Path
    name: str = "file"
    minimum: Severity = Severity.INFO

    @property
    def enabled(self) -> bool:
        return True

    async def send(self, alert: Alert) -> DeliveryResult:
        if alert.level < self.minimum:
            return DeliveryResult(self.name, False, "below channel minimum")
        payload = {
            "at": alert.at.isoformat(),
            "kind": alert.kind.value,
            "asset": alert.asset,
            "timeframe": alert.timeframe,
            "severity": alert.level.label,
            "title": alert.title,
            "detail": alert.detail,
            "text": alert.render(),
            "confidence": alert.confidence,
            "invalidation": alert.invalidation,
            "is_digest": alert.is_digest,
            "context": alert.context,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
            return DeliveryResult(self.name, True)
        except OSError as exc:
            return DeliveryResult(self.name, False, str(exc)[:200])


@dataclass(slots=True)
class WebhookChannel:
    """Posts JSON to a configured URL.

    The base for Discord and Telegram. The URL comes from the environment; with no URL
    the channel is disabled rather than pretending to work.
    """

    url: str = ""
    name: str = "webhook"
    minimum: Severity = Severity.NOTABLE
    timeout: float = _TIMEOUT

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def payload(self, alert: Alert) -> dict[str, Any]:
        return {"text": alert.render(), "kind": alert.kind.value, "asset": alert.asset}

    async def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(self.name, False, "not configured")
        if alert.level < self.minimum:
            return DeliveryResult(self.name, False, "below channel minimum")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json=self.payload(alert))
            if response.status_code >= 400:
                return DeliveryResult(self.name, False, f"HTTP {response.status_code}")
            return DeliveryResult(self.name, True)
        except (httpx.HTTPError, OSError) as exc:
            # Recorded, never raised: a failing channel must not stop the others.
            return DeliveryResult(self.name, False, _describe(exc))


@dataclass(slots=True)
class DiscordChannel(WebhookChannel):
    """A webhook shaped the way Discord expects."""

    name: str = "discord"

    def payload(self, alert: Alert) -> dict[str, Any]:
        colour = {
            Severity.INFO: 0x5B6674,
            Severity.NOTABLE: 0x5B9DD9,
            Severity.IMPORTANT: 0xD99A2B,
            Severity.CRITICAL: 0xE5534B,
        }[alert.level]
        fields = []
        if alert.kind.is_directional:
            fields.append(
                {"name": "confidence", "value": f"{alert.confidence:.0%}", "inline": True}
            )
            fields.append(
                {"name": "invalidated if", "value": "; ".join(alert.invalidation)[:1000]}
            )
        return {
            "embeds": [
                {
                    "title": f"{alert.asset} — {alert.title}"[:250],
                    "description": (alert.detail or "")[:2000],
                    "color": colour,
                    "fields": fields,
                    "footer": {
                        "text": "analysis only · not a guaranteed outcome · not advice"
                    },
                    "timestamp": alert.at.isoformat(),
                }
            ]
        }


@dataclass(slots=True)
class TelegramChannel:
    """Sends via the Telegram bot API.

    Token and chat id come from the environment. Both are required; with either missing
    the channel is disabled.
    """

    token: str = ""
    chat_id: str = ""
    name: str = "telegram"
    minimum: Severity = Severity.NOTABLE
    timeout: float = _TIMEOUT
    #: Overridable so tests can point at a local sink instead of Telegram.
    base_url: str = "https://api.telegram.org"

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, alert: Alert) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult(self.name, False, "not configured")
        if alert.level < self.minimum:
            return DeliveryResult(self.name, False, "below channel minimum")
        text = alert.render() + "\n\nanalysis only — not a guaranteed outcome, not advice"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text},
                )
            if response.status_code >= 400:
                return DeliveryResult(self.name, False, f"HTTP {response.status_code}")
            return DeliveryResult(self.name, True)
        except (httpx.HTTPError, OSError) as exc:
            return DeliveryResult(self.name, False, _describe(exc))


def _describe(exc: BaseException) -> str:
    """A non-empty description of a failure.

    ``str(exc)`` is empty for several httpx connection errors, which would produce a
    delivery result that records a failure without saying anything about it — the
    least useful possible outcome for something a reader only sees when it breaks.
    """
    text = str(exc).strip()
    return (text or type(exc).__name__)[:200]


def channels_from_env(
    feed_path: Path | None = None, env: dict[str, str] | None = None
) -> list[Channel]:
    """Assemble the configured channels from the environment.

    Every destination is read from an environment variable and never from a config
    file, so a checked-in repository cannot carry a webhook someone forgot about. The
    console and the local feed are always present; everything else appears only when it
    has been switched on deliberately.
    """
    source = env if env is not None else dict(os.environ)
    built: list[Channel] = [ConsoleChannel()]
    if feed_path is not None:
        built.append(FileChannel(path=feed_path))

    if url := source.get("MIE_ALERTS__DISCORD_WEBHOOK", "").strip():
        built.append(DiscordChannel(url=url))
    if url := source.get("MIE_ALERTS__WEBHOOK_URL", "").strip():
        built.append(WebhookChannel(url=url))
    token = source.get("MIE_ALERTS__TELEGRAM_TOKEN", "").strip()
    chat = source.get("MIE_ALERTS__TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        built.append(TelegramChannel(token=token, chat_id=chat))

    log.info(
        "alert_channels",
        enabled=[c.name for c in built if c.enabled],
        configured_at=utcnow().isoformat(),
    )
    return built


@dataclass(slots=True)
class _Recorder:
    """A channel that records instead of sending. For tests and dry runs."""

    name: str = "recorder"
    sent: list[Alert] = field(default_factory=list)
    minimum: Severity = Severity.INFO

    @property
    def enabled(self) -> bool:
        return True

    async def send(self, alert: Alert) -> DeliveryResult:
        self.sent.append(alert)
        return DeliveryResult(self.name, True)
