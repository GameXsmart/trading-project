"""Provider interface plus the shared HTTP machinery.

Every data source implements :class:`MarketDataProvider`. Nothing above this layer
knows that Binance paginates by `startTime` or that Kraken returns arrays — the
translation into :class:`~mie.core.types.Candle` stops here.

The three cross-cutting concerns every HTTP source needs — rate limiting, retry with
backoff, and a circuit breaker — live in :class:`HttpProvider` rather than being
reimplemented per exchange.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from mie.config.settings import ProviderConfig
from mie.core.errors import NotSupported, ProviderError, ProviderUnavailable, RateLimited
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import (
    Candle,
    FundingRate,
    GlobalMetricsPoint,
    MarketType,
    OpenInterestPoint,
    ProviderHealth,
)

log = get_logger(__name__)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "HttpProvider",
    "MarketDataProvider",
    "ProviderCapabilities",
    "TokenBucket",
]


# --------------------------------------------------------------------- throttling


class TokenBucket:
    """Async token bucket.

    Sustains ``rate`` requests/second while allowing a burst of ``burst``. Chosen over
    a fixed sleep because exchange limits are themselves burst-tolerant: a naive
    per-request delay wastes most of the available budget during backfill.
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = max(1, burst)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns the seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
                waited += sleep_for
                await asyncio.sleep(sleep_for)

    @property
    def available(self) -> float:
        now = time.monotonic()
        return min(self.capacity, self._tokens + (now - self._last) * self.rate)


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stops hammering a provider that is already failing.

    Without this, a dead endpoint costs one full timeout per request per asset per
    timeframe — which is how a single outage turns into an ingestion stall. After
    ``failure_threshold`` consecutive failures the circuit opens and calls are
    rejected immediately until the cooldown elapses; one probe then decides whether
    to close it again.
    """

    def __init__(self, failure_threshold: int = 4, cooldown_s: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self.last_error: str | None = None

    def allows(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.opened_at is not None and (time.monotonic() - self.opened_at) >= self.cooldown_s:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow the probe

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at = None
        self.last_error = None

    def record_failure(self, error: str) -> None:
        self.consecutive_failures += 1
        self.last_error = error
        # A failed half-open probe re-opens immediately; no need to burn the full
        # threshold again on a provider we just confirmed is still down.
        if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }


# -------------------------------------------------------------------- interface


@dataclass(slots=True)
class ProviderCapabilities:
    """What a provider can actually serve.

    The manager consults this before dispatching, so an unsupported request fails
    over instantly instead of costing a round trip and a parse error.
    """

    timeframes: frozenset[Timeframe]
    market_types: frozenset[MarketType] = frozenset({MarketType.SPOT})
    max_candles_per_request: int = 1000
    max_history_days: int | None = None
    supports_funding: bool = False
    supports_open_interest: bool = False
    supports_global_metrics: bool = False
    unsupported_assets: frozenset[str] = field(default_factory=frozenset)


class MarketDataProvider(ABC):
    """Contract every data source implements."""

    name: str = "abstract"
    kind: str = "exchange"

    def __init__(self, config: ProviderConfig, symbol_overrides: dict[str, str] | None = None) -> None:
        self.config = config
        # Per-asset overrides for venues whose ticker differs from the canonical
        # symbol (Kraken's XBT for BTC, CoinGecko's slugs).
        self.symbol_overrides = {k.upper(): v for k, v in (symbol_overrides or {}).items()}
        self.breaker = CircuitBreaker(
            config.circuit_breaker.failure_threshold, config.circuit_breaker.cooldown_s
        )

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        """Return candles ascending by ``open_time``, covering ``[start, end)``."""

    @abstractmethod
    async def health(self) -> ProviderHealth: ...

    async def fetch_funding(self, asset: str, limit: int = 100) -> list[FundingRate]:
        raise NotSupported(self.name, "funding rates not available")

    async def fetch_open_interest(self, asset: str, limit: int = 100) -> list[OpenInterestPoint]:
        raise NotSupported(self.name, "open interest not available")

    async def fetch_global_metrics(self) -> GlobalMetricsPoint:
        raise NotSupported(self.name, "global metrics not available")

    async def close(self) -> None:  # noqa: B027 - optional hook; not every provider holds resources
        """Release network resources. Safe to call more than once."""

    # ---------------------------------------------------------------- symbols

    def symbol_for(self, asset: str, quote: str = "USDT") -> str:
        """Translate a canonical asset into this venue's symbol."""
        asset = asset.upper()
        if asset in self.symbol_overrides:
            return self.symbol_overrides[asset]
        return self._default_symbol(asset, quote)

    def _default_symbol(self, asset: str, quote: str) -> str:
        return f"{asset}{quote}"

    def supports(
        self, asset: str, timeframe: Timeframe, market_type: MarketType = MarketType.SPOT
    ) -> bool:
        caps = self.capabilities
        return (
            timeframe in caps.timeframes
            and market_type in caps.market_types
            and asset.upper() not in caps.unsupported_assets
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name} priority={self.config.priority}>"


# ------------------------------------------------------------------ http shared


class HttpProvider(MarketDataProvider):
    """Base for REST providers: one shared client, throttle, retry, error mapping."""

    base_url: str = ""

    def __init__(self, config: ProviderConfig, symbol_overrides: dict[str, str] | None = None) -> None:
        super().__init__(config, symbol_overrides)
        self._client: httpx.AsyncClient | None = None
        self._bucket = TokenBucket(config.rate_limit.rate, config.rate_limit.burst)
        self._url = config.base_url or self.base_url

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._url,
                timeout=httpx.Timeout(self.config.timeout_s),
                headers={"User-Agent": "mie-market-intelligence/0.1 (analytics; read-only)"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Throttled GET with bounded retries and provider-specific error mapping.

        Retries use exponential backoff with jitter: synchronised retries across
        concurrently-backfilling assets would recreate the burst that triggered the
        rate limit in the first place.
        """
        attempts = max(1, self.config.max_retries)
        last_error: Exception | None = None

        for attempt in range(attempts):
            await self._bucket.acquire()
            try:
                response = await self.client.get(path, params=params)
            except httpx.TimeoutException:
                last_error = ProviderUnavailable(self.name, f"timeout on {path}")
                log.debug("provider_timeout", provider=self.name, path=path, attempt=attempt)
            except httpx.HTTPError as exc:
                last_error = ProviderUnavailable(self.name, f"transport error: {exc}")
                log.debug("provider_transport_error", provider=self.name, error=str(exc)[:200])
            else:
                if response.status_code == 429:
                    retry_after = _retry_after(response)
                    last_error = RateLimited(self.name, "rate limited", retry_after)
                    await asyncio.sleep(retry_after or self._backoff(attempt))
                    continue
                if response.status_code in (418, 403):
                    # Binance uses 418 for an IP ban after repeated limit breaches;
                    # 403 is typically geo-blocking. Neither is worth retrying.
                    raise ProviderUnavailable(
                        self.name, f"blocked with HTTP {response.status_code}"
                    )
                if 500 <= response.status_code < 600:
                    last_error = ProviderUnavailable(
                        self.name, f"server error {response.status_code}"
                    )
                elif response.status_code >= 400:
                    detail = response.text[:200]
                    raise ProviderError(self.name, f"HTTP {response.status_code}: {detail}")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderError(self.name, f"malformed JSON from {path}: {exc}") from exc

            if attempt < attempts - 1:
                await asyncio.sleep(self._backoff(attempt))

        raise last_error or ProviderUnavailable(self.name, f"request to {path} failed")

    def _backoff(self, attempt: int) -> float:
        base = self.config.retry_backoff_s * (2**attempt)
        return base * (0.5 + random.random())  # full jitter around the base delay

    async def health(self) -> ProviderHealth:
        """Default health probe: time the provider's cheapest endpoint."""
        started = time.monotonic()
        try:
            await self._health_probe()
        except Exception as exc:
            return ProviderHealth(
                provider=self.name,
                ok=False,
                error=str(exc)[:300],
                latency_ms=(time.monotonic() - started) * 1000,
                detail=self.breaker.snapshot(),
                checked_at=utcnow(),
            )
        return ProviderHealth(
            provider=self.name,
            ok=True,
            latency_ms=(time.monotonic() - started) * 1000,
            detail=self.breaker.snapshot(),
            checked_at=utcnow(),
        )

    async def _health_probe(self) -> None:
        raise NotImplementedError


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
