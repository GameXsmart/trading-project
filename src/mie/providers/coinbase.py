"""Coinbase Exchange provider — public market-data endpoints only.

Secondary source. Its value is not redundancy for its own sake: Coinbase is a
USD-denominated venue with a different participant base, so agreement between it and
Binance is real corroboration, and disagreement is a signal worth recording as a
`SOURCE_DISCREPANCY` quality event.

Two constraints shape the implementation: only six granularities are offered, and a
single request returns at most 300 candles.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from mie.core.errors import NotSupported, ProviderError
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, utcnow
from mie.core.types import Candle, MarketType
from mie.providers.base import HttpProvider, ProviderCapabilities

log = get_logger(__name__)

__all__ = ["CoinbaseProvider"]

# Coinbase granularities are fixed seconds. 30m, 4h, 12h and 1w have no native
# equivalent; the manager fails over rather than silently resampling, because a
# resampled bar from a different venue is not the same object as a native one.
_GRANULARITY: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.H1: 3600,
    Timeframe.D1: 86400,
}

_MAX_CANDLES = 300

# Assets with no Coinbase USD spot market at time of writing. Declaring them means
# the manager skips this provider instantly instead of spending a 404 to find out.
_UNSUPPORTED = frozenset({"BNB"})


class CoinbaseProvider(HttpProvider):
    name = "coinbase"
    kind = "exchange"
    base_url = "https://api.exchange.coinbase.com"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            timeframes=frozenset(_GRANULARITY),
            market_types=frozenset({MarketType.SPOT}),
            max_candles_per_request=_MAX_CANDLES,
            unsupported_assets=_UNSUPPORTED,
        )

    def _default_symbol(self, asset: str, quote: str) -> str:
        # Coinbase products are dash-separated and USD-quoted; USDT maps to USD.
        normalised = "USD" if quote.upper() in ("USD", "USDT", "USDC") else quote.upper()
        return f"{asset.upper()}-{normalised}"

    async def _health_probe(self) -> None:
        await self._get("/time")

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        granularity = _GRANULARITY.get(timeframe)
        if granularity is None:
            raise NotSupported(self.name, f"timeframe {timeframe}")
        if asset.upper() in _UNSUPPORTED:
            raise NotSupported(self.name, f"no {asset.upper()} market")

        product = self.symbol_for(asset, quote)
        requested = min(limit or _MAX_CANDLES, _MAX_CANDLES)

        # The API takes an explicit window, not a count, so derive one from whichever
        # bounds the caller supplied.
        if end is None:
            end = utcnow()
        if start is None:
            start = end - timedelta(seconds=granularity * requested)
        start, end = ensure_utc(start), ensure_utc(end)

        span = int((end - start).total_seconds() // granularity)
        if span > _MAX_CANDLES:
            start = end - timedelta(seconds=granularity * _MAX_CANDLES)

        params: dict[str, Any] = {
            "granularity": granularity,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        }
        payload = await self._get(f"/products/{product}/candles", params)
        if not isinstance(payload, list):
            raise ProviderError(self.name, f"unexpected candles payload for {product}")

        now = utcnow()
        candles: list[Candle] = []
        for row in payload:
            # Coinbase row order is [time, low, high, open, close, volume] — note that
            # it is *not* the conventional OHLC ordering.
            try:
                open_time = datetime.fromtimestamp(int(row[0]), tz=now.tzinfo)
                candles.append(
                    Candle(
                        asset=asset.upper(),
                        quote="USD",
                        source=self.name,
                        market_type=MarketType.SPOT,
                        timeframe=timeframe,
                        open_time=open_time,
                        low=float(row[1]),
                        high=float(row[2]),
                        open=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        is_final=timeframe.close_time(open_time) <= now,
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise ProviderError(self.name, f"malformed candle row {row!r}: {exc}") from exc

        # Coinbase treats `end` as inclusive, so the boundary bar comes back too.
        # Every window in this system is half-open [start, end); letting the extra bar
        # through would double-count it at page boundaries during backfill and would
        # hand the caller the forming bar in live use.
        candles = [c for c in candles if start <= c.open_time < end]

        # Coinbase returns newest-first; every consumer expects ascending time.
        candles.sort(key=lambda c: c.open_time)
        if limit:
            candles = candles[-limit:]
        return candles
