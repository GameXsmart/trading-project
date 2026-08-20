"""Kraken provider — public OHLC endpoint only.

Third failover tier and a third independent price opinion. Kraken's API has one
structural limitation worth being explicit about: it returns roughly the most recent
720 candles from a `since` cursor and offers no end bound, so it is a good
corroborator and a poor deep-backfill source. `max_history_days` encodes that so the
backfill planner does not choose it for long ranges.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mie.core.errors import NotSupported, ProviderError
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, utcnow
from mie.core.types import Candle, MarketType
from mie.providers.base import HttpProvider, ProviderCapabilities

log = get_logger(__name__)

__all__ = ["KrakenProvider"]

# Kraken expresses intervals in minutes. 12h has no native equivalent.
_INTERVAL_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
}

_MAX_CANDLES = 720


class KrakenProvider(HttpProvider):
    name = "kraken"
    kind = "exchange"
    base_url = "https://api.kraken.com"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            timeframes=frozenset(_INTERVAL_MINUTES),
            market_types=frozenset({MarketType.SPOT}),
            max_candles_per_request=_MAX_CANDLES,
            # ~720 bars from the cursor; on 1m that is half a day.
            max_history_days=None,
        )

    def _default_symbol(self, asset: str, quote: str) -> str:
        normalised = "USD" if quote.upper() in ("USD", "USDT", "USDC") else quote.upper()
        return f"{asset.upper()}{normalised}"

    async def _health_probe(self) -> None:
        await self._get("/0/public/Time")

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        minutes = _INTERVAL_MINUTES.get(timeframe)
        if minutes is None:
            raise NotSupported(self.name, f"timeframe {timeframe}")

        pair = self.symbol_for(asset, quote)
        params: dict[str, Any] = {"pair": pair, "interval": minutes}
        if start is not None:
            # `since` is exclusive, so step back one bar to include the start bar.
            params["since"] = int(ensure_utc(start).timestamp()) - timeframe.seconds

        payload = await self._get("/0/public/OHLC", params)
        result = _unwrap(self.name, payload)

        # The result key is Kraken's own normalised pair name (XXBTZUSD for XBTUSD),
        # which is not derivable from the request, so take the first non-"last" key.
        series_key = next((k for k in result if k != "last"), None)
        if series_key is None:
            raise ProviderError(self.name, f"no OHLC series for {pair}")
        rows = result[series_key]
        if not isinstance(rows, list):
            raise ProviderError(self.name, f"unexpected OHLC series for {pair}")

        now = utcnow()
        # `since` is stepped back a bar to include the start bar, and Kraken has no
        # end bound at all, so both edges are enforced here to keep the half-open
        # [start, end) contract every caller relies on.
        lower = ensure_utc(start) if start is not None else None
        upper = ensure_utc(end) if end is not None else None
        candles: list[Candle] = []
        for row in rows:
            # [time, open, high, low, close, vwap, volume, count]
            try:
                open_time = datetime.fromtimestamp(int(row[0]), tz=now.tzinfo)
                if (upper is not None and open_time >= upper) or (
                    lower is not None and open_time < lower
                ):
                    continue
                volume = float(row[6])
                vwap = float(row[5])
                candles.append(
                    Candle(
                        asset=asset.upper(),
                        quote="USD",
                        source=self.name,
                        market_type=MarketType.SPOT,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=volume,
                        # Kraken gives VWAP rather than quote volume; their product is
                        # the quote-denominated turnover, which is what we store.
                        quote_volume=volume * vwap if vwap else None,
                        trades=int(row[7]) if len(row) > 7 else None,
                        is_final=timeframe.close_time(open_time) <= now,
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise ProviderError(self.name, f"malformed OHLC row {row!r}: {exc}") from exc

        candles.sort(key=lambda c: c.open_time)
        if limit:
            candles = candles[-limit:]
        return candles


def _unwrap(provider: str, payload: Any) -> dict[str, Any]:
    """Kraken reports errors in the body with HTTP 200, so they must be unwrapped."""
    if not isinstance(payload, dict):
        raise ProviderError(provider, f"unexpected payload type {type(payload).__name__}")
    errors = payload.get("error") or []
    if errors:
        raise ProviderError(provider, "; ".join(str(e) for e in errors))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ProviderError(provider, "missing result object")
    return result
