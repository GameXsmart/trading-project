"""News sources.

RSS only, and deliberately so. Every feed here is public, keyless, and published by
the outlet for exactly this purpose — no scraping, no terms to violate, and nothing
that stops working when a free tier changes.

Parsing uses the standard library rather than `feedparser`, because RSS and Atom
between them are about forty lines of XML handling and the dependency would buy
little. What it *would* buy — tolerance of malformed feeds — is handled explicitly:
a feed that fails to parse is skipped with a warning rather than taking down the
fetch, since one outlet publishing broken XML must not cost the other six.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from mie.core.logging import get_logger
from mie.core.timeframes import UTC, utcnow
from mie.news.types import NewsItem

log = get_logger(__name__)

__all__ = ["DEFAULT_FEEDS", "NewsFetcher", "RSSFeed"]

#: Namespaces that appear in the wild across these feeds.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


class RSSFeed:
    """One outlet's feed, with the weight its reporting carries."""

    def __init__(self, name: str, url: str, weight: float = 1.0) -> None:
        self.name = name
        self.url = url
        #: Source credibility, used when scoring importance. A story carried only by
        #: an aggregator is weaker evidence than one carried by a primary outlet.
        self.weight = weight

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RSSFeed {self.name}>"


#: Checked-in defaults. All verified reachable without credentials.
DEFAULT_FEEDS: tuple[RSSFeed, ...] = (
    RSSFeed("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", weight=1.0),
    RSSFeed("cointelegraph", "https://cointelegraph.com/rss", weight=0.85),
    RSSFeed("decrypt", "https://decrypt.co/feed", weight=0.85),
    RSSFeed("bitcoinmagazine", "https://bitcoinmagazine.com/feed", weight=0.75),
    RSSFeed("cryptoslate", "https://cryptoslate.com/feed/", weight=0.7),
    RSSFeed("newsbtc", "https://www.newsbtc.com/feed/", weight=0.6),
    RSSFeed("bitcoinist", "https://bitcoinist.com/feed/", weight=0.6),
)


class NewsFetcher:
    """Fetches and parses RSS feeds concurrently."""

    def __init__(
        self,
        feeds: Sequence[RSSFeed] = DEFAULT_FEEDS,
        timeout_s: float = 15.0,
        max_items_per_feed: int = 60,
    ) -> None:
        self.feeds = tuple(feeds)
        self.timeout_s = timeout_s
        self.max_items_per_feed = max_items_per_feed
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                follow_redirects=True,
                headers={"User-Agent": "mie-market-intelligence/0.1 (analytics; read-only)"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def fetch_all(self) -> list[NewsItem]:
        """Fetch every configured feed. One broken feed must not lose the rest."""
        results = await asyncio.gather(
            *(self.fetch(feed) for feed in self.feeds), return_exceptions=True
        )
        items: list[NewsItem] = []
        for feed, result in zip(self.feeds, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("news_feed_failed", feed=feed.name, error=str(result)[:200])
                continue
            items.extend(result)
        log.info("news_fetched", feeds=len(self.feeds), items=len(items))
        return items

    async def fetch(self, feed: RSSFeed) -> list[NewsItem]:
        """Fetch and parse one feed."""
        response = await self.client.get(feed.url)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        return parse_feed(response.text, feed.name)[: self.max_items_per_feed]

    def source_weight(self, name: str) -> float:
        feed = next((f for f in self.feeds if f.name == name), None)
        return feed.weight if feed else 0.5

    async def __aenter__(self) -> NewsFetcher:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def parse_feed(body: str, source: str) -> list[NewsItem]:
    """Parse RSS 2.0 or Atom into news items.

    Malformed XML yields an empty list rather than raising: feeds break, and losing
    one outlet is much better than losing the fetch.
    """
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        log.warning("news_feed_unparseable", source=source, error=str(exc)[:160])
        return []

    items: list[NewsItem] = []
    # RSS 2.0
    for element in root.iter("item"):
        item = _rss_item(element, source)
        if item is not None:
            items.append(item)
    # Atom
    for element in root.iter(f"{{{_NS['atom']}}}entry"):
        item = _atom_entry(element, source)
        if item is not None:
            items.append(item)
    return items


def _rss_item(element: ElementTree.Element, source: str) -> NewsItem | None:
    title = _text(element, "title")
    link = _text(element, "link")
    if not title or not link:
        return None
    published = (
        _parse_date(_text(element, "pubDate"))
        or _parse_date(_text(element, f"{{{_NS['dc']}}}date"))
        or utcnow()
    )
    summary = _text(element, "description") or _text(element, f"{{{_NS['content']}}}encoded")
    return NewsItem(
        source=source,
        title=_clean(title),
        url=link.strip(),
        published_at=published,
        summary=_clean(summary)[:1200],
    )


def _atom_entry(element: ElementTree.Element, source: str) -> NewsItem | None:
    title = _text(element, f"{{{_NS['atom']}}}title")
    link_element = element.find(f"{{{_NS['atom']}}}link")
    link = link_element.get("href") if link_element is not None else None
    if not title or not link:
        return None
    published = (
        _parse_date(_text(element, f"{{{_NS['atom']}}}published"))
        or _parse_date(_text(element, f"{{{_NS['atom']}}}updated"))
        or utcnow()
    )
    summary = _text(element, f"{{{_NS['atom']}}}summary") or _text(
        element, f"{{{_NS['atom']}}}content"
    )
    return NewsItem(
        source=source,
        title=_clean(title),
        url=link.strip(),
        published_at=published,
        summary=_clean(summary)[:1200],
    )


def _text(element: ElementTree.Element, tag: str) -> str:
    found = element.find(tag)
    return (found.text or "").strip() if found is not None else ""


def _clean(raw: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Feed summaries are frequently HTML fragments. A regex is the wrong tool for
    general HTML, but for stripping tags out of a short summary before tokenising it
    is adequate and avoids a parser dependency for cosmetic work.
    """
    import re

    without_tags = re.sub(r"<[^>]+>", " ", raw)
    unescaped = (
        without_tags.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#039;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", unescaped).strip()


def _parse_date(raw: str) -> datetime | None:
    """Parse RFC 822 or ISO 8601, returning tz-aware UTC.

    A feed that omits a timezone is assumed UTC — the only workable assumption, and
    flagged here so the choice is visible rather than silent.
    """
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
