"""News engine.

Fetch → deduplicate → classify → score. Produces :class:`NewsEvent` objects from raw
feeds, with every field requirement §8 asks for.

The ordering is deliberate. Deduplication happens *before* classification, so each
story is classified once from its earliest article rather than repeatedly from every
outlet's rewrite — which would be wasteful, and worse, would let the loudest
republisher's phrasing determine the sentiment of a story it did not break.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from mie.core.logging import get_logger
from mie.core.timeframes import utcnow
from mie.news.classify import NewsClassifier
from mie.news.dedup import Cluster, Deduplicator
from mie.news.sources import DEFAULT_FEEDS, NewsFetcher, RSSFeed
from mie.news.types import NewsEvent, NewsItem

log = get_logger(__name__)

__all__ = ["NewsEngine"]


class NewsEngine:
    """Turns RSS feeds into classified, deduplicated news events."""

    def __init__(
        self,
        feeds: Sequence[RSSFeed] = DEFAULT_FEEDS,
        fetcher: NewsFetcher | None = None,
        deduplicator: Deduplicator | None = None,
        classifier: NewsClassifier | None = None,
        max_age: timedelta = timedelta(days=7),
    ) -> None:
        self.fetcher = fetcher or NewsFetcher(feeds)
        self.deduplicator = deduplicator or Deduplicator()
        self.classifier = classifier or NewsClassifier()
        #: How far back a story may have broken and still be reported. Wider than
        #: the feeds' own window so a story is never truncated mid-coverage.
        self.max_age = max_age
        #: Clusters seen previously, so a story re-run days later is recognised
        #: rather than treated as new because this fetch has no memory.
        self._known: list[Cluster] = []

    async def fetch_events(self) -> list[NewsEvent]:
        """Fetch every feed and return classified events, newest first."""
        items = await self.fetcher.fetch_all()
        return self.process(items)

    def process(self, items: Sequence[NewsItem]) -> list[NewsEvent]:
        """Deduplicate and classify a batch of articles.

        **Cluster first, filter second.** Filtering by age before clustering splits a
        story whose coverage straddles the cutoff, which destroys the very coverage
        count deduplication exists to produce. Measured on live feeds: with the filter
        applied first, three genuinely-merged stories were broken apart because one
        outlet published 84 hours ago and another 71, and the batch reported them as
        six separate single-outlet stories.

        So every fetched article is clustered, and the age test is then applied to the
        *story*, using its most recent coverage.
        """
        # A mis-dated future item would otherwise sort to the top and dominate every
        # "latest news" view.
        horizon = utcnow() + timedelta(hours=6)
        usable = [i for i in items if i.published_at <= horizon]
        if not usable:
            return []

        all_clusters = self.deduplicator.cluster(usable)
        cutoff = utcnow() - self.max_age
        # Judged on the *most recent* coverage, not on when the story broke: a story
        # that first ran five days ago and is still being written about today is live
        # news, and dropping it for the age of its first article would discard the
        # coverage that makes it significant. The event still reports `published_at`
        # as the break time, which is when the information actually reached the market.
        clusters = [c for c in all_clusters if c.latest.published_at >= cutoff]
        if not clusters:
            return []
        recycled = self.deduplicator.find_recycled(clusters, self._known)

        events = [
            self._to_event(cluster, recycled.get(cluster.cluster_id))
            for cluster in clusters
        ]
        self._remember(clusters)

        log.info(
            "news_processed",
            articles=len(usable),
            events=len(events),
            recycled=sum(1 for e in events if e.is_recycled),
        )
        return sorted(events, key=lambda e: e.published_at, reverse=True)

    def _to_event(self, cluster: Cluster, recycled_from: str | None) -> NewsEvent:
        lead = cluster.representative
        # Classify from the whole cluster's titles so a rewording that adds the asset
        # name is not lost, but keep the lead article's own title as the label.
        combined_titles = " ".join(item.title for item in cluster.items)
        body = " ".join(item.summary for item in cluster.items)[:4000]

        relevance = self.classifier.relevance(combined_titles, body)
        category = self.classifier.category(f"{combined_titles} {body}")
        sentiment, score, hits = self.classifier.sentiment(combined_titles, body)
        source_weight = max(
            (self.fetcher.source_weight(name) for name in cluster.outlets), default=0.5
        )
        importance = self.classifier.importance(
            category=category,
            coverage=len(cluster.outlets),
            source_weight=source_weight,
            relevance=relevance,
            sentiment_magnitude=abs(score),
        )
        confidence = self.classifier.confidence(hits, relevance, category)

        return NewsEvent(
            cluster_id=cluster.cluster_id,
            title=lead.title,
            url=lead.url,
            published_at=lead.published_at,
            sources=sorted(cluster.outlets),
            category=category,
            sentiment=sentiment,
            sentiment_score=score,
            relevance=relevance,
            importance=importance,
            confidence=confidence,
            is_recycled=recycled_from is not None,
            recycled_from=recycled_from,
            article_count=len(cluster.items),
        )

    def _remember(self, clusters: Sequence[Cluster], keep: int = 500) -> None:
        """Retain recent clusters so recycled stories can be spotted next fetch."""
        by_id = {c.cluster_id: c for c in self._known}
        for cluster in clusters:
            existing = by_id.get(cluster.cluster_id)
            if existing is None:
                by_id[cluster.cluster_id] = cluster
            else:
                existing.items.extend(cluster.items)
        self._known = sorted(
            by_id.values(), key=lambda c: c.earliest.published_at, reverse=True
        )[:keep]

    # ------------------------------------------------------------------ queries

    @staticmethod
    def for_asset(
        events: Sequence[NewsEvent], asset: str, min_relevance: float = 0.5
    ) -> list[NewsEvent]:
        """Events genuinely about an asset, most relevant first."""
        return sorted(
            (e for e in events if e.relevance_for(asset) >= min_relevance),
            key=lambda e: (-e.importance, -e.published_at.timestamp()),
        )

    @staticmethod
    def market_sentiment(
        events: Sequence[NewsEvent], asset: str | None = None
    ) -> tuple[float, int]:
        """Importance-weighted mean sentiment, and how many events informed it.

        Recycled stories are excluded: a re-run is not new information, and counting
        it again would let one old development sway the reading repeatedly.
        """
        relevant = [
            e
            for e in events
            if not e.is_recycled
            and (asset is None or e.relevance_for(asset) >= 0.5)
        ]
        if not relevant:
            return 0.0, 0
        weights = [e.importance * e.confidence for e in relevant]
        total = sum(weights)
        if total <= 0:
            return 0.0, len(relevant)
        weighted = sum(e.sentiment_score * w for e, w in zip(relevant, weights, strict=True))
        return round(weighted / total, 4), len(relevant)

    async def close(self) -> None:
        await self.fetcher.close()

    async def __aenter__(self) -> NewsEngine:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
