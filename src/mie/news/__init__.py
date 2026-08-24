"""Phase 5: news ingestion, deduplication and classification."""

from mie.news.classify import ASSET_ALIASES, CATEGORY_RULES, NewsClassifier
from mie.news.dedup import Cluster, Deduplicator, jaccard, normalise_title, shingles
from mie.news.engine import NewsEngine
from mie.news.impact import (
    CATEGORY_IMPACT_PRIORS,
    EventImpactModel,
    ImpactEstimate,
    ImpactMeasurement,
    ImpactValidator,
)
from mie.news.sources import DEFAULT_FEEDS, NewsFetcher, RSSFeed, parse_feed
from mie.news.types import (
    AssetRelevance,
    EventCategory,
    NewsEvent,
    NewsItem,
    Sentiment,
)

__all__ = [
    "ASSET_ALIASES",
    "CATEGORY_IMPACT_PRIORS",
    "CATEGORY_RULES",
    "DEFAULT_FEEDS",
    "AssetRelevance",
    "Cluster",
    "Deduplicator",
    "EventCategory",
    "EventImpactModel",
    "ImpactEstimate",
    "ImpactMeasurement",
    "ImpactValidator",
    "NewsClassifier",
    "NewsEngine",
    "NewsEvent",
    "NewsFetcher",
    "NewsItem",
    "RSSFeed",
    "Sentiment",
    "jaccard",
    "normalise_title",
    "parse_feed",
    "shingles",
]
