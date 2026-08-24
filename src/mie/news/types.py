"""News vocabulary.

Requirement §8 asks that every news event carry source, timestamp, asset relevance,
category, sentiment, estimated importance, confidence, and potential market impact.
These types make each of those an explicit field rather than something inferred later
from prose.

Two distinctions the design insists on:

**A story is not an article.** Twelve outlets republishing one Reuters wire is one
event with wide coverage, not twelve events. Conflating them would make "how many
outlets are reporting this" — a genuine importance signal — indistinguishable from
"how aggressively does this outlet republish".

**Sentiment is not impact.** An article can be written in glowing terms about
something the market has already priced, or in neutral terms about a catastrophe.
Sentiment describes the text; impact is a claim about prices, and it has to be
measured against them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mie.core.timeframes import utcnow

__all__ = [
    "AssetRelevance",
    "EventCategory",
    "NewsEvent",
    "NewsItem",
    "Sentiment",
]


class EventCategory(StrEnum):
    """What kind of thing happened.

    Category drives the impact model: a hack and a partnership announcement are not
    the same sort of event even when both are 'negative' or 'positive'.
    """

    REGULATION = "regulation"
    ENFORCEMENT = "enforcement"
    ETF = "etf"
    SECURITY_INCIDENT = "security_incident"
    EXCHANGE = "exchange"
    PROTOCOL = "protocol"
    PARTNERSHIP = "partnership"
    ADOPTION = "adoption"
    MACRO = "macro"
    MARKET_MOVE = "market_move"
    LISTING = "listing"
    FUNDING = "funding"
    OTHER = "other"

    @property
    def base_importance(self) -> float:
        """Prior importance in [0, 1], before any evidence about this specific story.

        Regulatory action and security incidents move markets; a funding round for a
        small project usually does not. These are priors, not measurements, and the
        impact model is what tests whether they survive contact with price data.
        """
        return {
            "regulation": 0.75,
            "enforcement": 0.75,
            "etf": 0.80,
            "security_incident": 0.85,
            "exchange": 0.65,
            "protocol": 0.55,
            "partnership": 0.35,
            "adoption": 0.45,
            "macro": 0.70,
            "market_move": 0.30,
            "listing": 0.50,
            "funding": 0.25,
            "other": 0.20,
        }[self.value]


class Sentiment(StrEnum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

    @property
    def score(self) -> float:
        return {
            "very_negative": -1.0,
            "negative": -0.5,
            "neutral": 0.0,
            "positive": 0.5,
            "very_positive": 1.0,
        }[self.value]

    @classmethod
    def from_score(cls, score: float) -> Sentiment:
        if score >= 0.55:
            return cls.VERY_POSITIVE
        if score >= 0.15:
            return cls.POSITIVE
        if score <= -0.55:
            return cls.VERY_NEGATIVE
        if score <= -0.15:
            return cls.NEGATIVE
        return cls.NEUTRAL


class NewsItem(BaseModel):
    """One article as published by one outlet, before any interpretation."""

    model_config = ConfigDict(frozen=True)

    source: str
    title: str
    url: str
    published_at: datetime
    summary: str = ""
    fetched_at: datetime = Field(default_factory=utcnow)

    @property
    def text(self) -> str:
        """Title and summary together, for classification."""
        return f"{self.title}. {self.summary}".strip()


class AssetRelevance(BaseModel):
    """How strongly one article concerns one asset."""

    model_config = ConfigDict(frozen=True)

    asset: str
    #: In [0, 1]. Title mentions weigh far more than a passing body reference.
    score: float
    mentions: int = 0
    in_title: bool = False


class NewsEvent(BaseModel):
    """A deduplicated story, classified and scored.

    One event may be backed by many articles from many outlets; ``coverage`` is how
    many distinct outlets carried it, which is the most honest importance signal
    available without waiting to see what prices did.
    """

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    title: str
    url: str
    #: Publication time of the *earliest* article in the cluster — when the story
    #: broke, not when the last outlet got round to it.
    published_at: datetime
    sources: list[str] = Field(default_factory=list)
    category: EventCategory = EventCategory.OTHER
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0
    relevance: list[AssetRelevance] = Field(default_factory=list)
    importance: float = 0.0
    #: How much to trust this classification, distinct from how important it is.
    confidence: float = 0.0
    #: True when the story appears to be a re-run of something already reported.
    is_recycled: bool = False
    recycled_from: str | None = None
    article_count: int = 1
    detected_at: datetime = Field(default_factory=utcnow)

    @property
    def coverage(self) -> int:
        """Distinct outlets carrying the story."""
        return len(set(self.sources))

    @property
    def assets(self) -> list[str]:
        """Assets this story is genuinely about, most relevant first."""
        return [r.asset for r in sorted(self.relevance, key=lambda r: -r.score)]

    def relevance_for(self, asset: str) -> float:
        match = next((r for r in self.relevance if r.asset == asset.upper()), None)
        return match.score if match else 0.0

    def summary(self) -> str:
        assets = ", ".join(self.assets[:3]) or "market-wide"
        flags = " [recycled]" if self.is_recycled else ""
        return (
            f"[{self.category}] {self.title[:70]} — {assets} | "
            f"{self.sentiment} | importance {self.importance:.2f} | "
            f"{self.coverage} outlet(s){flags}"
        )
