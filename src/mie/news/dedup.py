"""Deduplication and recycled-story detection.

Requirement §8: detect duplicate and recycled news, and distinguish a genuinely new
story from one being reposted.

This matters more than it first appears. Without it:

* **Importance becomes uncountable.** "How many outlets are covering this" is the best
  pre-price signal of a story's significance — but only if one story republished by
  twelve outlets counts as one story with twelve outlets, not as twelve stories.
* **Recycled news reads as fresh news.** Outlets routinely re-run an old development
  with a new headline. Treated as new, it manufactures the appearance of a fresh
  catalyst that no market participant is reacting to, because they read it last week.

The method is **IDF-weighted token overlap** over normalised titles: deterministic,
model-free, and calibrated against measured data rather than guessed.

The obvious approach — Jaccard over bigram shingles — was tried first and measured
against 129 live articles from seven outlets. It failed badly. Outlets rewrite
headlines rather than republishing them verbatim, so bigram overlap collapses even
for unmistakably identical stories: two articles about the same Ray Dalio statement
scored 0.167 against a 0.55 threshold, and the whole batch produced *zero* merges.
The representation, not the threshold, was wrong.

Weighting tokens by inverse document frequency fixes it, because it encodes what
actually identifies a story: shared *rare* tokens. Two headlines both containing
"bitcoin" have said almost nothing — nearly every headline does. Two both containing
"dalio" are almost certainly about the same thing.

The failure mode remains the safe one: under-merging (two variants stay separate)
rather than over-merging unrelated stories.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from mie.core.logging import get_logger
from mie.news.types import NewsItem

log = get_logger(__name__)

__all__ = [
    "Cluster",
    "Deduplicator",
    "document_frequencies",
    "jaccard",
    "normalise_title",
    "shingles",
    "token_weights",
    "tokens",
    "weighted_similarity",
]

#: Titles this similar are treated as the same story.
#:
#: Calibrated against 129 live articles from seven outlets, by reading the ranked
#: cross-outlet pairs and marking which were genuinely the same story:
#:
#:   0.384  Ray Dalio on owning bitcoin        (same story)
#:   0.356  Binance opens trading to AI agents (same story)
#:   0.325  poll on Trump family crypto        (same story)
#:   0.310  CFTC chair on crypto rules         (same story)
#:   0.289  Clarity Act op-ed vs news report   (RELATED, different stories)
#:   0.177  Term Finance exploit               (same story, missed)
#:
#: At 0.30 every merge is correct. Lowering to 0.25 gains the Term Finance pair but
#: also merges the Clarity Act op-ed with a news report about it. Since the stated
#: preference is to under-merge rather than over-merge — a slightly low importance
#: score beats one inflated by wrongly-pooled coverage — 0.30 is the right cut, and
#: the missed pairs are the accepted cost.
_SIMILARITY_THRESHOLD = 0.30

#: Retained for the shingle helpers, which stay available for callers wanting strict
#: phrase matching. The clusterer no longer uses them.
_SHINGLE_SIZE = 2

#: Floor on any token's weight, so no token is ever worth exactly nothing.
#:
#: This is what makes the metric safe at both extremes of corpus size, and both
#: extremes were hit during development:
#:
#: * A hard "ignore tokens above 30% document frequency" cutoff zeroed the shared rare
#:   token that identified a story whenever the batch was small — in six headlines, a
#:   token in two of them is already 33%.
#: * Pure IDF has the same cliff at the top: log((N+1)/(df+1)) is exactly zero for a
#:   token present in every document, so in a two-article batch two *identical*
#:   headlines scored zero similarity.
#:
#: With a floor the weighting degrades gracefully into plain Jaccard when IDF has
#: nothing to say, which is precisely the right fallback.
_MIN_TOKEN_WEIGHT = 0.15

#: An article joining a cluster later than this is a re-run, not coverage of a
#: breaking story. Set well past a normal news cycle so that the ordinary lag between
#: outlets is not mistaken for recycling.
_RECYCLE_AFTER = timedelta(hours=36)

#: Words carrying no distinguishing information in a crypto-news title.
_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "into", "over", "under", "about", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "it's", "his", "her", "their", "our", "your", "my", "we", "they", "he", "she", "you", "i", "not", "no", "nor", "so", "such", "can", "could", "may", "might", "will", "would", "shall", "should", "must", "have", "has", "had", "do", "does", "did", "done", "new", "now", "here", "there", "what", "which", "who", "whom", "how", "why", "when", "where", "all", "any", "both", "each", "few", "more", "most", "other", "some", "only", "own", "same", "too", "very", "just", "also", "amid", "says", "say", "said", "report", "reports", "according"]
)


def normalise_title(title: str) -> str:
    """Lower-case, strip punctuation and outlet furniture, collapse whitespace.

    Removes the trailing outlet suffix that many feeds append (" - CoinDesk"), which
    would otherwise make the same wire story look different at every outlet — exactly
    the comparison this module exists to get right.
    """
    text = title.lower().strip()
    # Trailing " - Outlet" / " | Outlet" attribution.
    text = re.sub(r"\s+[-|–—]\s+[a-z0-9 .']{2,30}$", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(title: str) -> list[str]:
    """Content tokens of a normalised title."""
    return [
        word
        for word in normalise_title(title).split()
        if word not in _STOPWORDS and len(word) > 1
    ]


def shingles(title: str, size: int = _SHINGLE_SIZE) -> frozenset[str]:
    """Overlapping n-grams of content tokens.

    Word order carries real meaning here — "SEC approves ETF" and "ETF approves SEC"
    share every token — so shingles rather than a bag of words.
    """
    words = tokens(title)
    if len(words) < size:
        return frozenset(words)
    return frozenset(
        " ".join(words[i : i + size]) for i in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two shingle sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def document_frequencies(titles: Sequence[str]) -> dict[str, int]:
    """How many titles contain each token."""
    counts: dict[str, int] = {}
    for title in titles:
        for token in set(tokens(title)):
            counts[token] = counts.get(token, 0) + 1
    return counts


def token_weights(titles: Sequence[str]) -> dict[str, float]:
    """Smoothed inverse-document-frequency weight per token.

    ``max(floor, log((N + 1) / (df + 1)))`` — the standard smoothed form with a floor.
    A token shared by two of six headlines stays clearly more informative than one
    shared by four, which is the distinction clustering depends on, and the floor stops
    ubiquitous tokens from becoming literally weightless.
    """
    total = len(titles)
    if not total:
        return {}
    counts = document_frequencies(titles)
    return {
        token: max(_MIN_TOKEN_WEIGHT, math.log((total + 1) / (count + 1)))
        for token, count in counts.items()
    }


def weighted_similarity(
    left: frozenset[str], right: frozenset[str], weights: Mapping[str, float]
) -> float:
    """IDF-weighted Jaccard over token sets.

    A token absent from the corpus statistics gets a high default weight: it is by
    definition rare, and rare tokens are the identifying ones.
    """
    if not left or not right:
        return 0.0
    # An unseen token is by definition rare, so it gets the highest weight present.
    default = max(weights.values(), default=1.0)

    def weight(token: str) -> float:
        return weights.get(token, default)

    shared = sum(weight(t) for t in left & right)
    if shared <= 0:
        return 0.0
    union = sum(weight(t) for t in left | right)
    return shared / union if union > 0 else 0.0


@dataclass(slots=True)
class Cluster:
    """A group of articles telling the same story."""

    cluster_id: str
    items: list[NewsItem] = field(default_factory=list)
    signature: frozenset[str] = field(default_factory=frozenset)

    @property
    def earliest(self) -> NewsItem:
        return min(self.items, key=lambda i: i.published_at)

    @property
    def latest(self) -> NewsItem:
        return max(self.items, key=lambda i: i.published_at)

    @property
    def outlets(self) -> set[str]:
        return {item.source for item in self.items}

    @property
    def representative(self) -> NewsItem:
        """The article that best stands for the cluster.

        The earliest one: it is the story as first reported, before later rewrites
        added interpretation.
        """
        return self.earliest

    @property
    def span(self) -> timedelta:
        return self.latest.published_at - self.earliest.published_at


class Deduplicator:
    """Groups articles into stories and flags re-runs."""

    def __init__(
        self,
        threshold: float = _SIMILARITY_THRESHOLD,
        shingle_size: int = _SHINGLE_SIZE,
        recycle_after: timedelta = _RECYCLE_AFTER,
    ) -> None:
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.recycle_after = recycle_after
        #: Token weights from the most recent clustering pass, reused when comparing
        #: against previously-seen clusters.
        self._weights: dict[str, float] = {}

    def cluster(self, items: Sequence[NewsItem]) -> list[Cluster]:
        """Group articles into stories, oldest first.

        Token weights are derived from the batch being clustered, so "which words
        identify a story" adapts to what this news cycle is about rather than relying
        on a fixed stopword list.

        Greedy single-pass assignment against existing cluster signatures. Quadratic
        in the number of clusters, which is fine for a news feed's volume; if this
        ever handles thousands of items per cycle it becomes MinHash + LSH, and that
        threshold is worth stating rather than discovering.
        """
        ordered = sorted(items, key=lambda i: i.published_at)
        self._weights = token_weights([i.title for i in ordered])

        clusters: list[Cluster] = []
        for item in ordered:
            signature = frozenset(tokens(item.title))
            if not signature:
                continue

            best: Cluster | None = None
            best_score = 0.0
            for cluster in clusters:
                score = weighted_similarity(signature, cluster.signature, self._weights)
                if score > best_score:
                    best, best_score = cluster, score

            if best is not None and best_score >= self.threshold:
                best.items.append(item)
                # Union keeps the signature representative of every phrasing seen, so
                # a third outlet's rewording can still match the group.
                best.signature = best.signature | signature
            else:
                clusters.append(
                    Cluster(
                        cluster_id=_cluster_id(item.title),
                        items=[item],
                        signature=signature,
                    )
                )
        return clusters

    def is_recycled(self, cluster: Cluster, item: NewsItem) -> bool:
        """Whether ``item`` is a re-run rather than coverage of a breaking story."""
        return item.published_at - cluster.earliest.published_at > self.recycle_after

    def find_recycled(
        self, clusters: Sequence[Cluster], known: Sequence[Cluster] = ()
    ) -> dict[str, str]:
        """Map cluster ids that re-tell an older story to the id of that story.

        ``known`` is previously-seen clusters, which is what lets a story reappearing
        days later be recognised rather than treated as new because this fetch has no
        memory of the last one.
        """
        recycled: dict[str, str] = {}
        for cluster in clusters:
            for previous in known:
                if previous.cluster_id == cluster.cluster_id:
                    continue
                similarity = weighted_similarity(
                    cluster.signature, previous.signature, self._weights
                )
                if similarity < self.threshold:
                    continue
                if (
                    cluster.earliest.published_at - previous.earliest.published_at
                    > self.recycle_after
                ):
                    recycled[cluster.cluster_id] = previous.cluster_id
                    break
        return recycled


def _cluster_id(title: str) -> str:
    """Stable id derived from the normalised title.

    Content-derived rather than random so the same story recovers the same id across
    processes and restarts — which is what makes recycled detection work at all.
    """
    return hashlib.blake2b(normalise_title(title).encode(), digest_size=8).hexdigest()
