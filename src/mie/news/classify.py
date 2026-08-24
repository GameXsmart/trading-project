"""News classification: relevance, category, sentiment, importance.

A lexicon-and-rules classifier, and it is worth being direct about why rather than
reaching for a language model.

**There are no labels.** Nobody has annotated these feeds for "category" or "market
sentiment", and the obvious shortcut — have an LLM label a few thousand headlines,
then train on those labels — produces a model that imitates the labeller, complete
with its errors, while making them unauditable. A transparent rule set that an
operator can read, correct, and extend is more useful at this stage than an opaque
one of similar accuracy.

**The output is an input, not a conclusion.** These scores feed a prediction layer
that will itself be measured against realised prices. If the sentiment signal is
worthless, Phase 6-9 will say so with numbers, and no amount of classifier
sophistication would have saved a signal the market ignores. Building the measurement
first is the right order.

What the rules *do* handle carefully:

* **Negation.** "not approved" is not a positive story, and a bag-of-words lexicon
  gets this backwards with confidence.
* **Title weighting.** A headline mention is what a reader reacts to; a passing body
  reference usually is not.
* **Ambiguous tickers.** Several crypto tickers are ordinary English words. Matching
  them naively produces confident nonsense.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from mie.core.logging import get_logger
from mie.news.types import AssetRelevance, EventCategory, Sentiment

log = get_logger(__name__)

__all__ = ["ASSET_ALIASES", "CATEGORY_RULES", "NewsClassifier"]

#: Canonical asset → the words that mean it. Order matters only for readability.
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "bitcoin", "xbt"),
    "ETH": ("eth", "ethereum", "ether"),
    "SOL": ("sol", "solana"),
    "BNB": ("bnb", "binance coin"),
    "XRP": ("xrp", "ripple"),
    "ADA": ("ada", "cardano"),
    "DOGE": ("doge", "dogecoin"),
    "AVAX": ("avax", "avalanche"),
    "LINK": ("link", "chainlink"),
    "DOT": ("dot", "polkadot"),
}

#: Tickers that are also ordinary English words. Matching these as bare tickers
#: produces confident nonsense — "link" appears in half of all articles, and "dot"
#: is punctuation. For these, only the unambiguous long name counts.
_AMBIGUOUS_TICKERS = frozenset({"link", "dot", "ada", "sol"})

#: Category detection, most specific first — an ETF approval is a regulatory event,
#: but "ETF" is the more useful label, so it must be tested before "regulation".
CATEGORY_RULES: tuple[tuple[EventCategory, tuple[str, ...]], ...] = (
    (EventCategory.ETF, ("etf", "exchange-traded fund", "spot etf", "etp")),
    (
        EventCategory.SECURITY_INCIDENT,
        ("hack", "hacked", "exploit", "breach", "stolen", "drained", "rug pull",
         "vulnerability", "attack", "compromised"),
    ),
    (
        EventCategory.ENFORCEMENT,
        ("lawsuit", "sues", "sued", "charges", "indicted", "fined", "penalty",
         "settlement", "subpoena", "investigation", "probe", "sanctions"),
    ),
    (
        EventCategory.REGULATION,
        ("sec ", "regulator", "regulation", "regulatory", "cftc", "legislation",
         "bill", "congress", "senate", "mica", "compliance", "licence", "license"),
    ),
    (
        EventCategory.MACRO,
        ("federal reserve", "fed ", "fomc", "interest rate", "rate cut", "rate hike",
         "inflation", "cpi", "jobs report", "recession", "treasury yield", "gdp"),
    ),
    (
        EventCategory.EXCHANGE,
        ("exchange", "binance", "coinbase", "kraken", "okx", "bybit", "withdrawal",
         "outage", "halts trading", "delisting"),
    ),
    (
        EventCategory.PROTOCOL,
        ("upgrade", "hard fork", "mainnet", "testnet", "staking", "validator",
         "layer 2", "rollup", "consensus", "halving"),
    ),
    (EventCategory.LISTING, ("lists", "listing", "listed on", "trading pair")),
    (
        EventCategory.PARTNERSHIP,
        ("partnership", "partners with", "collaboration", "integrates", "teams up"),
    ),
    (
        EventCategory.ADOPTION,
        ("adoption", "accepts crypto", "treasury", "institutional", "legal tender",
         "payment", "merchant"),
    ),
    (EventCategory.FUNDING, ("raises", "funding round", "series a", "series b", "vc ")),
    (
        EventCategory.MARKET_MOVE,
        ("price", "rally", "surge", "plunge", "crash", "all-time high", "ath",
         "correction", "liquidation", "sell-off"),
    ),
)

#: Sentiment lexicon, weighted. Crypto-specific: "whale" and "burn" carry meaning here
#: that a general-purpose lexicon gets wrong or misses.
_POSITIVE: Mapping[str, float] = {
    "surge": 0.8, "surges": 0.8, "rally": 0.7, "rallies": 0.7, "soar": 0.9, "soars": 0.9,
    "jump": 0.6, "jumps": 0.6, "gain": 0.5, "gains": 0.5, "rise": 0.5, "rises": 0.5,
    "climb": 0.5, "climbs": 0.5, "approve": 0.8, "approved": 0.9, "approval": 0.8,
    "adoption": 0.6, "partnership": 0.5, "upgrade": 0.5, "launch": 0.4, "launches": 0.4,
    "record": 0.6, "high": 0.4, "bullish": 0.9, "breakout": 0.6, "accumulate": 0.5,
    "inflow": 0.6, "inflows": 0.6, "surged": 0.8, "milestone": 0.5, "optimism": 0.6,
    "boost": 0.6, "boosts": 0.6, "wins": 0.6, "won": 0.5, "success": 0.5, "green": 0.3,
    "recovery": 0.6, "rebound": 0.7, "surpasses": 0.6, "backing": 0.4, "surging": 0.8,
}

_NEGATIVE: Mapping[str, float] = {
    "crash": -0.9, "crashes": -0.9, "plunge": -0.9, "plunges": -0.9, "plummet": -0.9,
    "fall": -0.5, "falls": -0.5, "drop": -0.5, "drops": -0.5, "decline": -0.5,
    "slump": -0.7, "tumble": -0.7, "tumbles": -0.7, "sink": -0.6, "sinks": -0.6,
    "hack": -0.9, "hacked": -0.9, "exploit": -0.8, "breach": -0.8, "stolen": -0.8,
    "lawsuit": -0.7, "sues": -0.6, "charges": -0.6, "fined": -0.7, "ban": -0.8,
    "banned": -0.8, "reject": -0.7, "rejected": -0.8, "rejection": -0.7, "delay": -0.5,
    "delayed": -0.5, "bearish": -0.9, "selloff": -0.8, "sell-off": -0.8, "fear": -0.6,
    "liquidated": -0.7, "liquidation": -0.6, "outflow": -0.6, "outflows": -0.6,
    "warning": -0.5, "warns": -0.5, "risk": -0.3, "concern": -0.4, "concerns": -0.4,
    "scam": -0.8, "fraud": -0.8, "collapse": -0.9, "bankruptcy": -0.9, "halt": -0.5,
    "halts": -0.5, "outage": -0.6, "downturn": -0.6, "correction": -0.4, "red": -0.3,
}

#: Words that invert the sentiment of what follows them.
_NEGATIONS = frozenset(
    {"not", "no", "never", "without", "denies", "denied", "fails", "failed",
     "unlikely", "rules out", "refuses", "refused", "halts", "avoids"}
)

#: How far after a negation its effect reaches.
_NEGATION_WINDOW = 3


class NewsClassifier:
    """Turns article text into relevance, category, sentiment and importance."""

    def __init__(self, aliases: Mapping[str, Sequence[str]] = ASSET_ALIASES) -> None:
        self.aliases = {k: tuple(v) for k, v in aliases.items()}

    # ---------------------------------------------------------------- relevance

    def relevance(self, title: str, body: str = "") -> list[AssetRelevance]:
        """Which assets the article is actually about.

        A title mention counts for far more than a body mention: readers and markets
        react to headlines, and a body reference is often just context ("...unlike
        bitcoin, which...").
        """
        title_lower = title.lower()
        body_lower = body.lower()
        found: list[AssetRelevance] = []

        for asset, words in self.aliases.items():
            title_hits = sum(_count(word, title_lower) for word in self._usable(words))
            body_hits = sum(_count(word, body_lower) for word in self._usable(words))
            if not title_hits and not body_hits:
                continue
            # Title presence dominates; body mentions add a little and saturate.
            score = min(1.0, (0.7 if title_hits else 0.0) + min(0.3, body_hits * 0.1))
            found.append(
                AssetRelevance(
                    asset=asset,
                    score=round(score, 3),
                    mentions=title_hits + body_hits,
                    in_title=bool(title_hits),
                )
            )
        return sorted(found, key=lambda r: -r.score)

    def _usable(self, words: Sequence[str]) -> list[str]:
        """Drop aliases that are ordinary English words.

        `link`, `dot`, `sol` and `ada` appear constantly in unrelated prose. Matching
        them as tickers would attach Chainlink to every article containing a
        hyperlink.
        """
        return [w for w in words if w not in _AMBIGUOUS_TICKERS]

    # ----------------------------------------------------------------- category

    def category(self, text: str) -> EventCategory:
        """First matching rule wins; rules are ordered most-specific first."""
        lowered = f" {text.lower()} "
        for category, keywords in CATEGORY_RULES:
            if any(keyword in lowered for keyword in keywords):
                return category
        return EventCategory.OTHER

    # ---------------------------------------------------------------- sentiment

    def sentiment(self, title: str, body: str = "") -> tuple[Sentiment, float, int]:
        """Negation-aware lexicon sentiment.

        Returns the label, a score in [-1, 1], and the number of sentiment-bearing
        words found — the last of which drives confidence: a verdict from one matched
        word is a guess, not a reading.
        """
        title_score, title_hits = self._score_text(title)
        body_score, body_hits = self._score_text(body)

        hits = title_hits + body_hits
        if not hits:
            return Sentiment.NEUTRAL, 0.0, 0

        # The headline is what the market reads.
        if title_hits and body_hits:
            combined = title_score * 0.75 + body_score * 0.25
        else:
            combined = title_score if title_hits else body_score

        combined = max(-1.0, min(1.0, combined))
        return Sentiment.from_score(combined), round(combined, 3), hits

    def _score_text(self, text: str) -> tuple[float, int]:
        words = re.findall(r"[a-z\-]+", text.lower())
        total = 0.0
        hits = 0
        for index, word in enumerate(words):
            weight = _POSITIVE.get(word) or _NEGATIVE.get(word)
            if weight is None:
                continue
            hits += 1
            # "not approved" must not read as positive.
            window = words[max(0, index - _NEGATION_WINDOW) : index]
            if any(w in _NEGATIONS for w in window):
                weight = -weight
            total += weight
        if not hits:
            return 0.0, 0
        # Mean rather than sum: a long article should not out-score a punchy headline
        # merely by containing more words.
        return total / hits, hits

    # --------------------------------------------------------------- importance

    def importance(
        self,
        category: EventCategory,
        coverage: int,
        source_weight: float,
        relevance: Sequence[AssetRelevance],
        sentiment_magnitude: float,
    ) -> float:
        """How much this story matters, in [0, 1].

        Coverage is the strongest term and the most honest one: how many independent
        outlets judged the story worth running is real evidence about its
        significance, gathered before any price data is consulted. It saturates,
        because the difference between one outlet and four is large and the
        difference between eight and twelve is not.
        """
        breadth = min(1.0, 0.35 + 0.22 * (coverage - 1)) if coverage else 0.35
        specificity = max((r.score for r in relevance), default=0.35)
        strength = 0.6 + 0.4 * min(1.0, sentiment_magnitude)

        score = (
            category.base_importance * 0.40
            + breadth * 0.30
            + specificity * 0.15
            + source_weight * 0.15
        ) * strength
        return round(max(0.0, min(1.0, score)), 3)

    def confidence(
        self,
        sentiment_hits: int,
        relevance: Sequence[AssetRelevance],
        category: EventCategory,
    ) -> float:
        """How much to trust this classification.

        Low when the sentiment verdict rests on one word, when no asset could be
        identified, or when no category rule matched — all situations where the
        classifier is guessing and should say so.
        """
        lexical = min(1.0, sentiment_hits / 4.0)
        targeted = 1.0 if any(r.in_title for r in relevance) else (
            0.6 if relevance else 0.3
        )
        categorised = 0.4 if category is EventCategory.OTHER else 1.0
        return round(max(0.0, min(1.0, 0.25 + 0.75 * lexical * targeted * categorised)), 3)


def _count(word: str, text: str) -> int:
    """Whole-word occurrences, so 'eth' does not match 'ethics' or 'together'."""
    return len(re.findall(rf"\b{re.escape(word)}\b", text))
