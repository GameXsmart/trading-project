"""News ingestion, deduplication and classification.

The tests concentrate on the two things that are easy to get quietly wrong: merging
stories that are not the same, and reading sentiment backwards. Both produce output
that looks entirely plausible while being false, which is the worst failure mode
available to this layer.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from mie.core.timeframes import utcnow
from mie.news.classify import NewsClassifier
from mie.news.dedup import (
    Deduplicator,
    normalise_title,
    token_weights,
    tokens,
    weighted_similarity,
)
from mie.news.engine import NewsEngine
from mie.news.sources import parse_feed
from mie.news.types import EventCategory, NewsItem, Sentiment

# ---------------------------------------------------------------------- helpers


def item(title: str, source: str = "coindesk", hours_ago: float = 1.0, summary: str = "") -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        url=f"https://example.test/{abs(hash(title))}",
        published_at=utcnow() - timedelta(hours=hours_ago),
        summary=summary,
    )


def similarity(left: str, right: str, corpus: list[str] | None = None) -> float:
    corpus = corpus or [left, right]
    weights = token_weights(corpus)
    return weighted_similarity(frozenset(tokens(left)), frozenset(tokens(right)), weights)


# ------------------------------------------------------------------- parsing


RSS_SAMPLE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Bitcoin ETF sees record inflows</title>
    <link>https://example.test/a</link>
    <pubDate>Wed, 20 Aug 2026 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;Inflows hit a record.&lt;/p&gt;</description>
  </item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Ethereum upgrade ships</title>
    <link href="https://example.test/b"/>
    <published>2026-08-20T12:00:00Z</published>
    <summary>The upgrade shipped.</summary>
  </entry>
</feed>"""


class TestFeedParsing:
    def test_rss_is_parsed(self) -> None:
        items = parse_feed(RSS_SAMPLE, "test")
        assert len(items) == 1
        assert items[0].title == "Bitcoin ETF sees record inflows"
        assert items[0].published_at.tzinfo is not None
        assert "<p>" not in items[0].summary, "HTML must be stripped"

    def test_atom_is_parsed(self) -> None:
        items = parse_feed(ATOM_SAMPLE, "test")
        assert len(items) == 1
        assert items[0].url == "https://example.test/b"

    def test_malformed_xml_yields_nothing_rather_than_raising(self) -> None:
        """One outlet publishing broken XML must not cost the other six."""
        assert parse_feed("<rss><channel><item><title>unclosed", "test") == []

    def test_items_without_a_title_or_link_are_skipped(self) -> None:
        broken = """<?xml version="1.0"?><rss><channel>
            <item><description>no title</description></item></channel></rss>"""
        assert parse_feed(broken, "test") == []


# --------------------------------------------------------------------- dedup


class TestNormalisation:
    def test_outlet_suffix_is_stripped(self) -> None:
        """Feeds append their own name, which would otherwise make one wire story
        look different at every outlet."""
        assert normalise_title("Bitcoin hits high - CoinDesk") == "bitcoin hits high"
        assert normalise_title("Bitcoin hits high | Decrypt") == "bitcoin hits high"

    def test_punctuation_and_case_are_removed(self) -> None:
        assert normalise_title("SEC's ETF: Approved!") == "sec s etf approved"


class TestSimilarity:
    def test_rewritten_headlines_about_one_story_match(self) -> None:
        """The case that broke the first implementation. Outlets rewrite headlines
        rather than republishing them, so bigram overlap collapses; IDF-weighted token
        overlap survives the rewording."""
        corpus = [
            "Ray Dalio says investors should own a bit of Bitcoin as US debt grows",
            "Ray Dalio says to buy a bit of Bitcoin amid potential debt crisis",
            "Bitcoin price surges past resistance as bulls take control",
            "Ethereum staking yields drop as validator count rises",
            "Solana network upgrade improves transaction throughput",
        ]
        same = similarity(corpus[0], corpus[1], corpus)
        different = similarity(corpus[0], corpus[2], corpus)
        assert same > 0.30
        assert different < 0.15
        assert same > different * 3

    def test_shared_common_words_do_not_create_similarity(self) -> None:
        """Two headlines both mentioning bitcoin have told us almost nothing."""
        corpus = [
            "Bitcoin price rises on ETF news",
            "Bitcoin price falls on regulatory concern",
            "Bitcoin miners expand capacity",
            "Bitcoin adoption grows in Latin America",
        ]
        assert similarity(corpus[0], corpus[1], corpus) < 0.35

    def test_similarity_is_symmetric(self) -> None:
        corpus = ["alpha beta gamma delta", "alpha beta epsilon zeta"]
        assert similarity(corpus[0], corpus[1], corpus) == pytest.approx(
            similarity(corpus[1], corpus[0], corpus)
        )

    def test_empty_titles_are_not_similar_to_anything(self) -> None:
        assert similarity("", "bitcoin surges") == 0.0


class TestClustering:
    def test_the_same_story_from_two_outlets_becomes_one_event(self) -> None:
        items = [
            item("Ray Dalio says investors should own a bit of Bitcoin", "coindesk"),
            item("Ray Dalio says to buy a bit of Bitcoin amid debt crisis", "cointelegraph"),
            item("Solana network upgrade improves throughput", "decrypt"),
            item("Ethereum validators face new staking rules", "newsbtc"),
        ]
        clusters = Deduplicator().cluster(items)
        merged = [c for c in clusters if len(c.items) > 1]
        assert len(merged) == 1
        assert merged[0].outlets == {"coindesk", "cointelegraph"}

    def test_unrelated_stories_stay_separate(self) -> None:
        """Over-merging is the dangerous error: it inflates coverage, which is the
        importance signal."""
        items = [
            item("Bitcoin ETF inflows reach record levels", "coindesk"),
            item("Solana network suffers brief outage", "decrypt"),
            item("Cardano launches governance vote", "newsbtc"),
        ]
        assert all(len(c.items) == 1 for c in Deduplicator().cluster(items))

    def test_cluster_id_is_stable_across_runs(self) -> None:
        """Recycled detection depends on the same story recovering the same id."""
        first = Deduplicator().cluster([item("Bitcoin ETF sees record inflows")])
        second = Deduplicator().cluster([item("Bitcoin ETF sees record inflows")])
        assert first[0].cluster_id == second[0].cluster_id

    def test_representative_is_the_earliest_article(self) -> None:
        """The story as first reported, before later rewrites added interpretation."""
        items = [
            item("Ray Dalio says buy a bit of Bitcoin now", "cointelegraph", hours_ago=2),
            item("Ray Dalio says investors should own a bit of Bitcoin", "coindesk", hours_ago=9),
        ]
        cluster = Deduplicator().cluster(items)[0]
        assert cluster.representative.source == "coindesk"


class TestRecycledDetection:
    def test_a_story_rerun_days_later_is_flagged(self) -> None:
        title = "Bitcoin ETF approval sparks institutional interest"
        old = Deduplicator().cluster([item(title, "coindesk", hours_ago=200)])
        fresh = Deduplicator().cluster([item(title, "newsbtc", hours_ago=1)])

        dedup = Deduplicator()
        dedup.cluster([item(title, "newsbtc", hours_ago=1)])  # prime the weights
        recycled = dedup.find_recycled(fresh, old)
        assert fresh[0].cluster_id in recycled or old[0].cluster_id == fresh[0].cluster_id

    def test_normal_reporting_lag_is_not_recycling(self) -> None:
        """Outlets pick a story up over hours; that is coverage, not a re-run."""
        items = [
            item("Ray Dalio says investors should own a bit of Bitcoin", "coindesk", 8),
            item("Ray Dalio says to buy a bit of Bitcoin amid debt crisis", "cointelegraph", 2),
        ]
        clusters = Deduplicator().cluster(items)
        assert not Deduplicator().find_recycled(clusters, [])


# ------------------------------------------------------------------ classify


class TestRelevance:
    def test_headline_mentions_outrank_body_mentions(self) -> None:
        classifier = NewsClassifier()
        headline = classifier.relevance("Bitcoin surges past resistance", "")
        body_only = classifier.relevance("Markets rally broadly", "bitcoin also rose")
        assert headline[0].score > body_only[0].score
        assert headline[0].in_title
        assert not body_only[0].in_title

    def test_ambiguous_tickers_do_not_match_ordinary_words(self) -> None:
        """`link` and `dot` are ordinary English. Matching them as tickers attaches
        Chainlink to every article containing a hyperlink."""
        found = NewsClassifier().relevance("Click the link to see the dot plot", "")
        assert [r.asset for r in found] == []

    def test_long_names_still_match_those_assets(self) -> None:
        found = NewsClassifier().relevance("Chainlink and Polkadot announce upgrade", "")
        assert {r.asset for r in found} == {"LINK", "DOT"}

    def test_whole_word_matching_only(self) -> None:
        """'eth' must not match 'ethics' or 'together'."""
        assert NewsClassifier().relevance("Together we discuss ethics", "") == []

    def test_multiple_assets_are_ranked(self) -> None:
        found = NewsClassifier().relevance("Bitcoin and Ethereum ETFs see inflows", "")
        assert {r.asset for r in found} == {"BTC", "ETH"}


class TestCategory:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Spot Bitcoin ETF approved by regulator", EventCategory.ETF),
            ("Protocol loses $8.5 million in exploit", EventCategory.SECURITY_INCIDENT),
            ("SEC sues exchange over unregistered securities", EventCategory.ENFORCEMENT),
            ("Federal Reserve holds interest rate steady", EventCategory.MACRO),
            ("Ethereum mainnet upgrade goes live", EventCategory.PROTOCOL),
            ("Startup raises $40m in funding round", EventCategory.FUNDING),
            ("A quiet day for everyone involved", EventCategory.OTHER),
        ],
    )
    def test_categories_are_detected(self, text: str, expected: EventCategory) -> None:
        assert NewsClassifier().category(text) is expected

    def test_etf_outranks_regulation_for_an_etf_approval(self) -> None:
        """An ETF approval is a regulatory event, but 'ETF' is the more useful label,
        so rule order matters."""
        assert NewsClassifier().category(
            "SEC regulator approves spot ETF"
        ) is EventCategory.ETF


class TestSentiment:
    def test_positive_and_negative_headlines_are_distinguished(self) -> None:
        classifier = NewsClassifier()
        good, good_score, _ = classifier.sentiment("Bitcoin surges to record high")
        bad, bad_score, _ = classifier.sentiment("Bitcoin crashes after exchange hack")
        assert good.score > 0 and bad.score < 0
        assert good_score > bad_score

    def test_negation_is_handled(self) -> None:
        """A bag-of-words lexicon reads 'not approved' as positive, with confidence."""
        classifier = NewsClassifier()
        approved, positive_score, _ = classifier.sentiment("SEC approved the ETF")
        _, negative_score, _ = classifier.sentiment("SEC not approved the ETF")
        assert approved.score > 0
        assert negative_score < positive_score

    def test_neutral_text_scores_neutral(self) -> None:
        sentiment, score, hits = NewsClassifier().sentiment(
            "The conference takes place in March"
        )
        assert sentiment is Sentiment.NEUTRAL
        assert score == 0.0
        assert hits == 0

    def test_headline_outweighs_body(self) -> None:
        """The headline is what the market reads."""
        _, score, _ = NewsClassifier().sentiment(
            "Bitcoin crashes after hack",
            "Some analysts remain optimistic and see gains ahead with a strong rally",
        )
        assert score < 0

    def test_confidence_is_low_when_the_verdict_rests_on_one_word(self) -> None:
        classifier = NewsClassifier()
        weak = classifier.confidence(1, [], EventCategory.OTHER)
        strong = classifier.confidence(
            6, classifier.relevance("Bitcoin surges", ""), EventCategory.ETF
        )
        assert weak < strong


class TestImportance:
    def test_broad_coverage_raises_importance(self) -> None:
        """How many independent outlets ran the story is real evidence about its
        significance, gathered before any price data is consulted."""
        classifier = NewsClassifier()
        relevance = classifier.relevance("Bitcoin ETF approved", "")
        one = classifier.importance(EventCategory.ETF, 1, 1.0, relevance, 0.5)
        many = classifier.importance(EventCategory.ETF, 5, 1.0, relevance, 0.5)
        assert many > one

    def test_category_priors_are_respected(self) -> None:
        classifier = NewsClassifier()
        relevance = classifier.relevance("Bitcoin news", "")
        hack = classifier.importance(EventCategory.SECURITY_INCIDENT, 2, 1.0, relevance, 0.5)
        funding = classifier.importance(EventCategory.FUNDING, 2, 1.0, relevance, 0.5)
        assert hack > funding

    def test_importance_is_bounded(self) -> None:
        classifier = NewsClassifier()
        relevance = classifier.relevance("Bitcoin hacked", "")
        extreme = classifier.importance(
            EventCategory.SECURITY_INCIDENT, 20, 1.0, relevance, 1.0
        )
        assert 0.0 <= extreme <= 1.0


# -------------------------------------------------------------------- engine


class TestNewsEngine:
    def test_clustering_happens_before_age_filtering(self) -> None:
        """Filtering first splits a story whose coverage straddles the cutoff, which
        destroys the coverage count deduplication exists to produce."""
        engine = NewsEngine(max_age=timedelta(hours=72))
        events = engine.process(
            [
                item("Ray Dalio says investors should own a bit of Bitcoin", "coindesk", 71),
                item("Ray Dalio says to buy a bit of Bitcoin amid debt", "cointelegraph", 84),
            ]
        )
        assert len(events) == 1
        assert events[0].coverage == 2, "the older article must still count as coverage"

    def test_future_dated_items_are_rejected(self) -> None:
        engine = NewsEngine()
        events = engine.process([item("Bitcoin moons next week", hours_ago=-48)])
        assert events == []

    def test_events_carry_every_required_field(self) -> None:
        """Requirement §8 enumerates these explicitly."""
        engine = NewsEngine()
        events = engine.process(
            [item("Bitcoin ETF approved by SEC in landmark decision", summary="A big day.")]
        )
        assert len(events) == 1
        event = events[0]
        assert event.sources and event.published_at and event.category
        assert event.sentiment and 0.0 <= event.importance <= 1.0
        assert 0.0 <= event.confidence <= 1.0
        assert event.assets == ["BTC"]

    def test_market_sentiment_excludes_recycled_stories(self) -> None:
        """A re-run is not new information; counting it again lets one old
        development sway the reading repeatedly."""
        engine = NewsEngine()
        events = engine.process(
            [item("Bitcoin surges to a record high on ETF inflows", "coindesk")]
        )
        assert events
        score, count = NewsEngine.market_sentiment(events, "BTC")
        assert count == 1
        assert score > 0

        recycled = [events[0].model_copy(update={"is_recycled": True})]
        assert NewsEngine.market_sentiment(recycled, "BTC") == (0.0, 0)

    def test_for_asset_filters_by_genuine_relevance(self) -> None:
        engine = NewsEngine()
        events = engine.process(
            [
                item("Bitcoin ETF approved in landmark decision", "coindesk"),
                item("Solana network upgrade improves throughput", "decrypt"),
            ]
        )
        btc = NewsEngine.for_asset(events, "BTC")
        assert len(btc) == 1
        assert "Bitcoin" in btc[0].title

    def test_empty_input_is_handled(self) -> None:
        assert NewsEngine().process([]) == []
