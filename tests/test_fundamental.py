"""Tests del agente fundamental: léxico, RSS, calendario, RAG y bypass del LLM."""
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from agents.economic_calendar import (EconEvent, EconomicCalendar, generate_sample_calendar, parse_number,
                                      refresh_from_web, symbol_currencies)
from agents.fundamental_agent import FundamentalAgent
from agents.news import (FeedIngestor, NewsItem, aggregate, lexicon_score, load_sample_news, tag_entities)
from core.llm import LLMClient, MockLLM, extract_json, make_llm
from core.types import Direction
from memory.embeddings import HashingEmbedder
from memory.vector_store import NumpyVectorStore

NOW = datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc)
REAL_DATA = Path(__file__).resolve().parent.parent / "data"

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Dollar rallies after strong US payrolls</title><link>http://x/1</link>
<description>&lt;p&gt;The &lt;b&gt;dollar&lt;/b&gt; jumped.&lt;/p&gt;</description><pubDate>Fri, 18 Sep 2026 13:00:00 GMT</pubDate></item>
<item><title>Old news about gold</title><link>http://x/2</link><description>x</description><pubDate>Mon, 01 Jun 2026 13:00:00 GMT</pubDate></item>
<item><title>Dollar rallies after strong US payrolls</title><link>http://x/1b</link><description>dupe</description><pubDate>Fri, 18 Sep 2026 13:05:00 GMT</pubDate></item>
</channel></rss>""".encode()


def test_tag_entities_distinguishes_dollars():
    assert tag_entities("Canadian dollar gains as oil surges") == ["CAD", "OIL"]
    assert "USD" in tag_entities("Dollar slips as Fed cuts") and "CAD" not in tag_entities("Dollar slips as Fed cuts")
    assert set(tag_entities("Gold and bitcoin rally")) == {"XAU", "BTC"}


@pytest.mark.parametrize("text,entity,sign", [
    ("Fed signals rate hike as inflation hotter than expected", "USD", 1),
    ("ECB signals rate cuts as eurozone slips into recession", "EUR", -1),
    ("Dollar rallies on hawkish Powell", "USD", 1),
    ("Yen weakens as BoJ stays dovish", "JPY", -1),
    ("Fed signals rate hike as inflation hotter than expected", "XAU", -1),       # halcón = malo para el oro
    ("Gold jumps as safe haven demand rises", "XAU", 1),
    ("Gold slips as stronger dollar weighs", "XAU", -1),
    ("Bitcoin ETF inflows hit record, crypto rallies", "BTC", 1),
    ("Crypto crackdown: regulators ban exchange, hack drains funds", "BTC", -1),
])
def test_lexicon_direction(text, entity, sign):
    s = lexicon_score(text, entity)
    assert s * sign > 0.2, (text, entity, s)


def test_lexicon_negation_and_neutral():
    assert lexicon_score("Fed does not signal a rate hike", "USD") < lexicon_score("Fed signals a rate hike", "USD")
    assert lexicon_score("Markets await the weekly inventory report", "USD") == 0.0
    assert -1.0 <= lexicon_score("rate hike hawkish beats expectations strong jobs rate hike", "USD") <= 1.0


def test_movement_word_counts_only_near_entity():
    # "surge" describe a las acciones, no al dólar
    assert lexicon_score("US stocks surge as Fed holds", "USD") == 0.0
    assert lexicon_score("Dollar surges", "USD") > 0.2


def test_aggregate_recency_and_shrinkage():
    fresh = NewsItem("a", "", "reuters", NOW - timedelta(minutes=10))
    stale = NewsItem("b", "", "reuters", NOW - timedelta(hours=24))
    s_fresh, c_fresh = aggregate([(fresh, 0.8)], NOW)
    s_stale, c_stale = aggregate([(stale, 0.8)], NOW)
    assert s_fresh > s_stale > 0 and c_fresh > c_stale
    assert s_fresh < 100 * 0.8                              # shrinkage: una sola noticia no alcanza el máximo
    assert aggregate([], NOW) == (0.0, 0.0)


def test_feed_ingestor_parses_filters_dedups():
    calls = []
    fetch = lambda url: (calls.append(url), RSS)[1]
    ing = FeedIngestor({"reuters": ["http://feed"]}, fetcher=fetch, max_age_hours=48)
    items = ing.fetch_all(NOW)
    assert len(items) == 1, "descarta lo antiguo y deduplica por titular"
    assert items[0].summary.startswith("The dollar jumped") and "USD" in items[0].entities


def test_feed_failure_is_isolated_and_fallback_url_used():
    def fetch(url):
        if "bad" in url:
            raise ConnectionError("boom")
        return RSS
    ing = FeedIngestor({"a": ["http://bad", "http://good"], "b": ["http://bad2"]}, fetcher=fetch)
    items = ing.fetch_all(NOW)
    assert len(items) == 1 and "b:http://bad2" in ing.last_errors and "a:http://bad" in ing.last_errors


def test_parse_number():
    assert parse_number("2.5%") == 2.5 and parse_number("187K") == 187_000 and parse_number("1,5M") == 1.5e6
    assert parse_number("abc") is None and parse_number(None) is None


def _cal():
    return EconomicCalendar([
        EconEvent(NOW + timedelta(minutes=20), "USD", "high", "Non-Farm Employment Change", "180K"),
        EconEvent(NOW - timedelta(hours=2), "EUR", "medium", "German Flash PMI", "50.0", "49.0", "51.0"),
        EconEvent(NOW + timedelta(hours=5), "GBP", "low", "Whatever", None),
    ])


def test_blackout_and_tags():
    cal = _cal()
    ev = cal.blackout("EURUSD", NOW, 30, 30, "high")
    assert ev and ev.tag == "NFP"
    assert cal.blackout("EURUSD", NOW - timedelta(hours=3), 30, 30, "high") is None
    assert cal.blackout("EURJPY", NOW, 30, 30, "high") is None       # ni EUR ni JPY afectados por NFP
    assert "NFP" in cal.tags_at(NOW, "EURUSD") and "NFP" not in cal.tags_at(NOW, "EURGBP")
    assert symbol_currencies("XAUUSD") == ["XAU", "USD"]


def test_surprise_direction_and_inverse_series():
    cal = _cal()
    assert cal.surprise("EUR", NOW) > 0                               # PMI 51 vs 50 previsto
    cal2 = EconomicCalendar([EconEvent(NOW - timedelta(hours=1), "USD", "high", "Unemployment Rate", "4.0%", "4.0%", "4.4%")])
    assert cal2.surprise("USD", NOW) < 0                              # más paro = malo para USD
    assert cal.surprise("JPY", NOW) == 0.0


def test_calendar_json_roundtrip_and_merge(tmp_path):
    cal = _cal()
    p = tmp_path / "c.json"
    cal.to_json(p, sample=True)
    back = EconomicCalendar.from_json(p)
    assert [e.title for e in back.events] == [e.title for e in cal.events]
    back.merge([EconEvent(NOW + timedelta(minutes=20), "USD", "high", "Non-Farm Employment Change", "180K", None, "250K")])
    assert len(back.events) == 3 and [e for e in back.events if e.currency == "USD"][0].actual == "250K"
    assert EconomicCalendar.from_json(tmp_path / "missing.json").events == []


def test_refresh_from_web_normalizes():
    rows = [{"title": "CPI m/m", "country": "USD", "date": "2026-09-18T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
            {"title": "Bank Holiday", "country": "JPY", "date": "2026-09-18T00:00:00-04:00", "impact": "Holiday"}]
    evs = refresh_from_web(fetcher=lambda u: rows)
    assert len(evs) == 1 and evs[0].time == datetime(2026, 9, 18, 12, 30, tzinfo=timezone.utc) and evs[0].tag == "CPI"


def test_sample_calendar_has_nfp_on_first_friday():
    evs = generate_sample_calendar(datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 10, 31, tzinfo=timezone.utc))
    nfp = [e for e in evs if e.tag == "NFP" and e.currency == "USD" and "Non-Farm" in e.title]
    assert [e.time.day for e in nfp] == [4, 2] and all(e.time.weekday() == 4 for e in nfp)


@pytest.fixture()
def agent(settings, tmp_path):
    store = NumpyVectorStore("news_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
    ing = FeedIngestor({}, fetcher=lambda u: b"")
    return FundamentalAgent(settings, MockLLM(), store=store, calendar=EconomicCalendar(), ingestor=ing, offline_samples=True)


def _items():
    mk = lambda t, m: NewsItem(t, "", "reuters", NOW - timedelta(minutes=m), entities=tag_entities(t))
    return [
        mk("Dollar rallies after Fed signals rate hike, hawkish Powell", 20),
        mk("Strong US payrolls beat expectations, dollar jumps", 50),
        mk("Euro slips as ECB signals rate cuts amid recession fears", 30),
        mk("Eurozone economy contracts, euro weakens", 90),
        mk("Gold jumps as safe haven demand rises", 25),
    ]


def test_ingest_is_idempotent(agent):
    assert agent.ingest(_items()) == 5 and agent.ingest(_items()) == 0


def test_retrieve_filters_entity_and_age(agent):
    agent.ingest(_items() + [NewsItem("Dollar surges old", "", "reuters", NOW - timedelta(hours=100), entities=["USD"])])
    got = agent.retrieve("EUR", NOW)
    assert got and all("EUR" in i.entities for i in got)
    assert all(i.age_hours(NOW) <= 36 for i in agent.retrieve("USD", NOW))


def test_sentiment_eurusd_bearish_and_xauusd_positive_signs(agent):
    agent.ingest(_items())
    r = agent.sentiment("EURUSD", NOW)
    assert r.score < -25 and r.per_entity["USD"] > 0 > r.per_entity["EUR"] and -100 <= r.score <= 100
    assert r.method == "lexicon" and r.drivers
    sig = agent.analyze("EURUSD", NOW)
    assert sig.direction == Direction.SHORT and 0 < sig.confidence <= 1
    assert agent.sentiment("XAUUSD", NOW).per_entity["XAU"] > 0


def test_no_news_is_neutral_flat(agent):
    r = agent.sentiment("GBPJPY", NOW)
    assert r.score == 0.0 and agent.analyze("GBPJPY", NOW).direction == Direction.FLAT


def test_calendar_surprise_moves_score(settings, tmp_path):
    store = NumpyVectorStore("n_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
    cal = EconomicCalendar([EconEvent(NOW - timedelta(hours=1), "USD", "high", "Non-Farm Employment Change", "180K", "175K", "260K")])
    ag = FundamentalAgent(settings, MockLLM(), store=store, calendar=cal, ingestor=FeedIngestor({}, fetcher=lambda u: b""))
    assert ag.sentiment("EURUSD", NOW).per_entity["USD"] > 10


def test_refresh_offline_uses_samples_only_when_allowed(settings, tmp_path):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_DATA / "sample_news.json", settings.data_dir / "sample_news.json")

    def mk(offline):
        store = NumpyVectorStore("r_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
        return FundamentalAgent(settings, MockLLM(), store=store, calendar=EconomicCalendar(),
                                ingestor=FeedIngestor({}, fetcher=lambda u: b""), offline_samples=offline)
    assert mk(True).refresh(NOW) > 10 and mk(False).refresh(NOW) == 0


def test_sample_news_file_is_valid():
    items = load_sample_news(REAL_DATA / "sample_news.json", NOW)
    assert len(items) >= 15 and all(i.entities for i in items)


class FakeLLM(LLMClient):
    is_mock = False
    name = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, prompt, system=None):
        self.calls += 1
        return self.reply


def test_extract_json_tolerant():
    assert extract_json('Claro:\n```json\n{"score": 42, "confidence": 0.8}\n```') == {"score": 42, "confidence": 0.8}
    with pytest.raises(ValueError):
        extract_json("no json aquí")


def test_llm_score_used_clipped_and_cached(settings, tmp_path):
    store = NumpyVectorStore("l_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
    llm = FakeLLM('{"score": 250, "confidence": 0.9, "reasoning": "hawkish Fed"}')
    ag = FundamentalAgent(settings, llm, store=store, calendar=EconomicCalendar(), ingestor=FeedIngestor({}, fetcher=lambda u: b""))
    ag.ingest(_items())
    r1 = ag.sentiment("USDJPY", NOW)
    assert r1.method == "llm" and r1.per_entity["USD"] == 100.0           # recortado a 100
    calls = llm.calls
    ag.sentiment("USDJPY", NOW)
    assert llm.calls == calls, "la segunda consulta usa caché"


def test_llm_garbage_falls_back_to_lexicon(settings, tmp_path):
    store = NumpyVectorStore("g_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
    ag = FundamentalAgent(settings, FakeLLM("lo siento, no puedo"), store=store, calendar=EconomicCalendar(),
                          ingestor=FeedIngestor({}, fetcher=lambda u: b""))
    ag.ingest(_items())
    r = ag.sentiment("EURUSD", NOW)
    assert r.method == "lexicon" and r.score < 0


def test_make_llm_bypass_without_key(settings):
    assert isinstance(make_llm(settings), MockLLM) and make_llm(settings).is_mock
