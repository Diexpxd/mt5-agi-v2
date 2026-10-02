"""Noticias: modelo de datos, etiquetado por entidad, ingesta RSS y sentimiento léxico (bypass sin LLM)."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

_NOT_USD = r"(?<!canadian )(?<!australian )(?<!new zealand )(?<!hong kong )(?<!singapore )"
ENTITY_PATTERNS: Dict[str, str] = {
    "USD": _NOT_USD + r"\b(dollar|usd|greenback|fed|federal reserve|fomc|powell|treasury|treasuries|payrolls|nfp|"
                      r"us cpi|us inflation|us jobs)\b",
    "EUR": r"\b(euro|eur|ecb|lagarde|eurozone|euro zone|bundesbank|bund|german|germany)\b",
    "GBP": r"\b(pound|sterling|gbp|boe|bank of england|bailey|gilt|gilts|uk economy|uk inflation)\b",
    "JPY": r"\b(yen|jpy|boj|bank of japan|ueda|japan|japanese)\b",
    "CAD": r"\b(loonie|cad|canadian dollar|bank of canada|boc|canada|canadian)\b",
    "AUD": r"\b(aussie|aud|rba|reserve bank of australia|australia|australian)\b",
    "NZD": r"\b(kiwi|nzd|rbnz|new zealand)\b",
    "CHF": r"\b(swiss franc|chf|snb|swiss national bank|switzerland|swiss)\b",
    "XAU": r"\b(gold|xau|bullion|safe[- ]haven)\b",
    "BTC": r"\b(bitcoin|btc|crypto|cryptocurrency|spot etf)\b",
    "ETH": r"\b(ethereum|ether|eth)\b",
    "OIL": r"\b(oil|crude|brent|wti|opec)\b",
}
_ENTITY_RE = {k: re.compile(v, re.IGNORECASE) for k, v in ENTITY_PATTERNS.items()}


def tag_entities(text: str) -> List[str]:
    """Entidades (divisas/activos) mencionadas en ``text``."""
    return [k for k, rx in _ENTITY_RE.items() if rx.search(text)]


def symbol_entities(symbol: str) -> tuple[str, str]:
    """``EURUSD -> ('EUR','USD')``; ``XAUUSD -> ('XAU','USD')``; ``BTCUSD -> ('BTC','USD')``."""
    s = symbol.upper().replace("/", "")
    return s[:3], s[3:6]


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    published: datetime
    link: str = ""
    entities: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return hashlib.sha1(re.sub(r"\W+", " ", self.title.lower()).strip().encode()).hexdigest()[:20]

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()

    def age_hours(self, now: datetime) -> float:
        return max((now - self.published).total_seconds() / 3600.0, 0.0)


_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", s or ""))).strip()


DEFAULT_FEEDS: Dict[str, List[str]] = {
    "bloomberg": ["https://feeds.bloomberg.com/markets/news.rss"],
    "reuters": ["https://news.google.com/rss/search?q=site:reuters.com+markets+OR+forex+OR+fed&hl=en-US&gl=US&ceid=US:en"],
    # ForexLive pasó a llamarse investingLive; se prueban ambas URL.
    "forexlive": ["https://investinglive.com/feed", "https://www.forexlive.com/feed/news"],
}


def load_feeds(path: Path) -> Dict[str, List[str]]:
    """Feeds desde ``data/feeds.json`` (``{"nombre": ["url", ...]}``) con valores por defecto."""
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items() if not k.startswith("_")}
        except Exception as exc:
            log.warning("feeds.json inválido (%s); usando feeds por defecto", exc)
    return dict(DEFAULT_FEEDS)


def _http_fetch(url: str, timeout: float = 10.0) -> bytes:
    from core.net import http_get

    return http_get(url, timeout).content


class FeedIngestor:
    """Descarga y normaliza feeds RSS/Atom. Cada feed falla de forma aislada (nunca tumba al resto)."""

    def __init__(self, feeds: Dict[str, List[str]], fetcher: Callable[[str], bytes] = _http_fetch,
                 max_age_hours: float = 48.0) -> None:
        self.feeds = feeds
        self._fetch = fetcher
        self.max_age_hours = max_age_hours
        self.last_errors: Dict[str, str] = {}

    def fetch_all(self, now: datetime | None = None) -> List[NewsItem]:
        import feedparser

        now = now or datetime.now(timezone.utc)
        seen: Dict[str, NewsItem] = {}
        self.last_errors = {}
        for source, urls in self.feeds.items():
            got = False
            for url in urls:
                try:
                    parsed = feedparser.parse(self._fetch(url))
                    entries = parsed.entries or []
                    for e in entries:
                        ts = e.get("published_parsed") or e.get("updated_parsed")
                        pub = datetime(*ts[:6], tzinfo=timezone.utc) if ts else now
                        if pub > now + timedelta(hours=1) or (now - pub).total_seconds() > self.max_age_hours * 3600:
                            continue
                        item = NewsItem(clean_html(e.get("title", "")), clean_html(e.get("summary", ""))[:600],
                                        source, pub, e.get("link", ""))
                        if not item.title:
                            continue
                        item.entities = tag_entities(item.text)
                        seen.setdefault(item.id, item)
                    got = got or bool(entries)
                    if entries:
                        break
                except Exception as exc:
                    self.last_errors[f"{source}:{url}"] = repr(exc)[:160]
                    log.warning("Feed %s (%s) falló: %s", source, url, exc)
            if not got:
                log.info("Feed %s sin datos", source)
        return sorted(seen.values(), key=lambda i: i.published, reverse=True)


def load_sample_news(path: Path, now: datetime) -> List[NewsItem]:
    """Titulares de muestra con antigüedad relativa (solo modo offline/mock)."""
    if not path.exists():
        return []
    items = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        it = NewsItem(row["title"], row.get("summary", ""), row.get("source", "sample"),
                      now - timedelta(minutes=float(row.get("age_minutes", 30))))
        it.entities = tag_entities(it.text)
        items.append(it)
    return items


_POLICY_DATA = [
    (r"rate hike|hikes? (?:interest )?rates?|raises? (?:interest )?rates?|tighten(?:s|ing)?", 1.0),
    (r"hawkish|higher for longer", 0.9),
    (r"hotter than expected|above expectations|beats? (?:expectations|forecasts?)|stronger than expected|"
     r"better than expected|tops? (?:estimates|forecasts)", 0.7),
    (r"strong (?:jobs|payrolls|data|growth)|payrolls beat|robust (?:growth|jobs|data)", 0.6),
    (r"inflation (?:rises|accelerates|surges|jumps|heats up)", 0.5),
    (r"growth accelerates|expands|expansion", 0.4),
    (r"rate cuts?|cuts? (?:interest )?rates?|easing|stimulus", -1.0),
    (r"dovish|pause[sd]? (?:its )?(?:rate )?hikes?", -0.9),
    (r"cooler than expected|below expectations|(?:misses|missed) (?:expectations|forecasts?)|weaker than expected|"
     r"worse than expected", -0.7),
    (r"weak (?:jobs|payrolls|data|growth)|payrolls miss", -0.7),
    (r"unemployment (?:rises|rose|jumps|climbs)|jobless claims (?:rise|rose|jump|climb)", -0.6),
    (r"recession|slowdown|contraction|contracts?|stagflation", -0.7),
    (r"crisis|default|turmoil|banking stress", -0.5),
]
_MOVE_UP = re.compile(r"\b(surges?|rall(?:y|ies|ied)|jumps?|climbs?|gains?|rises?|rose|strengthens?|advances?|soars?|"
                      r"firms?|rebounds?)\b", re.I)
_MOVE_DOWN = re.compile(r"\b(plunges?|slumps?|tumbles?|falls?|fell|drops?|dropped|slips?|slides?|weakens?|"
                        r"sinks?|declines?|retreats?|sell-?off)\b", re.I)
_NEG = re.compile(r"\b(not|no|fails? to|unlikely to|less likely|without|denies|rules out)\b", re.I)

_GOLD_EXTRA = [
    (r"safe[- ]haven (?:demand|flows?|buying)|flight to safety", 0.7),
    (r"geopolitical|war|conflict|tensions?", 0.4),
    (r"strong(?:er)? dollar|dollar (?:strengthens|rallies|jumps|surges)|yields? (?:rise|jump|surge|climb)", -0.7),
    (r"weak(?:er)? dollar|dollar (?:weakens|slips|falls|drops)|yields? (?:fall|drop|slip|tumble)", 0.7),
]
_CRYPTO_EXTRA = [
    (r"etf inflows?|institutional (?:demand|adoption)|adoption", 0.6),
    (r"etf outflows?|ban|crackdown|hack(?:ed)?|exploit|lawsuit|sec sues?", -0.7),
    (r"risk[- ]on", 0.4), (r"risk[- ]off", -0.5),
]
_INVERT_POLICY = {"XAU"}
_MOVE_ENTITIES_ONLY_NEAR = 6


def _apply_patterns(text: str, patterns, sign: float = 1.0) -> float:
    total = 0.0
    for pat, w in patterns:
        for m in re.finditer(pat, text, re.I):
            before = text[max(0, m.start() - 24):m.start()]
            flip = -0.8 if _NEG.search(before) else 1.0
            total += sign * w * flip
    return total


def _move_score(text: str, entity: str) -> float:
    rx = _ENTITY_RE.get(entity)
    if rx is None:
        return 0.0
    total = 0.0
    words = text.split()
    for m in rx.finditer(text):
        n_before = len(text[:m.end()].split())
        window = " ".join(words[n_before:n_before + _MOVE_ENTITIES_ONLY_NEAR])
        total += 0.5 * len(_MOVE_UP.findall(window)) - 0.5 * len(_MOVE_DOWN.findall(window))
    return total


def lexicon_score(text: str, entity: str) -> float:
    """Sentimiento de ``text`` sobre el valor de ``entity``, en [-1, 1]."""
    t = text.lower()
    total = _apply_patterns(t, _POLICY_DATA, sign=-1.0 if entity in _INVERT_POLICY else 1.0)
    if entity == "XAU":
        total += _apply_patterns(t, _GOLD_EXTRA)
    elif entity in ("BTC", "ETH"):
        total += _apply_patterns(t, _CRYPTO_EXTRA)
    total += _move_score(t, entity)
    return math.tanh(total / 1.5)


SOURCE_WEIGHT = {"bloomberg": 1.0, "reuters": 1.0, "forexlive": 0.9, "sample": 0.5}


def aggregate(items: Iterable[tuple[NewsItem, float]], now: datetime, half_life_h: float = 6.0,
              shrink: float = 1.0) -> tuple[float, float]:
    """Agrega puntuaciones de titulares en ``(score [-100, 100], confianza [0, 1])``."""
    num = den = 0.0
    for it, s in items:
        w = 0.5 ** (it.age_hours(now) / half_life_h) * SOURCE_WEIGHT.get(it.source, 0.8)
        num += w * s
        den += w
    return 100.0 * num / (den + shrink), 1.0 - math.exp(-den / 3.0)
