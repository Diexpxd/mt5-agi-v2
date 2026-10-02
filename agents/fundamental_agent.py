"""Agente de Análisis Fundamental (RAG de sentimiento)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import Settings
from core.llm import LLMClient, MockLLM
from core.types import Direction, Signal, utcnow
from memory.vector_store import Hit, open_store

from .economic_calendar import EconomicCalendar
from .news import (FeedIngestor, NewsItem, aggregate, lexicon_score, load_feeds, load_sample_news, symbol_entities)

log = logging.getLogger(__name__)

FX_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"}
ENTITY_QUERIES = {
    "USD": "US dollar Federal Reserve Fed FOMC rates inflation payrolls Treasury yields",
    "EUR": "euro ECB Lagarde eurozone inflation rates German economy",
    "GBP": "pound sterling Bank of England BoE inflation UK economy gilts",
    "JPY": "yen Bank of Japan BoJ Ueda intervention rates Japan economy",
    "CAD": "Canadian dollar loonie Bank of Canada oil Canada economy",
    "AUD": "Australian dollar aussie RBA China commodities Australia economy",
    "NZD": "New Zealand dollar kiwi RBNZ dairy",
    "CHF": "Swiss franc SNB safe haven Switzerland",
    "XAU": "gold bullion safe haven real yields dollar central bank buying",
    "BTC": "bitcoin crypto ETF flows regulation adoption",
    "ETH": "ethereum ether staking ETF crypto",
    "OIL": "crude oil Brent WTI OPEC inventories demand",
}


@dataclass
class SentimentResult:
    symbol: str
    score: float                              # [-100, 100]; >0 favorece comprar el símbolo
    confidence: float                         # [0, 1]
    drivers: List[str] = field(default_factory=list)
    per_entity: Dict[str, float] = field(default_factory=dict)
    n_items: int = 0
    method: str = "lexicon"
    generated_at: datetime = field(default_factory=utcnow)

    def label(self) -> str:
        s = self.score
        return ("muy alcista" if s >= 50 else "alcista" if s >= 20 else "muy bajista" if s <= -50
                else "bajista" if s <= -20 else "neutral")


class FundamentalAgent:
    """Ingiere noticias, las indexa (RAG) y puntúa sentimiento por símbolo."""

    name = "fundamental"

    def __init__(self, settings: Settings, llm: LLMClient | None = None, store=None,
                 calendar: EconomicCalendar | None = None, ingestor: FeedIngestor | None = None,
                 offline_samples: Optional[bool] = None, min_abs_score: float = 25.0) -> None:
        self.settings = settings
        self.llm = llm or MockLLM()
        self.store = store if store is not None else open_store("news", settings.chroma_dir)
        self.calendar = calendar or EconomicCalendar.from_json(settings.data_dir / "economic_calendar.json")
        self.ingestor = ingestor or FeedIngestor(load_feeds(settings.data_dir / "feeds.json"))
        self.offline_samples = settings.use_mock_mt5 if offline_samples is None else offline_samples
        self.min_abs_score = min_abs_score
        self._llm_cache: Dict[tuple, Tuple[float, Tuple[float, float, str]]] = {}
        self.cache_ttl = 600.0

    def ingest(self, items: List[NewsItem]) -> int:
        new = [i for i in items if not self.store.has(i.id)]
        if new:
            self.store.add(
                [i.id for i in new], [i.text for i in new],
                [{"title": i.title, "source": i.source, "ts": i.published.timestamp(), "ents": "|" + "|".join(i.entities) + "|",
                  "link": i.link} for i in new])
        return len(new)

    def refresh(self, now: datetime | None = None) -> int:
        now = now or utcnow()
        items = [] if self.offline_samples else self.ingestor.fetch_all(now)
        if not items and self.offline_samples:
            items = load_sample_news(self.settings.data_dir / "sample_news.json", now)
            log.debug("Modo offline: %d titulares de muestra", len(items))
        return self.ingest(items)

    @staticmethod
    def _to_item(h: Hit) -> NewsItem:
        m = h.metadata
        title = m.get("title", h.text)
        summary = h.text[len(title) + 2:] if h.text.startswith(title) else ""
        ents = [e for e in str(m.get("ents", "")).split("|") if e]
        return NewsItem(title, summary, m.get("source", ""), datetime.fromtimestamp(float(m.get("ts", 0)), tz=timezone.utc),
                        m.get("link", ""), ents)

    def retrieve(self, entity: str, now: datetime, k: int = 12, max_age_h: float = 36.0) -> List[NewsItem]:
        cutoff = (now - timedelta(hours=max_age_h)).timestamp()
        hits = self.store.query(ENTITY_QUERIES.get(entity, entity), k=max(k * 6, 60), where={"ts": {"$gte": cutoff}})
        scored: List[Tuple[float, NewsItem]] = []
        for h in hits:
            item = self._to_item(h)
            if entity not in item.entities or item.published > now + timedelta(hours=1):
                continue
            scored.append((0.6 * h.score + 0.4 * 0.5 ** (item.age_hours(now) / 6.0), item))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:k]]

    def _llm_entity(self, entity: str, items: List[NewsItem], now: datetime) -> Optional[Tuple[float, float, str]]:
        key = (entity, tuple(i.id for i in items))
        cached = self._llm_cache.get(key)
        if cached and time.time() - cached[0] < self.cache_ttl:
            return cached[1]
        lines = "\n".join(f"{n}. [hace {i.age_hours(now):.1f} h, {i.source}] {i.title}" for n, i in enumerate(items, 1))
        prompt = (f"Titulares recientes sobre {entity}:\n{lines}\n\n"
                  f"Evalúa el impacto NETO esperado sobre el VALOR de {entity} en las próximas horas "
                  f"(positivo = {entity} se aprecia). Pondera más lo reciente y lo de alto impacto. "
                  # literal SIN formatear: contiene llaves JSON
                  'Responde SOLO con JSON: {"score": <entero -100..100>, "confidence": <0..1>, "reasoning": "<máx 160 caracteres>"}')
        res = self.llm.generate_json(prompt, system="Eres un analista macro cuantitativo. Sé conciso y no inventes datos.")
        try:
            out = (max(-100.0, min(100.0, float(res["score"]))), max(0.0, min(1.0, float(res.get("confidence", 0.5)))),
                   str(res.get("reasoning", ""))[:200])
        except Exception:
            return None
        self._llm_cache[key] = (time.time(), out)
        return out

    def entity_score(self, entity: str, now: datetime) -> Tuple[float, float, List[str], str, int]:
        items = self.retrieve(entity, now)
        method = "lexicon"
        drivers: List[str] = []
        scored = [(i, lexicon_score(i.text, entity)) for i in items]
        score, conf = aggregate(scored, now)
        if not self.llm.is_mock and items:
            res = self._llm_entity(entity, items, now)
            if res is not None:
                score, conf, why = res
                method = "llm"
                drivers.append(f"LLM: {why}")
        for it, s in sorted(scored, key=lambda x: -abs(x[1]))[:3]:
            if abs(s) >= 0.15:
                drivers.append(f"{'+' if s > 0 else '-'}{abs(s):.2f} [{it.source}] {it.title[:110]}")
        if entity in FX_CURRENCIES:
            surprise = self.calendar.surprise(entity, now)
            if surprise:
                score = max(-100.0, min(100.0, score + 25.0 * surprise))
                drivers.append(f"sorpresa macro {entity}: {surprise:+.2f}")
                conf = max(conf, 0.5)
        return score, conf, drivers, method, len(items)

    def sentiment(self, symbol: str, now: datetime | None = None) -> SentimentResult:
        now = now or utcnow()
        base, quote = symbol_entities(symbol)
        sb, cb, db, mb, nb = self.entity_score(base, now)
        sq, cq, dq, mq, nq = self.entity_score(quote, now)
        if base in FX_CURRENCIES and quote in FX_CURRENCIES:
            score = sb - sq
        else:
            score = 0.7 * sb - 0.3 * sq
        score = max(-100.0, min(100.0, score))
        drivers = [f"{base}: {d}" for d in db] + [f"{quote}: {d}" for d in dq]
        return SentimentResult(
            symbol=symbol, score=score, confidence=max(cb, cq) if base in FX_CURRENCIES else cb,
            drivers=drivers[:6], per_entity={base: round(sb, 1), quote: round(sq, 1)}, n_items=nb + nq,
            method="llm" if "llm" in (mb, mq) else "lexicon", generated_at=now)

    def analyze(self, symbol: str, now: datetime | None = None) -> Signal:
        r = self.sentiment(symbol, now)
        if abs(r.score) < self.min_abs_score or r.confidence < 0.2:
            direction = Direction.FLAT
        else:
            direction = Direction.LONG if r.score > 0 else Direction.SHORT
        return Signal(symbol, direction, min(1.0, abs(r.score) / 100.0 * (0.5 + 0.5 * r.confidence)), source=self.name,
                      rationale=r.drivers, meta={"score": r.score, "confidence": r.confidence, "method": r.method,
                                                 "per_entity": r.per_entity, "n_items": r.n_items})
