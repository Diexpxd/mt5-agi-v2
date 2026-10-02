"""Calendario macroeconómico (JSON): ventanas de silencio, etiquetas de evento y sorpresa de datos."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
FAIRECONOMY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# (patrón en el título, etiqueta) - la primera coincidencia gana
EVENT_TAGS = [
    (r"non[- ]?farm|nfp|payrolls?", "NFP"),
    (r"\bfomc\b|federal funds|fed (?:chair|press)", "FOMC"),
    (r"\bcpi\b|consumer price", "CPI"),
    (r"\bppi\b|producer price", "PPI"),
    (r"interest rate|rate decision|refinancing rate|cash rate|bank rate|policy rate", "RATE_DECISION"),
    (r"\bgdp\b", "GDP"),
    (r"retail sales", "RETAIL_SALES"),
    (r"\bpmi\b|\bism\b", "PMI"),
    (r"jobless claims|unemployment claims", "JOBLESS_CLAIMS"),
    (r"unemployment rate|employment change", "EMPLOYMENT"),
    (r"speaks|speech|testif", "CB_SPEECH"),
]
_INVERSE_TITLES = ("unemployment rate", "jobless claims", "unemployment claims", "continuing claims")


@dataclass
class EconEvent:
    time: datetime                       # UTC
    currency: str
    impact: str                          # low | medium | high
    title: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    @property
    def tag(self) -> str:
        t = self.title.lower()
        for pat, tag in EVENT_TAGS:
            if re.search(pat, t):
                return tag
        return "OTHER"

    def to_json(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.astimezone(timezone.utc).isoformat()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "EconEvent":
        t = datetime.fromisoformat(str(d["time"]).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return cls(t.astimezone(timezone.utc), str(d["currency"]).upper(), str(d.get("impact", "low")).lower(),
                   d["title"], d.get("forecast"), d.get("previous"), d.get("actual"))


def symbol_currencies(symbol: str) -> List[str]:
    """Divisas que afectan a un símbolo: ``EURUSD -> [EUR, USD]``; ``XAUUSD -> [XAU, USD]`` (USD es la relevante)."""
    s = symbol.upper().replace("/", "")
    return [s[:3], s[3:6]]


def parse_number(s: Optional[str]) -> Optional[float]:
    """``'2.5%' -> 2.5``, ``'187K' -> 187000``, ``'1.2M' -> 1.2e6``, ``'-0.3B'`` -> -3e8; ``None`` si no es numérico."""
    if s is None:
        return None
    m = re.fullmatch(r"\s*([-+]?\d+(?:[.,]\d+)?)\s*([kKmMbBtT%]?)\s*", str(s))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * {"": 1, "%": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[m.group(2).lower()]


class EconomicCalendar:
    """Colección de eventos con consultas temporales."""

    def __init__(self, events: Iterable[EconEvent] = ()) -> None:
        self.events: List[EconEvent] = sorted(events, key=lambda e: e.time)

    @classmethod
    def from_json(cls, path: Path) -> "EconomicCalendar":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = raw["events"] if isinstance(raw, dict) else raw
            return cls(EconEvent.from_json(r) for r in rows)
        except Exception as exc:
            log.warning("Calendario inválido (%s): %s", path, exc)
            return cls()

    def to_json(self, path: Path, sample: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_sample": sample, "_note": "DATOS DE EJEMPLO: ejecuta scripts/update_calendar.py" if sample else "",
                   "events": [e.to_json() for e in self.events]}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    def merge(self, events: Iterable[EconEvent]) -> None:
        by_key = {(e.time, e.currency, e.title): e for e in self.events}
        for e in events:
            by_key[(e.time, e.currency, e.title)] = e
        self.events = sorted(by_key.values(), key=lambda e: e.time)

    def between(self, start: datetime, end: datetime, min_impact: str = "medium",
                currencies: Optional[Iterable[str]] = None) -> List[EconEvent]:
        cur = {c.upper() for c in currencies} if currencies else None
        rank = IMPACT_RANK[min_impact]
        return [e for e in self.events if start <= e.time <= end and IMPACT_RANK.get(e.impact, 0) >= rank
                and (cur is None or e.currency in cur)]

    def blackout(self, symbol: str, now: datetime, minutes_before: int = 30, minutes_after: int = 30,
                 min_impact: str = "high") -> Optional[EconEvent]:
        evs = self.between(now - timedelta(minutes=minutes_after), now + timedelta(minutes=minutes_before),
                           min_impact, symbol_currencies(symbol))
        return min(evs, key=lambda e: abs((e.time - now).total_seconds())) if evs else None

    def tags_at(self, now: datetime, symbol: str | None = None, window_minutes: int = 120) -> List[str]:
        cur = symbol_currencies(symbol) if symbol else None
        evs = self.between(now - timedelta(minutes=window_minutes), now + timedelta(minutes=window_minutes),
                           "medium", cur)
        return sorted({e.tag for e in evs if e.tag != "OTHER"})

    def upcoming(self, now: datetime, hours: float = 24, min_impact: str = "medium",
                 currencies: Optional[Iterable[str]] = None) -> List[EconEvent]:
        return self.between(now, now + timedelta(hours=hours), min_impact, currencies)

    def surprise(self, currency: str, now: datetime, lookback_hours: float = 12.0) -> float:
        num = den = 0.0
        for e in self.between(now - timedelta(hours=lookback_hours), now, "low", [currency]):
            a, f = parse_number(e.actual), parse_number(e.forecast)
            if a is None or f is None:
                continue
            z = max(-1.0, min(1.0, (a - f) / max(abs(f), 1e-9) * 5.0))     # 20 % de desvío = sorpresa máxima
            if any(k in e.title.lower() for k in _INVERSE_TITLES):
                z = -z
            w = {"high": 1.0, "medium": 0.5, "low": 0.2}[e.impact] * (1 - (now - e.time).total_seconds() / (lookback_hours * 3600))
            num += w * z
            den += w
        return num / den if den else 0.0


def refresh_from_web(url: str = FAIRECONOMY_URL, timeout: float = 10.0, fetcher=None) -> List[EconEvent]:
    """Descarga el calendario semanal (formato Forex Factory / faireconomy) y lo normaliza. Lanza si falla."""
    if fetcher is None:
        from core.net import http_get

        rows = http_get(url, timeout).json()
    else:
        rows = fetcher(url)
    out: List[EconEvent] = []
    for d in rows:
        impact = str(d.get("impact", "")).lower()
        if impact not in IMPACT_RANK:                # "Holiday" u otros
            continue
        t = datetime.fromisoformat(str(d["date"]).replace("Z", "+00:00"))
        out.append(EconEvent(t.astimezone(timezone.utc), str(d.get("country", "")).upper(), impact, d.get("title", ""),
                             d.get("forecast") or None, d.get("previous") or None, d.get("actual") or None))
    return out


def _first_weekday(year: int, month: int, weekday: int) -> datetime:
    d = datetime(year, month, 1, tzinfo=timezone.utc)
    return d + timedelta(days=(weekday - d.weekday()) % 7)


def generate_sample_calendar(start: datetime, end: datetime) -> List[EconEvent]:
    """Calendario SINTÉTICO de ejemplo (fechas aproximadas por reglas)."""
    ev: List[EconEvent] = []
    y, m = start.year, start.month
    while datetime(y, m, 1, tzinfo=timezone.utc) <= end:
        nfp = _first_weekday(y, m, 4).replace(hour=12, minute=30)
        ev.append(EconEvent(nfp, "USD", "high", "Non-Farm Employment Change", "180K", "175K"))
        ev.append(EconEvent(nfp, "USD", "high", "Unemployment Rate", "4.1%", "4.1%"))
        cpi = datetime(y, m, 12, 12, 30, tzinfo=timezone.utc)
        while cpi.weekday() > 4:
            cpi += timedelta(days=1)
        ev.append(EconEvent(cpi, "USD", "high", "CPI m/m", "0.3%", "0.2%"))
        ev.append(EconEvent(cpi + timedelta(days=3), "USD", "medium", "Retail Sales m/m", "0.4%", "0.3%"))
        second_thu = _first_weekday(y, m, 3) + timedelta(days=7)
        ev.append(EconEvent(second_thu.replace(hour=12, minute=15), "EUR", "high", "ECB Interest Rate Decision", "2.00%", "2.00%"))
        ev.append(EconEvent(second_thu.replace(hour=11), "GBP", "high", "BoE Interest Rate Decision", "4.00%", "4.00%"))
        third_wed = _first_weekday(y, m, 2) + timedelta(days=14)
        ev.append(EconEvent(third_wed.replace(hour=18), "USD", "high", "FOMC Statement", None, None))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    d = start.replace(hour=12, minute=30, second=0, microsecond=0)
    d += timedelta(days=(3 - d.weekday()) % 7)
    while d <= end:
        ev.append(EconEvent(d, "USD", "medium", "Unemployment Claims", "220K", "218K"))
        d += timedelta(days=7)
    return [e for e in ev if start <= e.time <= end]
