"""Diario de operaciones cerradas + aprendizaje (RAG sobre el propio historial)."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.types import ClosedTrade, Direction

log = logging.getLogger(__name__)

PRIOR_N = 8            # pseudo-operaciones neutras para la contracción bayesiana
MIN_BUCKET_N = 5       # muestra mínima para que un cubo influya
W_MIN, W_MAX = 0.25, 1.5


def _default(o: Any) -> Any:
    try:
        return float(o)
    except Exception:
        return str(o)


def bucket_weight(rs: Sequence[float], prior_n: int = PRIOR_N) -> Tuple[float, Dict[str, float]]:
    """Peso de riesgo a partir de los resultados R de un cubo."""
    n = len(rs)
    if n < MIN_BUCKET_N:
        return 1.0, {"n": n, "mean_r": float(sum(rs) / n) if n else 0.0, "significant": 0.0}
    m = sum(rs) / n
    s = math.sqrt(sum((r - m) ** 2 for r in rs) / max(n - 1, 1))
    e = n / (n + prior_n) * m
    w = min(max(1.0 + e, W_MIN), W_MAX)
    negative = (m + s / math.sqrt(n)) < 0
    if negative:
        w = min(w, 0.4)
    return w, {"n": n, "mean_r": m, "std_r": s, "shrunk_e": e, "significant": 1.0 if negative else 0.0}


def stats_of(rs: Sequence[float], pnls: Sequence[float] | None = None) -> Dict[str, float]:
    """``n, wins, win_rate, avg_r, profit_factor`` (PF = ganancias brutas / pérdidas brutas, en R)."""
    n = len(rs)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0, "total_pnl": 0.0}
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r < 0)
    wins = sum(1 for r in rs if r > 0)
    return {"n": n, "wins": wins, "win_rate": wins / n, "avg_r": sum(rs) / n,
            "profit_factor": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "total_pnl": float(sum(pnls)) if pnls is not None else 0.0}


@dataclass
class MemoryAdvice:
    """Consejo de la memoria para una operación candidata."""

    risk_weight: float = 1.0
    hist_win_rate: Optional[float] = None
    n_similar: int = 0
    lessons: List[str] = field(default_factory=list)
    buckets: Dict[str, float] = field(default_factory=dict)


def context_tags(symbol: str, direction: Direction | int, ctx: Dict[str, Any], when: datetime) -> List[str]:
    """Etiquetas de cubo de una operación: sesión, eventos, régimen, día, dirección."""
    tags = ["ALL", f"dir:{Direction(int(direction)).label}", f"dow:{when.strftime('%a')}"]
    if ctx.get("session"):
        tags.append(f"session:{ctx['session']}")
    for ev in ctx.get("event_tags", []) or []:
        tags.append(f"event:{ev}")
    if ctx.get("regime"):
        tags.append(f"regime:{ctx['regime']}")
    if ctx.get("wyckoff") and ctx["wyckoff"] != "unclear":
        tags.append(f"wyckoff:{ctx['wyckoff']}")
    return tags


def describe_context(symbol: str, direction: Direction | int, ctx: Dict[str, Any], when: datetime) -> str:
    """Texto descriptivo (SIN resultado) que se indexa y se consulta en el RAG de operaciones."""
    parts = [symbol, Direction(int(direction)).label, *context_tags(symbol, direction, ctx, when)]
    s = ctx.get("sentiment")
    if s is not None:
        parts.append("sentiment:" + ("bullish" if s >= 20 else "bearish" if s <= -20 else "neutral"))
    st = ctx.get("structure")
    if st:
        parts.append("structure:" + ("bullish" if st > 0 else "bearish"))
    if ctx.get("setup"):
        parts.extend(f"setup:{x}" for x in ctx["setup"])
    return " ".join(parts)


class TradeJournal:
    """SQLite + almacén vectorial. Seguro entre hilos."""

    def __init__(self, db_path: Path, vector_store=None) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.store = vector_store
        with self._lock:
            self._db.execute("""CREATE TABLE IF NOT EXISTS trades (
                ticket INTEGER PRIMARY KEY, symbol TEXT, direction INTEGER, volume REAL, open_time TEXT, close_time TEXT,
                price_open REAL, price_close REAL, sl REAL, tp REAL, profit REAL, risk_amount REAL, exit_reason TEXT,
                context TEXT)""")
            self._db.commit()

    def record(self, t: ClosedTrade) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.ticket, t.symbol, int(t.direction), t.volume, t.open_time.isoformat(), t.close_time.isoformat(),
                 t.price_open, t.price_close, t.sl, t.tp, t.profit, t.risk_amount, t.exit_reason,
                 json.dumps(t.context, default=_default)))
            self._db.commit()
        if self.store is not None:
            try:
                self.store.add([str(t.ticket)], [describe_context(t.symbol, t.direction, t.context, t.open_time)],
                               [{"symbol": t.symbol, "direction": int(t.direction), "r": float(t.r_multiple),
                                 "win": 1 if t.profit > 0 else 0, "profit": float(t.profit),
                                 "ts": t.close_time.timestamp(),
                                 "summary": f"{t.symbol} {t.direction.label} {t.r_multiple:+.2f}R ({t.exit_reason})"}])
            except Exception as exc:
                log.warning("No se pudo indexar el trade %s en la BD vectorial: %s", t.ticket, exc)

    def trades(self, symbol: str | None = None, limit: int | None = None) -> List[ClosedTrade]:
        q = "SELECT * FROM trades" + (" WHERE symbol=?" if symbol else "") + " ORDER BY close_time"
        with self._lock:
            rows = self._db.execute(q, (symbol,) if symbol else ()).fetchall()
        out = [ClosedTrade(
            ticket=r[0], symbol=r[1], direction=Direction(r[2]), volume=r[3], open_time=datetime.fromisoformat(r[4]),
            close_time=datetime.fromisoformat(r[5]), price_open=r[6], price_close=r[7], sl=r[8], tp=r[9], profit=r[10],
            risk_amount=r[11], exit_reason=r[12] or "", context=json.loads(r[13] or "{}")) for r in rows]
        return out[-limit:] if limit else out

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    def has(self, ticket: int) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM trades WHERE ticket=?", (ticket,)).fetchone() is not None

    def recent_r(self, n: int = 10) -> List[float]:
        return [t.r_multiple for t in self.trades(limit=n)]

    def stats(self, symbol: str | None = None, tag: str | None = None) -> Dict[str, float]:
        ts = self._select(symbol, tag)
        return stats_of([t.r_multiple for t in ts], [t.profit for t in ts])

    def _select(self, symbol: str | None, tag: str | None) -> List[ClosedTrade]:
        out = []
        for t in self.trades(symbol):
            if tag in (None, "ALL") or tag in context_tags(t.symbol, t.direction, t.context, t.open_time):
                out.append(t)
        return out

    def _bucket_keys(self, symbol: str, direction: Direction | int, ctx: Dict[str, Any], when: datetime) -> List[Tuple[str, str]]:
        keys: List[Tuple[str, str]] = []
        for tag in context_tags(symbol, direction, ctx, when):
            keys.append((symbol, tag))
            if tag.startswith("event:") or tag == "ALL":
                keys.append(("*", tag))                # los eventos macro afectan a toda la cartera
        return keys

    def bucket_table(self) -> Dict[Tuple[str, str], List[float]]:
        table: Dict[Tuple[str, str], List[float]] = {}
        for t in self.trades():
            for key in self._bucket_keys(t.symbol, t.direction, t.context, t.open_time):
                table.setdefault(key, []).append(t.r_multiple)
        return table

    def advise(self, symbol: str, direction: Direction | int, ctx: Dict[str, Any], when: datetime,
               k: int = 12) -> MemoryAdvice:
        table = self.bucket_table()
        weights: Dict[str, float] = {}
        lessons: List[str] = []
        for key in self._bucket_keys(symbol, direction, ctx, when):
            rs = table.get(key, [])
            w, info = bucket_weight(rs)
            if info["n"] < MIN_BUCKET_N:
                continue
            label = f"{key[0]}/{key[1]}"
            weights[label] = w
            if w <= 0.75 or w >= 1.2:
                lessons.append(self._lesson(key, rs, w))
        if weights:
            lo, hi = min(weights.values()), max(weights.values())
            weight = lo if lo < 1.0 else min(hi, 1.25)      # lo negativo domina; lo positivo solo suma con cautela
        else:
            weight = 1.0
        hist, n_sim = None, 0
        if self.store is not None and self.store.count() >= 8:
            hits = self.store.query(describe_context(symbol, direction, ctx, when), k=k, where={"symbol": symbol})
            if len(hits) >= 8:
                n_sim = len(hits)
                hist = sum(float(h.metadata.get("win", 0)) for h in hits) / n_sim
        return MemoryAdvice(risk_weight=weight, hist_win_rate=hist, n_similar=n_sim, lessons=lessons, buckets=weights)

    @staticmethod
    def _lesson(key: Tuple[str, str], rs: Sequence[float], w: float) -> str:
        scope = "cualquier símbolo" if key[0] == "*" else key[0]
        tag = key[1].replace("event:", "durante ").replace("session:", "en la sesión ").replace("regime:", "en régimen ")
        tag = tag.replace("dir:", "en operaciones ").replace("dow:", "los ").replace("wyckoff:", "con Wyckoff ")
        tag = "en general" if tag == "ALL" else tag
        wins = sum(1 for r in rs if r > 0)
        m = sum(rs) / len(rs)
        verb = "Pierdo" if w < 1 else "Gano"
        return (f"{verb} operando {scope} {tag}: {wins}/{len(rs)} ganadas, R medio {m:+.2f} "
                f"→ {'reduzco' if w < 1 else 'aumento'} el peso de riesgo a {w:.2f}.")

    def lessons(self, limit: int = 10) -> List[str]:
        out: List[Tuple[float, str]] = []
        for key, rs in self.bucket_table().items():
            w, info = bucket_weight(rs)
            if info["n"] >= MIN_BUCKET_N and (w <= 0.75 or w >= 1.2) and key[1] != "ALL":
                out.append((abs(w - 1.0), self._lesson(key, rs, w)))
        return [s for _, s in sorted(out, key=lambda x: -x[0])[:limit]]
