"""Brokers de ejecución con interfaz común: ``PaperBroker`` (simulado) y ``MT5Broker`` (cuenta DEMO de MT5)."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Protocol

from .exceptions import DataUnavailableError, OrderRejectedError
from .mt5_connection import MT5Connection
from .types import ApprovedOrder, Direction, OrderResult, PositionInfo

log = logging.getLogger(__name__)


@dataclass
class ClosedFill:
    """Posición cerrada, con resultado neto (incluye comisiones y swap)."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    price_open: float
    price_close: float
    profit: float
    open_time: datetime
    close_time: datetime
    reason: str = ""


class Broker(Protocol):
    name: str

    def account(self) -> SimpleNamespace: ...
    def positions(self) -> List[PositionInfo]: ...
    def open(self, order: ApprovedOrder) -> OrderResult: ...
    def close(self, ticket: int, reason: str = "") -> OrderResult: ...
    def update(self, now: datetime) -> None: ...
    def sync_closed(self, since: datetime) -> List[ClosedFill]: ...


class MT5Broker:
    """Envía órdenes a la cuenta DEMO conectada. Toda la seguridad vive en ``MT5Connection``."""

    name = "demo"

    def __init__(self, conn: MT5Connection) -> None:
        self.conn = conn
        self._reported: set[int] = set()

    def account(self) -> SimpleNamespace:
        a = self.conn.get_account()
        return SimpleNamespace(equity=float(a.equity), balance=float(a.balance), margin_free=float(a.margin_free),
                               margin_level=float(a.margin_level), leverage=float(a.leverage),
                               currency=getattr(a, "currency", "USD"))

    def positions(self) -> List[PositionInfo]:
        return self.conn.get_positions()

    def open(self, order: ApprovedOrder) -> OrderResult:
        return self.conn.send_order(order)

    def close(self, ticket: int, reason: str = "") -> OrderResult:
        return self.conn.close_position(ticket, reason or "AGI close")

    def update(self, now: datetime) -> None:      # el broker real ejecuta SL/TP por sí mismo
        return None

    def sync_closed(self, since: datetime) -> List[ClosedFill]:
        now = self.conn.server_time()
        deals = self.conn.get_deals(since - timedelta(days=1), now + timedelta(days=1))
        if deals.empty:
            return []
        deals = deals[deals["magic"] == self.conn.settings.magic_number]
        out: List[ClosedFill] = []
        for pid, g in deals.groupby("position_id"):
            pid = int(pid)
            ins, outs = g[g["entry"] == 0], g[g["entry"] != 0]
            if pid in self._reported or ins.empty or outs.empty:
                continue
            first, last = ins.iloc[0], outs.iloc[-1]
            pnl = float((g["profit"] + g["commission"].fillna(0) + g["swap"].fillna(0)).sum())
            direction = Direction.LONG if int(first["type"]) == 0 else Direction.SHORT
            reason = str(last.get("comment") or "")
            out.append(ClosedFill(pid, str(first["symbol"]), direction, float(first["volume"]), float(first["price"]),
                                  float(last["price"]), pnl, first["time"].to_pydatetime(), last["time"].to_pydatetime(),
                                  reason))
            self._reported.add(pid)
        return out


@dataclass
class PaperPosition:
    ticket: int
    symbol: str
    direction: int
    volume: float
    price_open: float
    sl: float
    tp: float
    open_time: str
    last_checked: str
    comment: str = ""


class PaperBroker:
    """Simulador de cuenta: fills a bid/ask reales, SL/TP evaluados con las velas M1 (SL primero en empates)."""

    name = "paper"

    def __init__(self, conn: MT5Connection, initial_balance: float = 10_000.0, commission_per_lot: float = 7.0,
                 leverage: float = 100.0, state_file: Path | None = None) -> None:
        self.conn = conn
        self.balance = float(initial_balance)
        self.commission_per_lot = commission_per_lot
        self.leverage = leverage
        self.state_file = state_file
        self._lock = threading.RLock()
        self._pos: Dict[int, PaperPosition] = {}
        self._closed_pending: List[ClosedFill] = []
        self._next = 900_000
        self._load()

    def _load(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        try:
            d = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.balance, self._next = d["balance"], d["next"]
            self._pos = {int(k): PaperPosition(**v) for k, v in d["positions"].items()}
            self._closed_pending = [ClosedFill(**{**c, "direction": Direction(c["direction"]),
                                                  "open_time": datetime.fromisoformat(c["open_time"]),
                                                  "close_time": datetime.fromisoformat(c["close_time"])})
                                    for c in d.get("closed_pending", [])]
        except Exception as exc:
            log.warning("paper_state.json inválido (%s): se empieza limpio", exc)

    def _save(self) -> None:
        if not self.state_file:
            return
        if not self.state_file.parent.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        pending = [{**asdict(c), "direction": int(c.direction), "open_time": c.open_time.isoformat(),
                    "close_time": c.close_time.isoformat()} for c in self._closed_pending]
        self.state_file.write_text(json.dumps({"balance": self.balance, "next": self._next, "closed_pending": pending,
                                               "positions": {k: asdict(v) for k, v in self._pos.items()}}),
                                   encoding="utf-8")

    def _view(self, p: PaperPosition) -> PositionInfo:
        spec = self.conn.get_symbol_spec(p.symbol)
        try:
            q = self.conn.get_quote(p.symbol)
            cur = q["bid"] if p.direction > 0 else q["ask"]
        except DataUnavailableError:
            cur = p.price_open
        return PositionInfo(
            ticket=p.ticket, symbol=p.symbol, direction=Direction(p.direction), volume=p.volume, price_open=p.price_open,
            sl=p.sl, tp=p.tp, price_current=cur, profit=spec.pnl(p.direction, p.price_open, cur, p.volume),
            open_time=datetime.fromisoformat(p.open_time), magic=self.conn.settings.magic_number, comment=p.comment)

    def positions(self) -> List[PositionInfo]:
        with self._lock:
            return [self._view(p) for p in self._pos.values()]

    def account(self) -> SimpleNamespace:
        with self._lock:
            pos = self.positions()
            floating = sum(p.profit for p in pos)
            margin = 0.0
            for p in pos:
                spec = self.conn.get_symbol_spec(p.symbol)
                margin += p.volume * spec.notional_per_lot(p.price_open) / self.leverage
            equity = self.balance + floating
            return SimpleNamespace(equity=equity, balance=self.balance, margin_free=equity - margin,
                                   margin_level=(equity / margin * 100.0) if margin > 0 else 0.0,
                                   leverage=self.leverage, currency="USD")

    def open(self, order: ApprovedOrder) -> OrderResult:
        self.conn._authority.verify(order)               # misma barrera que el broker real
        with self._lock:
            spec = self.conn.get_symbol_spec(order.symbol)
            if not (spec.volume_min - 1e-9 <= order.volume <= spec.volume_max + 1e-9):
                raise OrderRejectedError(f"Volumen inválido {order.volume} (min {spec.volume_min}, max {spec.volume_max})")
            q = self.conn.get_quote(order.symbol)
            buy = order.direction == Direction.LONG
            price = q["ask"] if buy else q["bid"]
            now = self.conn.server_time()
            self._next += 1
            comm = self.commission_per_lot * order.volume / 2
            self.balance -= comm
            self._pos[self._next] = PaperPosition(self._next, order.symbol, int(order.direction), order.volume, price,
                                                  order.sl, order.tp, now.isoformat(), now.isoformat(), order.comment)
            self._save()
            log.info("PAPER OPEN %s %s %.2f @ %.5f sl=%.5f tp=%.5f", order.symbol, "BUY" if buy else "SELL",
                     order.volume, price, order.sl, order.tp)
            return OrderResult(ok=True, ticket=self._next, price=price, volume=order.volume, retcode=10009,
                               comment="paper fill", simulated=True)

    def _close(self, p: PaperPosition, price: float, when: datetime, reason: str) -> ClosedFill:
        spec = self.conn.get_symbol_spec(p.symbol)
        gross = spec.pnl(p.direction, p.price_open, price, p.volume)
        comm = self.commission_per_lot * p.volume / 2
        self.balance += gross - comm
        entry_comm = self.commission_per_lot * p.volume / 2
        fill = ClosedFill(p.ticket, p.symbol, Direction(p.direction), p.volume, p.price_open, price,
                          gross - comm - entry_comm, datetime.fromisoformat(p.open_time), when, reason)
        self._closed_pending.append(fill)
        self._pos.pop(p.ticket, None)
        log.info("PAPER CLOSE %s %s @ %.5f (%s) pnl=%.2f", p.symbol, p.ticket, price, reason, fill.profit)
        return fill

    def close(self, ticket: int, reason: str = "manual") -> OrderResult:
        with self._lock:
            p = self._pos.get(ticket)
            if p is None:
                raise OrderRejectedError(f"No existe la posición {ticket}")
            q = self.conn.get_quote(p.symbol)
            price = q["bid"] if p.direction > 0 else q["ask"]
            self._close(p, price, self.conn.server_time(), reason)
            self._save()
            return OrderResult(ok=True, ticket=ticket, price=price, volume=p.volume, retcode=10009, simulated=True)

    def update(self, now: datetime) -> None:
        with self._lock:
            for p in list(self._pos.values()):
                last = datetime.fromisoformat(p.last_checked)
                try:
                    bars = self.conn.get_rates_range(p.symbol, "M1", last, now + timedelta(minutes=1))
                except DataUnavailableError:
                    continue
                bars = bars[bars.index > last]
                if bars.empty:
                    continue
                spec = self.conn.get_symbol_spec(p.symbol)
                spread = spec.spread_points * spec.point
                for ts, bar in bars.iterrows():
                    hi, lo = float(bar["high"]), float(bar["low"])
                    if p.direction > 0:
                        hit_sl, hit_tp = bool(p.sl and lo <= p.sl), bool(p.tp and hi >= p.tp)
                    else:                                       # las velas son BID; un corto cierra al ASK
                        hit_sl, hit_tp = bool(p.sl and hi + spread >= p.sl), bool(p.tp and lo + spread <= p.tp)
                    if hit_sl or hit_tp:
                        self._close(p, p.sl if hit_sl else p.tp, ts.to_pydatetime(), "sl" if hit_sl else "tp")
                        break
                else:
                    p.last_checked = bars.index[-1].isoformat()
            self._save()

    def sync_closed(self, since: datetime) -> List[ClosedFill]:
        with self._lock:
            out, self._closed_pending = self._closed_pending, []
            self._save()
            return out
