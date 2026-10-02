"""Orquestador central: coordina el enjambre de agentes."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from core.exceptions import MT5ConnectionError, RealAccountBlockedError
from core.execution import Broker, ClosedFill
from core.mt5_connection import MT5Connection
from core.timeframes import bars_per_day, tf_minutes
from core.types import ClosedTrade, Direction, PortfolioState, Signal, TradeProposal, utcnow
from memory.trade_journal import TradeJournal

from .economic_calendar import EconomicCalendar
from .fundamental_agent import FundamentalAgent, SentimentResult
from .risk_agent import PortfolioTracker, RiskAgent, RiskContext
from .technical_agent import TechnicalAgent

log = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    edge_scale: float = 0.08              # ventaja máxima sobre el equilibrio que puede aportar una señal perfecta
    prior_weight: float = 20.0            # pseudo-observaciones de la señal frente a la memoria
    sentiment_weight: float = 0.25        # cuánto modula la confianza el sentimiento alineado/contrario
    sentiment_veto: float = -0.40         # alineación <= esto (con confianza suficiente) => veto fundamental
    sentiment_min_conf: float = 0.30
    cooldown_bars: int = 4                # barras de espera tras cerrar un trade del mismo símbolo
    tech_bars: int = 600
    returns_window: int = 800
    news_refresh_seconds: int = 600
    async_news: bool = True               # refresco de feeds en hilo aparte (la red nunca bloquea el ciclo)


@dataclass
class CycleReport:
    time: datetime
    equity: float = 0.0
    analyzed: List[str] = field(default_factory=list)
    opened: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    closed: List[str] = field(default_factory=list)
    vetoed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def compose_proposal(cfg: "OrchestratorConfig", sig: Signal, sent_score: float, sent_conf: float, entry: float, spec,
                     event_tags: List[str], journal: TradeJournal, now: datetime, symbol: str | None = None):
    """Convierte una señal técnica + sentimiento + memoria en ``TradeProposal`` (compartida por vivo y backtest)."""
    sym = symbol or sig.symbol
    d = int(sig.direction)
    align = d * sent_score / 100.0
    if align <= cfg.sentiment_veto and sent_conf >= cfg.sentiment_min_conf:
        return None, {}, f"veto fundamental (sentimiento {sent_score:+.0f} contrario)", None
    conf = float(np.clip(sig.confidence * (1.0 + cfg.sentiment_weight * align * max(sent_conf, 0.3)), 0.0, 1.0))
    dist = abs(sig.entry - sig.sl)
    rr = abs(sig.tp - sig.entry) / dist if dist > 0 else 0.0
    sl, tp = round(entry - d * dist, spec.digits), round(entry + d * rr * dist, spec.digits)
    context = {"session": sig.meta.get("session"), "regime": sig.meta.get("regime"), "wyckoff": sig.meta.get("wyckoff"),
               "structure": sig.meta.get("structure"), "sentiment": round(sent_score, 1), "event_tags": event_tags,
               "tech_confidence": round(sig.confidence, 3), "confidence": round(conf, 3),
               "setup": [k for k, v in sig.meta.get("components", {}).items() if abs(v) > 0.15], "atr": sig.meta.get("atr")}
    advice = journal.advise(sym, sig.direction, context, now)
    p_be = 1.0 / (1.0 + rr) if rr > 0 else 0.5
    p = p_be + conf * cfg.edge_scale
    if advice.hist_win_rate is not None and advice.n_similar >= 8:
        p = (cfg.prior_weight * p + advice.n_similar * advice.hist_win_rate) / (cfg.prior_weight + advice.n_similar)
    p = float(np.clip(p, 0.05, 0.85))
    context["memory"] = {"weight": round(advice.risk_weight, 3), "hist_win_rate": advice.hist_win_rate, "n_similar": advice.n_similar}
    proposal = TradeProposal(sym, sig.direction, entry, sl, tp, p, conf, context, advice.risk_weight, now)
    return proposal, context, None, advice


class Orchestrator:
    """Coordina agentes, broker y memoria. Un ciclo (``run_cycle``) es idempotente por vela cerrada."""

    def __init__(self, settings: Settings, conn: MT5Connection, broker: Broker, technical: TechnicalAgent,
                 fundamental: FundamentalAgent, risk: RiskAgent, journal: TradeJournal,
                 calendar: EconomicCalendar | None = None, config: OrchestratorConfig | None = None) -> None:
        self.settings = settings
        self.conn = conn
        self.broker = broker
        self.technical = technical
        self.fundamental = fundamental
        self.risk = risk
        self.journal = journal
        self.calendar = calendar or fundamental.calendar
        self.cfg = config or OrchestratorConfig()
        self.tracker = PortfolioTracker(settings.state_dir / "risk_state.json")
        self.paused = False
        self.last_signals: Dict[str, Signal] = {}
        self.last_sentiment: Dict[str, SentimentResult] = {}
        self.decision_log: Deque[dict] = deque(maxlen=200)
        self.last_cycle: Optional[CycleReport] = None
        self._last_bar: Dict[str, pd.Timestamp] = {}
        self._last_close: Dict[str, datetime] = {}
        self._closes: Dict[str, pd.Series] = {}
        self._last_news = datetime.min.replace(tzinfo=timezone.utc)
        self._news_thread: Optional[threading.Thread] = None
        self._last_calendar = datetime.min.replace(tzinfo=timezone.utc)
        self._subs: List[Callable[[str, dict], None]] = []
        self._cycle_lock = threading.RLock()
        self._meta_file = settings.state_dir / "open_trades.json"
        self._open_meta: Dict[str, dict] = self._load_meta()
        self.equity_history: Deque[tuple] = deque(maxlen=20_000)
        self._equity_file = settings.state_dir / "equity.csv"
        self._load_equity()

    def now(self) -> datetime:
        return self.conn.server_time()

    def subscribe(self, cb: Callable[[str, dict], None]) -> None:
        self._subs.append(cb)

    def _emit(self, kind: str, **payload: Any) -> None:
        for cb in self._subs:
            try:
                cb(kind, payload)
            except Exception as exc:
                log.warning("suscriptor de eventos falló: %s", exc)

    def _load_meta(self) -> Dict[str, dict]:
        try:
            return json.loads(self._meta_file.read_text(encoding="utf-8")) if self._meta_file.exists() else {}
        except Exception:
            return {}

    def _load_equity(self) -> None:
        if not self._equity_file.exists():
            return
        try:
            for line in self._equity_file.read_text(encoding="utf-8").splitlines()[-5000:]:
                t, v = line.split(",")
                self.equity_history.append((datetime.fromisoformat(t), float(v)))
        except Exception as exc:
            log.warning("equity.csv ilegible (%s)", exc)

    def _record_equity(self, now: datetime, equity: float) -> None:
        self.equity_history.append((now, equity))
        try:
            with self._equity_file.open("a", encoding="utf-8") as f:
                f.write(f"{now.isoformat()},{equity:.2f}\n")
        except OSError:
            pass

    def _save_meta(self) -> None:
        self._meta_file.parent.mkdir(parents=True, exist_ok=True)
        self._meta_file.write_text(json.dumps(self._open_meta, default=str), encoding="utf-8")

    def bootstrap(self, train_missing: bool = False, train_bars: int = 12_000) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for sym in self.settings.symbols:
            if self.technical.load(sym, self.settings.timeframe):
                r = self.technical.models[sym].report
                out[sym] = f"cargado ({r.arch}, edge {r.edge:+.3f}, {'confiable' if r.trusted() else 'no confiable → ignorado'})"
            elif train_missing:
                try:
                    df = self.conn.get_rates(sym, self.settings.timeframe, train_bars)
                    r = self.technical.train(sym, df, self.settings.timeframe)
                    out[sym] = f"entrenado (edge {r.edge:+.3f}, {'confiable' if r.trusted() else 'no confiable → ignorado'})"
                except Exception as exc:
                    out[sym] = f"error entrenando: {exc}"
            else:
                out[sym] = "sin modelo (solo reglas SMC)"
        return out

    def pause(self) -> None:
        self.paused = True
        self._emit("paused")

    def resume(self) -> None:
        self.paused = False
        self._emit("resumed")

    def close_all(self, reason: str = "manual") -> int:
        n = 0
        for p in self.broker.positions():
            try:
                self.broker.close(p.ticket, reason)
                n += 1
            except Exception as exc:
                log.error("No se pudo cerrar %s: %s", p.ticket, exc)
        self._reconcile(self.now(), CycleReport(self.now()))
        return n

    def run_cycle(self) -> CycleReport:
        with self._cycle_lock:
            now = self.now()
            rep = CycleReport(now)
            self.conn.ensure_connected()
            self.broker.update(now)
            self._reconcile(now, rep)
            pf = self._snapshot(now)
            rep.equity = pf.equity
            self._record_equity(now, pf.equity)
            self._refresh_news(now)
            if self.paused:
                rep.notes.append("sistema en pausa")
            else:
                for sym in self.settings.symbols:
                    try:
                        pf = self._process_symbol(sym, pf, now, rep)
                    except (RealAccountBlockedError, MT5ConnectionError):
                        raise
                    except Exception as exc:
                        log.exception("Error procesando %s", sym)
                        rep.errors.append(f"{sym}: {exc!r}")
            self.last_cycle = rep
            return rep

    def _snapshot(self, now: datetime) -> PortfolioState:
        return self.tracker.snapshot(self.broker.account(), self.broker.positions(), now)

    def _refresh_news(self, now: datetime) -> None:
        if (now - self._last_news).total_seconds() < self.cfg.news_refresh_seconds:
            return
        if self._news_thread is not None and self._news_thread.is_alive():
            return
        self._last_news = now

        def work() -> None:
            try:
                n = self.fundamental.refresh(now)
                if n:
                    log.info("Noticias: %d titulares nuevos indexados", n)
            except Exception as exc:
                log.warning("refresh de noticias falló: %s", exc)
            self._refresh_calendar(now)

        if self.cfg.async_news:
            self._news_thread = threading.Thread(target=work, name="news-refresh", daemon=True)
            self._news_thread.start()
        else:
            work()

    def _refresh_calendar(self, now: datetime) -> None:
        if self.conn.is_mock or self.calendar is None:
            return
        if (now - self._last_calendar).total_seconds() < 6 * 3600:
            return
        self._last_calendar = now
        try:
            from .economic_calendar import refresh_from_web

            events = refresh_from_web()
            self.calendar.merge(events)
            self.calendar.to_json(self.settings.data_dir / "economic_calendar.json")
            log.info("Calendario macro actualizado: %d eventos esta semana", len(events))
        except Exception as exc:
            log.warning("No se pudo actualizar el calendario macro: %s", exc)

    def _returns_frame(self) -> Optional[pd.DataFrame]:
        if not self._closes:
            return None
        frame = pd.concat({s: np.log(c).diff() for s, c in self._closes.items()}, axis=1)
        return frame.dropna(how="all").tail(self.cfg.returns_window)

    def _process_symbol(self, sym: str, pf: PortfolioState, now: datetime, rep: CycleReport) -> PortfolioState:
        df = self.conn.get_rates(sym, self.settings.timeframe, self.cfg.tech_bars)
        bar_time = df.index[-1]
        self._closes[sym] = df["close"].astype(float)
        if self._last_bar.get(sym) == bar_time:
            return pf                                             # sin vela nueva: nada que decidir
        self._last_bar[sym] = bar_time
        held = [p for p in pf.positions if p.symbol == sym]
        if len(held) >= self.settings.risk.max_positions_per_symbol:
            return pf
        last_close = self._last_close.get(sym)
        if last_close and (now - last_close) < timedelta(minutes=tf_minutes(self.settings.timeframe) * self.cfg.cooldown_bars):
            rep.notes.append(f"{sym}: en enfriamiento tras cierre")
            return pf

        spec = self.conn.get_symbol_spec(sym)
        sig = self.technical.analyze(sym, df, spec)
        self.last_signals[sym] = sig
        rep.analyzed.append(sym)
        if sig.direction == Direction.FLAT:
            return pf

        if sig.confidence < self.settings.risk.min_confidence / (1.0 + self.cfg.sentiment_weight):
            rep.notes.append(f"{sym}: señal {sig.direction.label} débil ({sig.confidence:.2f}), descartada")
            return pf
        sent = self.fundamental.sentiment(sym, now)
        self.last_sentiment[sym] = sent
        d = int(sig.direction)
        q = self.conn.get_quote(sym)
        entry = q["ask"] if d > 0 else q["bid"]
        event_tags = self.calendar.tags_at(now, sym) if self.calendar else []
        proposal, context, veto, advice = compose_proposal(self.cfg, sig, sent.score, sent.confidence, entry, spec,
                                                           event_tags, self.journal, now)
        if proposal is None:
            rep.vetoed.append(sym)
            self._log_decision(sym, now, sig, sent, None, veto or "veto")
            return pf
        sl, tp, p = proposal.sl, proposal.tp, proposal.win_prob
        specs = {pos.symbol: self.conn.get_symbol_spec(pos.symbol) for pos in pf.positions}
        specs[sym] = spec
        rctx = RiskContext(specs=specs, returns=self._returns_frame(), bars_per_day=bars_per_day(self.settings.timeframe),
                           quote=q, recent_r=self.journal.recent_r(10), now=now)
        decision = self.risk.evaluate(proposal, spec, pf, rctx)
        self._log_decision(sym, now, sig, sent, decision, decision.summary())
        if not decision.approved:
            rep.rejected.append(f"{sym}: {'; '.join(decision.reasons)}")
            self._emit("risk_reject", symbol=sym, reasons=decision.reasons)
            return pf

        res = self.broker.open(decision.order)
        self._open_meta[str(res.ticket)] = {
            "context": context, "risk_amount": decision.risk_amount, "sl": sl, "tp": tp, "symbol": sym,
            "opened": now.isoformat(), "proposal_id": proposal.proposal_id, "lessons": advice.lessons}
        self._save_meta()
        rep.opened.append(f"{sym} {sig.direction.label} {decision.lots:.2f} @ {res.price:.5f}")
        self._emit("trade_opened", symbol=sym, direction=sig.direction.label, lots=decision.lots, price=res.price,
                   sl=sl, tp=tp, risk_amount=decision.risk_amount, risk_pct=decision.risk_pct, ticket=res.ticket,
                   rationale=sig.rationale, sentiment=sent.score, lessons=advice.lessons, win_prob=p)
        return self._snapshot(now)

    def _log_decision(self, sym: str, now: datetime, sig: Signal, sent: SentimentResult, decision, summary: str) -> None:
        entry = {"time": now.isoformat(), "symbol": sym, "direction": sig.direction.label, "tech_conf": round(sig.confidence, 3),
                 "sentiment": round(sent.score, 1), "summary": summary, "rationale": sig.rationale,
                 "checks": [(c.name, c.passed, c.detail) for c in decision.checks] if decision else []}
        self.decision_log.append(entry)
        log.info("[%s] %s %s → %s", now.strftime("%H:%M"), sym, sig.direction.label, summary)

    def _reconcile(self, now: datetime, rep: CycleReport) -> None:
        fills: List[ClosedFill] = self.broker.sync_closed(now - timedelta(days=3))
        for f in fills:
            meta = self._open_meta.pop(str(f.ticket), {})
            trade = ClosedTrade(
                ticket=f.ticket, symbol=f.symbol, direction=f.direction, volume=f.volume, open_time=f.open_time,
                close_time=f.close_time, price_open=f.price_open, price_close=f.price_close, sl=meta.get("sl", 0.0),
                tp=meta.get("tp", 0.0), profit=f.profit, risk_amount=float(meta.get("risk_amount", 0.0)),
                exit_reason=f.reason, context=meta.get("context", {}))
            self.journal.record(trade)
            self._last_close[f.symbol] = f.close_time
            rep.closed.append(f"{f.symbol} {f.profit:+.2f} ({trade.r_multiple:+.2f}R)")
            self._emit("trade_closed", symbol=f.symbol, profit=f.profit, r=trade.r_multiple, reason=f.reason, ticket=f.ticket)
        if fills:
            self._save_meta()

    def status(self) -> Dict[str, Any]:
        acct = self.broker.account()
        pos = self.broker.positions()
        pf = self.tracker
        return {
            "mode": self.broker.name, "paused": self.paused, "connected": self.conn.is_alive(),
            "backend": "MOCK" if self.conn.is_mock else "MT5", "equity": acct.equity, "balance": acct.balance,
            "daily_pnl": acct.equity - pf.day_start_equity if pf.day_start_equity else 0.0,
            "drawdown": (pf.peak_equity - acct.equity) / pf.peak_equity if pf.peak_equity else 0.0,
            "positions": pos, "latency": self.conn.latency.summary(),
            "last_cycle": self.last_cycle.time.isoformat() if self.last_cycle else None,
            "utc_offset_h": self.conn.utc_offset.total_seconds() / 3600, "reconnects": self.conn.reconnects,
        }

    def run_forever(self, stop: threading.Event) -> None:
        backoff = 2.0
        while not stop.is_set():
            try:
                self.run_cycle()
                backoff = 2.0
            except RealAccountBlockedError:
                log.critical("CUENTA REAL DETECTADA: sistema detenido por seguridad")
                self.paused = True
                self._emit("halt", reason="cuenta real detectada")
                stop.set()
                raise
            except MT5ConnectionError as exc:
                log.error("MT5 no disponible: %s (reintento en %.0fs)", exc, backoff)
                self._emit("connection_lost", error=str(exc))
                stop.wait(backoff)
                backoff = min(backoff * 2, 300.0)
                continue
            except Exception:
                log.exception("Error inesperado en el ciclo")
            stop.wait(self.settings.cycle_seconds)
