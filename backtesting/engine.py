"""Motor de backtesting propio: simula el pipeline COMPLETO (técnico + sentimiento opcional + memoria + riesgo)."""
from __future__ import annotations

import logging
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.economic_calendar import EconomicCalendar
from agents.orchestrator import OrchestratorConfig, compose_proposal
from agents.risk_agent import PortfolioTracker, RiskAgent, RiskContext
from agents.technical_agent import TechnicalAgent, TechnicalConfig
from config.settings import RiskLimits, Settings
from core.approval import ApprovalAuthority
from core.timeframes import bars_per_day, tf_minutes
from core.types import ClosedTrade, Direction, PortfolioState, PositionInfo, SymbolSpec, TradeProposal
from memory.trade_journal import TradeJournal

from .metrics import compute_metrics, drawdown_series, format_markdown, per_symbol

log = logging.getLogger(__name__)
SentimentFn = Callable[[str, datetime], Tuple[float, float]]


@dataclass
class BacktestConfig:
    timeframe: str = "M15"
    initial_balance: float = 10_000.0
    commission_per_lot: float = 7.0          # ida y vuelta, por lote
    slippage_points: float = 1.0             # adverso en entradas y salidas por stop
    leverage: float = 100.0
    window: int = 600                        # velas que ve el agente técnico
    warmup: int = 250
    eval_every: int = 1                      # analizar cada N velas
    train_frac: float = 0.0                  # >0: entrena el modelo con ese tramo inicial y evalúa el resto
    epochs: int = 8
    learn: bool = True                       # la memoria aprende durante el backtest
    cooldown_bars: int = 4
    max_bars: Optional[int] = None
    risk: RiskLimits = field(default_factory=RiskLimits)
    orch: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)


@dataclass
class _Pos:
    ticket: int
    symbol: str
    direction: int
    lots: float
    entry: float
    sl: float
    tp: float
    open_time: datetime
    risk_amount: float
    context: dict


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: List[ClosedTrade]
    metrics: Dict[str, float]
    by_symbol: Dict[str, Dict[str, float]]
    reject_reasons: Dict[str, int]
    counters: Dict[str, int]
    config: dict

    def markdown(self) -> str:
        c = self.counters
        head = (f"Señales evaluadas: {c.get('evaluated', 0)} · aprobadas: {c.get('approved', 0)} · rechazadas por riesgo: "
                f"{c.get('rejected', 0)} · vetos fundamentales: {c.get('vetoed', 0)}\n\n")
        rej = ""
        if self.reject_reasons:
            rej = "\n\nMotivos de rechazo: " + ", ".join(f"{k} ×{v}" for k, v in sorted(self.reject_reasons.items(), key=lambda kv: -kv[1])[:6])
        return head + format_markdown(self.metrics, self.by_symbol) + rej


class BacktestEngine:
    """Simula el sistema sobre datos históricos ``{símbolo: DataFrame OHLCV}`` (índice UTC, velas cerradas)."""

    def __init__(self, data: Dict[str, pd.DataFrame], specs: Dict[str, SymbolSpec], config: BacktestConfig | None = None,
                 calendar: EconomicCalendar | None = None, sentiment_fn: SentimentFn | None = None,
                 journal: TradeJournal | None = None, technical: TechnicalAgent | None = None) -> None:
        self.cfg = config or BacktestConfig()
        self.data = {s: d for s, d in data.items()}
        self.specs = specs
        self.calendar = calendar
        self.sentiment_fn = sentiment_fn
        self._tmp = Path(tempfile.mkdtemp(prefix="bt_"))
        self.journal = journal or TradeJournal(self._tmp / "journal.sqlite")
        self.settings = Settings(timeframe=self.cfg.timeframe, symbols=tuple(data), risk=self.cfg.risk, state_dir=self._tmp,
                                 data_dir=self._tmp, model_dir=self._tmp / "models", chroma_dir=self._tmp / "chroma",
                                 log_dir=self._tmp)
        self.technical = technical or TechnicalAgent(self.settings, self.cfg.technical)
        self.risk = RiskAgent(self.cfg.risk, authority=ApprovalAuthority(ttl_seconds=1e12), calendar=calendar,
                              clock=lambda: 0.0)

    def _returns(self) -> pd.DataFrame:
        closes = pd.concat({s: d["close"] for s, d in self.data.items()}, axis=1)
        return np.log(closes).diff()

    def _close_position(self, p: _Pos, price: float, when: datetime, reason: str, state: dict) -> None:
        spec = self.specs[p.symbol]
        comm = self.cfg.commission_per_lot * p.lots
        gross = spec.pnl(p.direction, p.entry, price, p.lots)
        state["balance"] += gross - comm / 2.0                     # la mitad de la comisión ya se cobró al entrar
        trade = ClosedTrade(ticket=p.ticket, symbol=p.symbol, direction=Direction(p.direction), volume=p.lots,
                            open_time=p.open_time, close_time=when, price_open=p.entry, price_close=price, sl=p.sl,
                            tp=p.tp, profit=gross - comm, risk_amount=p.risk_amount, exit_reason=reason, context=p.context)
        state["trades"].append(trade)
        state["positions"].pop(p.symbol, None)
        state["last_close"][p.symbol] = when
        if self.cfg.learn:
            self.journal.record(trade)

    def run(self, progress: Callable[[int, int], None] | None = None) -> BacktestResult:
        cfg = self.cfg
        syms = list(self.data)
        times = sorted(set().union(*[set(d.index) for d in self.data.values()]))
        if cfg.max_bars:
            times = times[: cfg.max_bars]
        arr = {s: {k: d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")} for s, d in self.data.items()}
        spread_arr = {s: (self.data[s]["spread"].to_numpy(dtype=float) if "spread" in self.data[s] else np.full(len(self.data[s]), self.specs[s].spread_points))
                      for s in syms}
        idx = {s: self.data[s].index.get_indexer(times) for s in syms}
        returns = self._returns()

        eval_start: Dict[str, int] = {s: cfg.warmup for s in syms}
        model_scores: Dict[str, pd.Series] = {}
        if cfg.train_frac > 0:
            for s in syms:
                n_train = int(len(self.data[s]) * cfg.train_frac)
                rep = self.technical.train(s, self.data[s].iloc[:n_train], cfg.timeframe, epochs=cfg.epochs, save=False)
                log.info("%s: modelo entrenado con %d barras (edge %+.3f, confiable=%s)", s, n_train, rep.edge, rep.trusted())
                model_scores[s] = self.technical.model_score_series(s, self.data[s])
                eval_start[s] = max(cfg.warmup, n_train)

        state = {"balance": cfg.initial_balance, "positions": {}, "trades": [], "last_close": {}}
        pending: Dict[str, Tuple[TradeProposal, float, float]] = {}
        tracker = PortfolioTracker()
        equity_pts: List[Tuple[pd.Timestamp, float]] = []
        exposed = 0
        counters = Counter()
        reasons: Counter = Counter()
        ticket = 0
        bpd = bars_per_day(cfg.timeframe)
        cooldown = pd.Timedelta(minutes=tf_minutes(cfg.timeframe) * cfg.cooldown_bars)
        last_close_px = {s: float("nan") for s in syms}

        for k, t in enumerate(times):
            now = t.to_pydatetime()
            for s in syms:
                p = idx[s][k]
                if p < 0:
                    continue
                o, h, l, c = (arr[s][x][p] for x in ("open", "high", "low", "close"))
                spec = self.specs[s]
                spread = (spread_arr[s][p] or spec.spread_points) * spec.point
                slip = cfg.slippage_points * spec.point
                if s in pending and s not in state["positions"]:
                    prop, lots, risk_amount = pending.pop(s)
                    d = int(prop.direction)
                    fill = o + spread + slip if d > 0 else o - slip
                    ticket += 1
                    state["balance"] -= cfg.commission_per_lot * lots / 2.0
                    state["positions"][s] = _Pos(ticket, s, d, lots, fill, prop.sl, prop.tp, now, risk_amount, prop.context)
                pos = state["positions"].get(s)
                if pos is not None:
                    if pos.direction > 0:
                        hit_sl, hit_tp = l <= pos.sl, h >= pos.tp
                        if hit_sl:
                            self._close_position(pos, min(o, pos.sl) - slip, now, "sl", state)
                        elif hit_tp:
                            self._close_position(pos, max(o, pos.tp), now, "tp", state)
                    else:                                              # cortos: se cierran al ASK
                        hit_sl, hit_tp = h + spread >= pos.sl, l + spread <= pos.tp
                        if hit_sl:
                            self._close_position(pos, max(o + spread, pos.sl) + slip, now, "sl", state)
                        elif hit_tp:
                            self._close_position(pos, min(o + spread, pos.tp), now, "tp", state)
                last_close_px[s] = c

            floating = 0.0
            pos_infos: List[PositionInfo] = []
            margin = 0.0
            for s, pos in state["positions"].items():
                spec = self.specs[s]
                px = last_close_px[s]
                sp = spec.spread_points * spec.point
                cur = px if pos.direction > 0 else px + sp
                pnl = spec.pnl(pos.direction, pos.entry, cur, pos.lots)
                floating += pnl
                margin += pos.lots * spec.notional_per_lot(pos.entry) / cfg.leverage
                pos_infos.append(PositionInfo(pos.ticket, s, Direction(pos.direction), pos.lots, pos.entry, pos.sl, pos.tp, cur, pnl, pos.open_time))
            equity = state["balance"] + floating
            equity_pts.append((t, equity))
            exposed += 1 if state["positions"] else 0
            tracker.update(equity, now)
            if progress and k % 500 == 0:
                progress(k, len(times))
            if equity <= 0:
                log.warning("Cuenta liquidada en %s", t)
                break

            pf = PortfolioState(equity=equity, balance=state["balance"], free_margin=equity - margin,
                                margin_level=(equity / margin * 100.0) if margin > 0 else 0.0, positions=pos_infos,
                                day_start_equity=tracker.day_start_equity, peak_equity=tracker.peak_equity, leverage=cfg.leverage)
            for s in syms:
                p = idx[s][k]
                if p < eval_start[s] or (p % max(cfg.eval_every, 1)) != 0 or s in state["positions"] or s in pending:
                    continue
                lc = state["last_close"].get(s)
                if lc is not None and (t - pd.Timestamp(lc)) < cooldown:
                    continue
                window = self.data[s].iloc[max(0, p - cfg.window + 1): p + 1]
                ms = model_scores[s].iloc[p] if s in model_scores and not np.isnan(model_scores[s].iloc[p]) else None
                spec = self.specs[s]
                sig = self.technical.analyze(s, window, spec, model_score=ms)
                if sig.direction == Direction.FLAT or sig.confidence < cfg.risk.min_confidence / (1.0 + cfg.orch.sentiment_weight):
                    continue
                counters["evaluated"] += 1
                sent_score, sent_conf = self.sentiment_fn(s, now) if self.sentiment_fn else (0.0, 0.0)
                d = int(sig.direction)
                sp = (spread_arr[s][p] or spec.spread_points) * spec.point
                entry_est = sig.entry + sp if d > 0 else sig.entry            # ASK para largos, BID para cortos
                tags = self.calendar.tags_at(now, s) if self.calendar else []
                prop, ctx, veto, advice = compose_proposal(cfg.orch, sig, sent_score, sent_conf, entry_est, spec, tags,
                                                           self.journal, now, symbol=s)
                if prop is None:
                    counters["vetoed"] += 1
                    continue
                rctx = RiskContext(specs=self.specs, returns=returns.loc[:t].tail(cfg.orch.returns_window), bars_per_day=bpd,
                                   quote={"bid": sig.entry, "ask": sig.entry + sp}, recent_r=self.journal.recent_r(10) if cfg.learn else [],
                                   now=now)
                dec = self.risk.evaluate(prop, spec, pf, rctx)
                if not dec.approved:
                    counters["rejected"] += 1
                    reasons[dec.reasons[0].split(":")[0] if dec.reasons else "?"] += 1
                    continue
                counters["approved"] += 1
                pending[s] = (prop, dec.lots, dec.risk_amount)
                pf.positions = pf.positions + [PositionInfo(-1, s, prop.direction, dec.lots, entry_est, prop.sl, prop.tp, entry_est, 0.0, now)]

        equity = pd.Series([e for _, e in equity_pts], index=pd.DatetimeIndex([t for t, _ in equity_pts]), name="equity")
        # posiciones abiertas al final: se cierran a mercado para contabilizar
        for s, pos in list(state["positions"].items()):
            spec = self.specs[s]
            last = last_close_px[s]
            self._close_position(pos, last if pos.direction > 0 else last + spec.spread_points * spec.point, times[-1].to_pydatetime(), "fin_backtest", state)
        trades = state["trades"]
        metrics = compute_metrics(equity, trades, exposure=exposed / max(len(equity_pts), 1))
        cfg_dict = {k: v for k, v in asdict(cfg).items()}
        return BacktestResult(equity, trades, metrics, per_symbol(trades), dict(reasons), dict(counters), cfg_dict)
