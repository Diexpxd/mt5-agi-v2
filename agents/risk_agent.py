"""Agente de Gestión de Riesgo (The Bouncer)."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import RiskLimits
from core.approval import DEFAULT_AUTHORITY, ApprovalAuthority
from core.types import (ApprovedOrder, Direction, PortfolioState, PositionInfo, RiskCheck, RiskDecision,
                        SymbolSpec, TradeProposal, utcnow)

from .economic_calendar import EconomicCalendar

log = logging.getLogger(__name__)
MIN_RISK_PCT = 0.0005          # por debajo de 0.05 % del equity no merece la pena operar


def kelly_fraction(p: float, b: float) -> float:
    """Criterio de Kelly para una apuesta binaria."""
    if b <= 0 or not (0.0 < p < 1.0):
        return 0.0
    return p - (1.0 - p) / b


def dynamic_kelly_scale(base_fraction: float, drawdown: float, max_drawdown: float, consecutive_losses: int = 0,
                        memory_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Multiplicador dinámico sobre ``f*``."""
    dd_scale = max(0.0, 1.0 - drawdown / max_drawdown) if max_drawdown > 0 else 1.0
    streak = max(0.4, 1.0 - 0.15 * min(consecutive_losses, 4))
    mem = min(max(memory_weight, 0.25), 1.5)
    return base_fraction * dd_scale * streak * mem, {"dd_scale": dd_scale, "streak_scale": streak, "memory": mem}


def round_down_to_step(x: float, step: float) -> float:
    """Redondea ``x`` hacia abajo al múltiplo de ``step`` (nunca sobre-arriesga)."""
    if step <= 0:
        return x
    return round(math.floor(x / step + 1e-9) * step, 8)


def portfolio_var(exposures: Dict[str, float], returns: pd.DataFrame, confidence: float, bars_per_day: float,
                  window: int = 750) -> Optional[float]:
    """VaR a 1 día (importe positivo, moneda de la cuenta) = máx(paramétrico, histórico)."""
    cols = [s for s in exposures if s in returns.columns]
    if not cols:
        return None
    R = returns[cols].dropna().tail(window)
    if len(R) < 60:
        return None
    w = np.array([exposures[s] for s in cols], dtype=float)
    cov = R.cov().to_numpy()
    cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    z = NormalDist().inv_cdf(confidence)
    var_param = z * math.sqrt(max(float(w @ cov @ w) * bars_per_day, 0.0))
    rp = R.to_numpy() @ w
    var_hist = -float(np.quantile(rp, 1.0 - confidence)) * math.sqrt(bars_per_day)
    return max(var_param, var_hist, 0.0)


def size_from_risk(equity: float, risk_pct: float, entry: float, sl: float, spec: SymbolSpec) -> float:
    """Lotes (sin redondear) que arriesgan ``risk_pct * equity`` si se toca el SL."""
    per_lot = spec.risk_per_lot(entry, sl)
    return equity * risk_pct / per_lot if per_lot > 0 else 0.0


def pair_legs(symbol: str) -> Tuple[str, str]:
    s = symbol.upper().replace("/", "")
    return s[:3], s[3:6]


def signed_notional(direction: int, lots: float, price: float, spec: SymbolSpec) -> float:
    """Nocional con signo (moneda de la cuenta): ``dirección * lotes * tamaño_contrato * precio`` (USD-quoted)."""
    return direction * lots * spec.notional_per_lot(price)


class PortfolioTracker:
    """Mantiene equity de inicio de día y pico histórico (persistente entre reinicios)."""

    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = state_file
        self.day: Optional[str] = None
        self.day_start_equity = 0.0
        self.peak_equity = 0.0
        if state_file and state_file.exists():
            try:
                d = json.loads(state_file.read_text(encoding="utf-8"))
                self.day, self.day_start_equity, self.peak_equity = d["day"], d["day_start_equity"], d["peak_equity"]
            except Exception as exc:
                log.warning("risk_state.json inválido: %s", exc)

    def update(self, equity: float, now: datetime) -> None:
        today = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if self.day != today or self.day_start_equity <= 0:
            self.day, self.day_start_equity = today, equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.state_file:
            if not self.state_file.parent.exists():
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"day": self.day, "day_start_equity": self.day_start_equity,
                                                   "peak_equity": self.peak_equity}), encoding="utf-8")

    def snapshot(self, account, positions: List[PositionInfo], now: datetime) -> PortfolioState:
        self.update(float(account.equity), now)
        return PortfolioState(
            equity=float(account.equity), balance=float(account.balance), free_margin=float(account.margin_free),
            margin_level=float(account.margin_level), positions=positions, day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity, leverage=float(getattr(account, "leverage", 100) or 100),
            currency=getattr(account, "currency", "USD"))


@dataclass
class RiskContext:
    """Datos de mercado/cuenta auxiliares que no forman parte de la propuesta."""

    specs: Dict[str, SymbolSpec] = field(default_factory=dict)          # símbolo -> especificación (incluye posiciones abiertas)
    returns: Optional[pd.DataFrame] = None                              # retornos log por barra, columnas = símbolos
    bars_per_day: float = 96.0
    quote: Optional[Dict[str, float]] = None                            # {'bid','ask'} del símbolo propuesto
    margin_per_lot: Optional[float] = None                              # de order_calc_margin, si está disponible
    recent_r: List[float] = field(default_factory=list)                 # R de las últimas operaciones (más reciente al final)
    now: Optional[datetime] = None


class RiskAgent:
    """Filtro algorítmico estricto: aprueba (con tamaño) o rechaza (con motivos)."""

    name = "risk"

    def __init__(self, limits: RiskLimits | None = None, authority: ApprovalAuthority | None = None,
                 calendar: EconomicCalendar | None = None, clock=time.time) -> None:
        self.limits = limits or RiskLimits()
        self.authority = authority or DEFAULT_AUTHORITY
        self.calendar = calendar
        self._clock = clock
        self.halted_reason: Optional[str] = None

    def position_risk(self, pos: PositionInfo, spec: SymbolSpec, equity: float) -> float:
        if not pos.sl:
            return equity * self.limits.max_risk_per_trade          # sin SL: se asume el riesgo máximo por operación
        cur = pos.price_current or pos.price_open
        dist = (cur - pos.sl) if pos.direction == Direction.LONG else (pos.sl - cur)
        return max(dist, 0.0) / spec.tick_size * spec.tick_value * pos.volume

    def _spec_for(self, symbol: str, ctx: RiskContext, fallback: SymbolSpec) -> SymbolSpec:
        return ctx.specs.get(symbol, fallback)

    def _exposures(self, pf: PortfolioState, ctx: RiskContext, fallback: SymbolSpec,
                   extra: Optional[Tuple[str, int, float, float, SymbolSpec]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        per_symbol: Dict[str, float] = {}
        per_ccy: Dict[str, float] = {}

        def add(symbol: str, direction: int, lots: float, price: float, spec: SymbolSpec) -> None:
            n = signed_notional(direction, lots, price, spec)
            per_symbol[symbol] = per_symbol.get(symbol, 0.0) + n
            base, quote = pair_legs(symbol)
            per_ccy[base] = per_ccy.get(base, 0.0) + n
            per_ccy[quote] = per_ccy.get(quote, 0.0) - n

        for p in pf.positions:
            add(p.symbol, int(p.direction), p.volume, p.price_current or p.price_open, self._spec_for(p.symbol, ctx, fallback))
        if extra:
            add(*extra)
        return per_symbol, per_ccy

    def _var(self, per_symbol: Dict[str, float], pf: PortfolioState, ctx: RiskContext, fallback: SymbolSpec,
             open_risk_fallback: float) -> Tuple[float, str]:
        if ctx.returns is not None:
            v = portfolio_var(per_symbol, ctx.returns, self.limits.var_confidence, ctx.bars_per_day)
            if v is not None:
                return v, "param+hist"
        return 0.0, "no evaluable (sin historial); acotado por riesgo abierto total"

    def evaluate(self, proposal: TradeProposal, spec: SymbolSpec, pf: PortfolioState,
                 ctx: RiskContext | None = None) -> RiskDecision:
        ctx = ctx or RiskContext()
        lim = self.limits
        now = ctx.now or utcnow()
        checks: List[RiskCheck] = []
        reasons: List[str] = []

        def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
            checks.append(RiskCheck(name, ok, detail))
            if not ok and fatal:
                reasons.append(f"{name}: {detail}")
            return ok

        def reject() -> RiskDecision:
            log.info("RIESGO RECHAZA %s %s: %s", proposal.symbol, proposal.direction.label, "; ".join(reasons))
            return RiskDecision(False, checks=checks, reasons=reasons)

        eq = pf.equity
        entry, sl, tp = proposal.entry, proposal.sl, proposal.tp
        d = int(proposal.direction)

        if not check("equity", eq > 0, f"equity={eq:.2f}"):
            return reject()
        daily_loss = -pf.daily_pnl / pf.day_start_equity if pf.day_start_equity > 0 else 0.0
        if not check("pérdida_diaria", daily_loss < lim.max_daily_loss,
                     f"{daily_loss * 100:.2f}% >= límite {lim.max_daily_loss * 100:.1f}%"):
            return reject()
        if not check("drawdown", pf.drawdown < lim.max_drawdown,
                     f"{pf.drawdown * 100:.2f}% >= límite {lim.max_drawdown * 100:.1f}%"):
            return reject()

        if not check("dirección", proposal.direction in (Direction.LONG, Direction.SHORT), "FLAT no se opera"):
            return reject()
        side_ok = (sl < entry < tp) if d > 0 else (tp < entry < sl)
        if not check("niveles", entry > 0 and sl > 0 and tp > 0 and side_ok,
                     f"SL/TP incoherentes con {proposal.direction.label} (entry={entry}, sl={sl}, tp={tp})"):
            return reject()
        min_stop = spec.stops_level_points * spec.point
        if not check("stops_level", abs(entry - sl) >= min_stop and abs(tp - entry) >= min_stop,
                     f"distancia < nivel mínimo del broker ({min_stop:.5g})"):
            return reject()
        if not check("confianza", proposal.confidence >= lim.min_confidence,
                     f"{proposal.confidence:.2f} < {lim.min_confidence:.2f}"):
            return reject()
        if not check("win_prob", 0.0 < proposal.win_prob < 1.0, f"p={proposal.win_prob}"):
            return reject()

        risk_dist, reward_dist = abs(entry - sl), abs(tp - entry)
        spread_px = ((ctx.quote["ask"] - ctx.quote["bid"]) if ctx.quote else spec.spread_points * spec.point)
        if not check("spread", spread_px <= lim.max_spread_to_sl * risk_dist,
                     f"spread {spread_px:.5g} > {lim.max_spread_to_sl * 100:.0f}% del SL ({risk_dist:.5g})"):
            return reject()
        # R:R neto de costes: el spread se paga al entrar
        b_net = max(reward_dist - spread_px, 0.0) / (risk_dist + spread_px)
        if not check("reward_risk", b_net >= lim.min_reward_risk, f"R:R neto {b_net:.2f} < {lim.min_reward_risk:.2f}"):
            return reject()

        if self.calendar is not None:
            ev = self.calendar.blackout(proposal.symbol, now, lim.news_blackout_minutes, lim.news_blackout_minutes, "high")
            if not check("noticias", ev is None,
                         f"evento de alto impacto '{ev.title}' ({ev.currency}) a {(ev.time - now).total_seconds() / 60:+.0f} min" if ev else ""):
                return reject()

        same_symbol = [p for p in pf.positions if p.symbol == proposal.symbol]
        if not check("posiciones_totales", len(pf.positions) < lim.max_positions, f"{len(pf.positions)} >= {lim.max_positions}"):
            return reject()
        if not check("posiciones_símbolo", len(same_symbol) < lim.max_positions_per_symbol,
                     f"{len(same_symbol)} >= {lim.max_positions_per_symbol} en {proposal.symbol}"):
            return reject()

        f_full = kelly_fraction(proposal.win_prob, b_net)
        if not check("kelly", f_full > 0, f"f*={f_full:.4f} <= 0 (p={proposal.win_prob:.3f}, b={b_net:.2f}): sin ventaja esperada"):
            return reject()
        losses = 0
        for r in reversed(ctx.recent_r):
            if r < 0:
                losses += 1
            else:
                break
        k, parts = dynamic_kelly_scale(lim.kelly_fraction, pf.drawdown, lim.max_drawdown, losses, proposal.risk_weight)
        f_used = f_full * k
        risk_pct = min(f_used, lim.max_risk_per_trade)
        if not check("riesgo_mínimo", risk_pct >= MIN_RISK_PCT,
                     f"riesgo efectivo {risk_pct * 100:.3f}% < {MIN_RISK_PCT * 100:.2f}% (f*={f_full:.3f}, k={k:.3f})"):
            return reject()

        per_lot_risk = spec.risk_per_lot(entry, sl)
        if not check("riesgo_por_lote", per_lot_risk > 0, "riesgo por lote no positivo"):
            return reject()
        lots_kelly = size_from_risk(eq, risk_pct, entry, sl, spec)
        caps: Dict[str, float] = {"kelly": lots_kelly, "volumen_máx": spec.volume_max}

        open_risk = sum(self.position_risk(p, self._spec_for(p.symbol, ctx, spec), eq) for p in pf.positions)
        caps["riesgo_abierto_total"] = max((lim.max_total_open_risk * eq - open_risk) / per_lot_risk, 0.0)

        _, ccy_now = self._exposures(pf, ctx, spec)
        notional_now = sum(abs(signed_notional(int(p.direction), p.volume, p.price_current or p.price_open,
                                               self._spec_for(p.symbol, ctx, spec))) for p in pf.positions)
        n_per_lot = spec.notional_per_lot(entry)
        caps["apalancamiento"] = max((lim.max_leverage * eq - notional_now) / n_per_lot, 0.0)

        base, quote = pair_legs(proposal.symbol)
        cap_ccy = lim.max_currency_exposure * eq
        for ccy, sign in ((base, d), (quote, -d)):
            e_lot, cur = sign * n_per_lot, ccy_now.get(ccy, 0.0)
            room = (cap_ccy - cur) / e_lot if e_lot > 0 else (cap_ccy + cur) / abs(e_lot)
            caps[f"exposición_{ccy}"] = max(room, 0.0)

        used_margin = max(eq - pf.free_margin, 0.0) if pf.free_margin else 0.0
        mpl = ctx.margin_per_lot or (n_per_lot / max(pf.leverage, 1.0))
        caps["margen_libre"] = max(pf.free_margin, 0.0) / mpl if mpl > 0 else float("inf")
        if lim.min_margin_level > 0 and mpl > 0:
            caps["nivel_margen"] = max((eq / (lim.min_margin_level / 100.0) - used_margin) / mpl, 0.0)

        var_limit = lim.max_var_pct * eq
        var_method = ""

        def var_with(lots: float) -> float:
            nonlocal var_method
            ps, _ = self._exposures(pf, ctx, spec, extra=(proposal.symbol, d, lots, entry, spec))
            v, var_method = self._var(ps, pf, ctx, spec, open_risk + lots * per_lot_risk)
            return v

        cand = min(caps.values())
        if var_with(cand) > var_limit:
            lo, hi = 0.0, cand
            for _ in range(24):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if var_with(mid) <= var_limit else (lo, mid)
            caps["VaR"] = lo
        binding = min(caps, key=caps.get)
        lots = round_down_to_step(min(caps.values()), spec.volume_step)
        lots = min(lots, spec.volume_max)
        detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(caps.items(), key=lambda kv: kv[1])[:3])
        if not check("tamaño", lots >= spec.volume_min - 1e-12,
                     f"no cabe el lote mínimo {spec.volume_min}: límite vinculante '{binding}' ({detail})"):
            return reject()
        checks.append(RiskCheck("límite_vinculante", True, f"{binding} -> {lots:.2f} lotes ({detail})"))

        risk_amount = lots * per_lot_risk
        var_after = var_with(lots)
        if not check("VaR", var_after <= var_limit * (1 + 1e-6),
                     f"{var_after:,.2f} > {var_limit:,.2f} ({var_method}, {lim.var_confidence * 100:.0f}% 1d)"):
            return reject()
        checks[-1].detail = f"{var_after:,.2f} <= {var_limit:,.2f} ({var_method}, {lim.var_confidence * 100:.0f}% 1d)"
        order = self.authority.sign(ApprovedOrder(
            proposal_id=proposal.proposal_id, symbol=proposal.symbol, direction=d, volume=lots, price=entry, sl=sl, tp=tp,
            issued_at=self._clock(), comment=f"AGI|{proposal.proposal_id}"))
        return RiskDecision(True, lots=lots, risk_pct=risk_amount / eq, risk_amount=risk_amount, kelly_full=f_full,
                            kelly_used=f_used, var_after=var_after, checks=checks, reasons=[], order=order)
