"""Tipos de dominio compartidos por todos los agentes."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def label(self) -> str:
        return {1: "BUY", -1: "SELL", 0: "FLAT"}[int(self)]


@dataclass
class SymbolSpec:
    """Especificación de contrato de un símbolo."""

    symbol: str
    digits: int = 5
    point: float = 0.00001
    tick_size: float = 0.00001
    tick_value: float = 1.0
    contract_size: float = 100_000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    currency_base: str = "EUR"
    currency_profit: str = "USD"
    spread_points: float = 10.0
    stops_level_points: float = 0.0
    filling_mode: int = 1

    def pnl(self, direction: int, entry: float, exit_: float, lots: float) -> float:
        return direction * (exit_ - entry) / self.tick_size * self.tick_value * lots

    def risk_per_lot(self, entry: float, sl: float) -> float:
        return abs(entry - sl) / self.tick_size * self.tick_value

    def notional_per_lot(self, price: float) -> float:
        if self.currency_profit == "USD" and self.currency_base != "USD":
            return self.contract_size * price
        if self.currency_base == "USD":
            return self.contract_size
        return self.contract_size * price


@dataclass
class Signal:
    """Salida de un agente de análisis (técnico o fundamental)."""

    symbol: str
    direction: Direction
    confidence: float                       # [0, 1]
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    source: str = ""
    rationale: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class TradeProposal:
    """Propuesta de operación que el orquestador entrega al agente de riesgo."""

    symbol: str
    direction: Direction
    entry: float
    sl: float
    tp: float
    win_prob: float                         # probabilidad estimada de acierto (p en Kelly)
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    risk_weight: float = 1.0                # peso aprendido por la memoria (0.25 - 1.5)
    timestamp: datetime = field(default_factory=utcnow)
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def reward_risk(self) -> float:
        risk = abs(self.entry - self.sl)
        return abs(self.tp - self.entry) / risk if risk > 0 else 0.0


@dataclass
class PositionInfo:
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    price_open: float
    sl: float = 0.0
    tp: float = 0.0
    price_current: float = 0.0
    profit: float = 0.0
    open_time: Optional[datetime] = None
    magic: int = 0
    comment: str = ""


@dataclass
class PortfolioState:
    """Foto del estado de la cuenta que consume el agente de riesgo."""

    equity: float
    balance: float
    free_margin: float = 0.0
    margin_level: float = 0.0               # 0 cuando no hay posiciones (MT5 lo reporta así)
    positions: List[PositionInfo] = field(default_factory=list)
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    leverage: float = 100.0
    currency: str = "USD"

    @property
    def daily_pnl(self) -> float:
        return self.equity - (self.day_start_equity or self.equity)

    @property
    def drawdown(self) -> float:
        peak = max(self.peak_equity, self.equity)
        return 0.0 if peak <= 0 else (peak - self.equity) / peak


@dataclass
class RiskCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ApprovedOrder:
    """Orden aprobada por el agente de riesgo. Solo ``RiskAgent`` puede firmarla."""

    proposal_id: str
    symbol: str
    direction: int
    volume: float
    price: float
    sl: float
    tp: float
    issued_at: float                        # epoch seconds
    comment: str = "AGI"
    token: str = ""


@dataclass
class RiskDecision:
    approved: bool
    lots: float = 0.0
    risk_pct: float = 0.0
    risk_amount: float = 0.0
    kelly_full: float = 0.0
    kelly_used: float = 0.0
    var_after: float = 0.0
    checks: List[RiskCheck] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    order: Optional[ApprovedOrder] = None

    def summary(self) -> str:
        head = "APROBADA" if self.approved else "RECHAZADA"
        if self.approved:
            return f"{head}: {self.lots:.2f} lotes, riesgo {self.risk_pct * 100:.2f}% (${self.risk_amount:,.2f})"
        return f"{head}: " + "; ".join(self.reasons) if self.reasons else head


@dataclass
class ClosedTrade:
    """Operación cerrada, unidad de aprendizaje de la memoria."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_time: datetime
    close_time: datetime
    price_open: float
    price_close: float
    sl: float
    tp: float
    profit: float                           # moneda de la cuenta, neto de comisiones
    risk_amount: float = 0.0                # pérdida planificada a SL en la apertura
    exit_reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def r_multiple(self) -> float:
        return self.profit / self.risk_amount if self.risk_amount > 0 else 0.0


@dataclass
class OrderResult:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    comment: str = ""
    simulated: bool = False
