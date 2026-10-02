"""Ensamblaje del sistema completo (cableado de agentes, memoria y broker)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from agents.economic_calendar import EconomicCalendar
from agents.fundamental_agent import FundamentalAgent
from agents.interface_agent import InterfaceAgent
from agents.orchestrator import Orchestrator, OrchestratorConfig
from agents.risk_agent import RiskAgent
from agents.technical_agent import TechnicalAgent, TechnicalConfig
from config.settings import Settings, load_settings
from core.execution import Broker, MT5Broker, PaperBroker
from core.llm import LLMClient, make_llm
from core.mt5_connection import MT5Connection
from memory.trade_journal import TradeJournal
from memory.vector_store import open_store

log = logging.getLogger(__name__)


@dataclass
class System:
    settings: Settings
    conn: MT5Connection
    broker: Broker
    llm: LLMClient
    calendar: EconomicCalendar
    technical: TechnicalAgent
    fundamental: FundamentalAgent
    risk: RiskAgent
    journal: TradeJournal
    orchestrator: Orchestrator
    interface: InterfaceAgent

    def start(self) -> None:
        self.conn.connect()
        self.orchestrator.bootstrap()


def build_system(settings: Settings | None = None, mt5: Any = None, mode: Optional[str] = None,
                 llm: LLMClient | None = None, vector_backend: str = "auto", technical_config: TechnicalConfig | None = None,
                 orch_config: OrchestratorConfig | None = None, paper_balance: float = 10_000.0) -> System:
    """Construye todos los componentes. ``mode``: ``paper`` (por defecto) | ``demo`` (órdenes a la cuenta DEMO)."""
    settings = settings or load_settings()
    settings.ensure_dirs()
    mode = (mode or settings.execution_mode).lower()
    if mode not in ("paper", "demo"):
        raise ValueError(f"execution_mode inválido: {mode!r} (usa 'paper' o 'demo')")
    conn = MT5Connection(settings, mt5=mt5)
    broker: Broker = (MT5Broker(conn) if mode == "demo" else
                      PaperBroker(conn, initial_balance=paper_balance, state_file=settings.state_dir / "paper_state.json"))
    llm = llm or make_llm(settings)
    calendar = EconomicCalendar.from_json(settings.data_dir / "economic_calendar.json")
    news_store = open_store("news", settings.chroma_dir, backend=vector_backend)
    trade_store = open_store("trades", settings.chroma_dir, backend=vector_backend)
    fundamental = FundamentalAgent(settings, llm, store=news_store, calendar=calendar)
    technical = TechnicalAgent(settings, technical_config)
    risk = RiskAgent(settings.risk, authority=conn.authority, calendar=calendar)
    journal = TradeJournal(settings.state_dir / "journal.sqlite", trade_store)
    orch = Orchestrator(settings, conn, broker, technical, fundamental, risk, journal, calendar, orch_config)
    interface = InterfaceAgent(settings, orch, llm)
    log.info("Sistema ensamblado: modo=%s backend_vectorial=%s llm=%s", mode, news_store.backend, llm.name)
    return System(settings, conn, broker, llm, calendar, technical, fundamental, risk, journal, orch, interface)
