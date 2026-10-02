"""Configuración central del sistema MT5 AGI V2."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent

try:  # python-dotenv es opcional
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except Exception:  # pragma: no cover - dotenv ausente
    pass


def is_mock_key(value: str | None) -> bool:
    """Indica si una clave es ausente o ficticia (``MOCK_*``)."""
    return (not value) or value.strip().upper().startswith("MOCK")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}


def _env_list(name: str, default: str) -> Tuple[str, ...]:
    return tuple(s.strip() for s in os.getenv(name, default).split(",") if s.strip())


def _state_root() -> Path:
    return ROOT_DIR / ("state_mock" if _env_bool("USE_MOCK_MT5", False) else "state")


@dataclass(frozen=True)
class RiskLimits:
    """Límites duros del agente de riesgo (The Bouncer)."""

    max_risk_per_trade: float = 0.01      # fracción máxima del equity arriesgada por operación
    kelly_fraction: float = 0.25          # Kelly fraccional (1.0 = Kelly completo, muy agresivo)
    min_reward_risk: float = 1.2          # R:R mínimo aceptado
    min_confidence: float = 0.55          # confianza mínima de la señal combinada
    max_spread_to_sl: float = 0.15        # spread / distancia_SL máximo
    max_total_open_risk: float = 0.04     # suma de riesgos (a SL) de todas las posiciones abiertas
    max_leverage: float = 10.0            # nocional total / equity
    max_currency_exposure: float = 6.0    # nocional neto por divisa / equity
    max_positions: int = 6
    max_positions_per_symbol: int = 1
    var_confidence: float = 0.95
    max_var_pct: float = 0.03             # VaR 1 día máximo como fracción del equity
    max_daily_loss: float = 0.03          # circuit breaker diario
    max_drawdown: float = 0.10            # circuit breaker de drawdown desde el pico
    news_blackout_minutes: int = 30       # ventana ± alrededor de eventos de alto impacto
    min_margin_level: float = 300.0       # % de nivel de margen mínimo tras abrir


@dataclass(frozen=True)
class Settings:
    """Ajustes globales inmutables."""

    mt5_login: int = field(default_factory=lambda: _env_int("MT5_LOGIN", 0))
    mt5_password: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: os.getenv("MT5_SERVER", ""))
    mt5_path: str = field(default_factory=lambda: os.getenv("MT5_PATH", ""))
    use_mock_mt5: bool = field(default_factory=lambda: _env_bool("USE_MOCK_MT5", False))
    mt5_max_retries: int = field(default_factory=lambda: _env_int("MT5_MAX_RETRIES", 5))
    mt5_init_timeout_ms: int = field(default_factory=lambda: _env_int("MT5_INIT_TIMEOUT_MS", 30_000))
    mt5_retry_base_delay: float = field(default_factory=lambda: _env_float("MT5_RETRY_DELAY", 1.0))
    mt5_utc_offset_hours: Optional[float] = field(
        default_factory=lambda: float(os.environ["MT5_UTC_OFFSET_HOURS"]) if os.getenv("MT5_UTC_OFFSET_HOURS", "").strip() else None)

    symbols: Tuple[str, ...] = field(default_factory=lambda: _env_list("SYMBOLS", "EURUSD,GBPUSD,USDJPY,XAUUSD"))
    timeframe: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "M15"))
    execution_mode: str = field(default_factory=lambda: os.getenv("EXECUTION_MODE", "paper").lower())
    cycle_seconds: int = field(default_factory=lambda: _env_int("CYCLE_SECONDS", 60))
    magic_number: int = field(default_factory=lambda: _env_int("MAGIC_NUMBER", 20260920))
    max_slippage_points: int = field(default_factory=lambda: _env_int("MAX_SLIPPAGE_POINTS", 20))

    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_KEY", "MOCK_KEY"))
    telegram_admin_ids: Tuple[int, ...] = field(
        default_factory=lambda: tuple(int(x) for x in _env_list("TELEGRAM_ADMIN_IDS", "") if x.lstrip("-").isdigit())
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "MOCK_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))

    data_dir: Path = ROOT_DIR / "data"
    state_dir: Path = field(default_factory=lambda: _state_root())
    log_dir: Path = ROOT_DIR / "logs"
    chroma_dir: Path = field(default_factory=lambda: _state_root() / "chroma")
    model_dir: Path = field(default_factory=lambda: _state_root() / "models")

    risk: RiskLimits = field(default_factory=RiskLimits)

    @property
    def telegram_is_mock(self) -> bool:
        return is_mock_key(self.telegram_token)

    @property
    def llm_is_mock(self) -> bool:
        return is_mock_key(self.gemini_api_key)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.state_dir, self.log_dir, self.chroma_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Construye ``Settings`` leyendo el entorno en el momento de la llamada."""
    s = Settings()
    s.ensure_dirs()
    return s
