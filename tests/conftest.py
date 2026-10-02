"""Fixtures compartidos. Ningún test toca red, MT5 real ni claves reales."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Garantiza que los tests jamás usen claves reales ni el terminal real.
os.environ["TELEGRAM_KEY"] = "MOCK_KEY"
os.environ["GEMINI_API_KEY"] = "MOCK_KEY"
os.environ["USE_MOCK_MT5"] = "1"
os.environ["MT5_LOGIN"] = "0"

from config.settings import Settings  # noqa: E402
from core.approval import ApprovalAuthority  # noqa: E402
from core.mock_mt5 import MockMT5  # noqa: E402
from core.mt5_connection import MT5Connection  # noqa: E402

SYMS = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        symbols=SYMS, mt5_max_retries=3, mt5_retry_base_delay=0.0, execution_mode="paper",
        data_dir=tmp_path / "data", state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
        chroma_dir=tmp_path / "state" / "chroma", model_dir=tmp_path / "state" / "models",
    )


@pytest.fixture()
def mock_mt5() -> MockMT5:
    return MockMT5(symbols=list(SYMS), seed=11, n_minutes=20_000)


@pytest.fixture()
def authority() -> ApprovalAuthority:
    return ApprovalAuthority(ttl_seconds=90.0)


@pytest.fixture()
def conn(settings, mock_mt5, authority) -> MT5Connection:
    c = MT5Connection(settings, mt5=mock_mt5, sleep=lambda s: None, authority=authority)
    c.connect()
    return c
