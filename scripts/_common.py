"""Utilidades comunes de los scripts (rutas, logging, conexión)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):                     # evita UnicodeEncodeError con emojis en consolas cp1252
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def force_mock() -> None:
    """Debe llamarse ANTES de ``load_settings()``: fuerza el backend MT5 simulado."""
    os.environ["USE_MOCK_MT5"] = "1"
