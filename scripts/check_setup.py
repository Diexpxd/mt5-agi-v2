"""Diagnóstico previo al arranque (solo LECTURA: nunca envía órdenes)."""
from __future__ import annotations

import argparse
import importlib
import sys

from _common import ROOT, force_mock

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
args = ap.parse_args()
if args.mock:
    force_mock()

from config.settings import load_settings, is_mock_key   # noqa: E402

ok_all = True


def line(ok: bool | None, msg: str) -> None:
    global ok_all
    mark = {True: "✅", False: "❌", None: "⚠️ "}[ok]
    ok_all &= ok is not False
    print(f"{mark} {msg}")


print(f"Python {sys.version.split()[0]} · proyecto {ROOT}\n")
for mod in ("numpy", "pandas", "torch", "MetaTrader5", "chromadb", "telebot", "feedparser", "matplotlib", "truststore",
            "google.genai", "pytest", "dotenv"):
    try:
        m = importlib.import_module(mod)
        line(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as exc:
        line(False if mod not in ("google.genai", "truststore", "dotenv") else None, f"{mod}: {type(exc).__name__} {str(exc)[:80]}")

s = load_settings()
print()
line(None if is_mock_key(s.telegram_token) else True, "TELEGRAM_KEY " + ("no configurada (bot en modo consola)" if s.telegram_is_mock else "configurada"))
line(None if s.telegram_is_mock or s.telegram_admin_ids else False, f"TELEGRAM_ADMIN_IDS = {list(s.telegram_admin_ids) or 'vacío (nadie podrá operar el bot)'}")
line(None if s.llm_is_mock else True, "GEMINI_API_KEY " + ("no configurada (sentimiento por léxico local)" if s.llm_is_mock else f"configurada ({s.gemini_model})"))
print(f"   modo de ejecución: {s.execution_mode} · símbolos: {', '.join(s.symbols)} · timeframe {s.timeframe}")

print("\n--- MetaTrader 5 ---")
from core.exceptions import MT5ConnectionError, RealAccountBlockedError   # noqa: E402
from core.mt5_connection import MT5Connection                              # noqa: E402

conn = MT5Connection(s)
try:
    conn.connect()
    a = conn.get_account()
    line(True, f"conectado ({'SIMULADOR' if conn.is_mock else 'terminal MT5'}) · cuenta #{a.login} · servidor {a.server}")
    line(True, f"cuenta DEMO verificada · equity {a.equity:,.2f} {a.currency} · apalancamiento 1:{a.leverage}")
    print(f"   offset hora-servidor vs UTC: {conn.utc_offset.total_seconds() / 3600:+.1f} h "
          f"({'calibrado' if conn.offset_calibrated else 'NO calibrado (mercado cerrado): fija MT5_UTC_OFFSET_HOURS en .env si no es 0'})")
    for sym in s.symbols:
        try:
            spec = conn.get_symbol_spec(sym)
            df = conn.get_rates(sym, s.timeframe, 300)
            q = conn.get_quote(sym)
            line(True, f"{sym}: {len(df)} velas {s.timeframe} · último cierre {df['close'].iloc[-1]:g} · spread {spec.spread_points:g} pts · lote {spec.volume_min}-{spec.volume_max}")
        except Exception as exc:
            line(False, f"{sym}: {exc}")
    print(f"   latencia: {conn.latency.summary()}")
except RealAccountBlockedError as exc:
    line(False, f"CUENTA REAL DETECTADA → el sistema se negará a operar.\n   {exc}\n   Abre una cuenta DEMO en el terminal MT5 (Archivo → Abrir una cuenta demo).")
except MT5ConnectionError as exc:
    line(False, f"no se pudo conectar: {exc}\n   ¿Está abierto el terminal MT5? Puedes indicar MT5_PATH / MT5_LOGIN / MT5_PASSWORD / MT5_SERVER en .env")
finally:
    try:
        conn.disconnect()
    except Exception:
        pass

print("\n--- Datos y modelos ---")
cal = s.data_dir / "economic_calendar.json"
line(cal.exists() or None, f"calendario macro: {'presente' if cal.exists() else 'ausente → ejecuta scripts/update_calendar.py'}")
models = list(s.model_dir.glob("*.pt")) if s.model_dir.exists() else []
line(bool(models) or None, f"modelos entrenados: {len(models)} {'(el agente técnico usará solo reglas SMC hasta que ejecutes scripts/train_models.py)' if not models else ''}")
print("\nRESULTADO:", "todo listo ✅" if ok_all else "hay problemas ❌ (revisa los puntos marcados)")
sys.exit(0 if ok_all else 1)
