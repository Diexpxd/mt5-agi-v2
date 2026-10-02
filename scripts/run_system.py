"""Arranque del sistema completo: orquestador + agentes + bot de Telegram."""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time

from _common import ROOT, force_mock

parser = argparse.ArgumentParser(description="MT5 AGI V2")
parser.add_argument("--mode", choices=["paper", "demo"], help="paper = simula fills; demo = órdenes a la cuenta DEMO")
parser.add_argument("--mock", action="store_true", help="usa el broker MT5 simulado (sin terminal)")
parser.add_argument("--console", action="store_true", help="bot por consola en lugar de Telegram")
parser.add_argument("--no-bot", action="store_true", help="sin bot")
parser.add_argument("--train", action="store_true", help="entrena los modelos que falten al arrancar")
parser.add_argument("--cycles", type=int, default=0, help="ejecuta N ciclos y termina (0 = indefinido)")
parser.add_argument("--fast", action="store_true", help="con --mock: avanza el reloj simulado sin esperar")
args = parser.parse_args()

if args.mock:
    force_mock()

from config.settings import load_settings                                   # noqa: E402
from core.exceptions import MT5ConnectionError, RealAccountBlockedError     # noqa: E402
from core.logging_setup import setup_logging                                # noqa: E402
from system import build_system                                             # noqa: E402
from bot.telegram_bot import BotRunner, TradingBot                          # noqa: E402

settings = load_settings()
setup_logging(settings.log_dir)
log = logging.getLogger("run_system")

try:
    system = build_system(settings, mode=args.mode)
    system.conn.connect()
except RealAccountBlockedError as exc:
    log.critical("🛑 %s", exc)
    raise SystemExit(2)
except MT5ConnectionError as exc:
    log.critical("No se pudo conectar a MT5: %s\nSugerencia: abre el terminal MT5 con una cuenta DEMO, o prueba con --mock.", exc)
    raise SystemExit(3)

acct = system.conn.get_account()
log.info("Conectado: %s | cuenta DEMO #%s | %s | modo=%s | LLM=%s | Telegram=%s",
         "MOCK" if system.conn.is_mock else "MT5", getattr(acct, "login", "?"), getattr(acct, "server", "?"),
         system.broker.name, system.llm.name, "MOCK" if settings.telegram_is_mock else "real")
for sym, status in system.orchestrator.bootstrap(train_missing=args.train).items():
    log.info("Modelo %s: %s", sym, status)

stop = threading.Event()
runner = BotRunner(settings, TradingBot(settings, system.interface), system.orchestrator)


def drive() -> None:
    """Bucle del orquestador. En modo mock avanza el reloj de mercado simulado."""
    n = 0
    mock = system.conn.backend if system.conn.is_mock else None
    while not stop.is_set():
        if mock is not None:
            mock.advance(15)
        try:
            rep = system.orchestrator.run_cycle()
            if rep.opened or rep.closed:
                log.info("ciclo: abiertas=%s cerradas=%s equity=%.2f", rep.opened, rep.closed, rep.equity)
        except RealAccountBlockedError:
            log.critical("CUENTA REAL DETECTADA: parada de seguridad")
            stop.set()
            return
        except MT5ConnectionError as exc:
            log.error("MT5 no disponible: %s", exc)
            stop.wait(10)
            continue
        except Exception:
            log.exception("error en el ciclo")
        n += 1
        if args.cycles and n >= args.cycles:
            stop.set()
            return
        stop.wait(0.0 if (mock is not None and args.fast) else settings.cycle_seconds)


try:
    if args.console:
        t = threading.Thread(target=drive, daemon=True, name="orchestrator")
        t.start()
        asyncio.run(runner.run_console())
    elif not args.no_bot and not settings.telegram_is_mock:
        runner.start_in_thread()
        drive()
    else:
        if not args.no_bot:
            log.warning("TELEGRAM_KEY no configurada: bot desactivado (usa --console para probarlo por consola)")
        drive()
except KeyboardInterrupt:
    log.info("Ctrl+C: apagando…")
finally:
    stop.set()
    time.sleep(0.2)
    st = system.orchestrator.status()
    log.info("Fin. Equity %.2f · posiciones abiertas %d (las abiertas conservan su SL/TP)", st["equity"], len(st["positions"]))
    system.conn.disconnect()
