"""Entrena (y guarda) el modelo temporal de cada símbolo con histórico de MT5."""
from __future__ import annotations

import argparse

from _common import force_mock

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
ap.add_argument("--bars", type=int, default=30_000)
ap.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--symbols", nargs="*")
args = ap.parse_args()
if args.mock:
    force_mock()

import logging                                                           # noqa: E402

from agents.technical_agent import TechnicalAgent, TechnicalConfig        # noqa: E402
from backtesting.data import download_history                             # noqa: E402
from config.settings import load_settings                                 # noqa: E402
from core.logging_setup import setup_logging                              # noqa: E402
from core.mt5_connection import MT5Connection                             # noqa: E402

s = load_settings()
setup_logging(s.log_dir, logging.WARNING)
conn = MT5Connection(s)
conn.connect()
agent = TechnicalAgent(s, TechnicalConfig(arch=args.arch))
print(f"{'símbolo':8} {'muestras val':>12} {'acierto dir.':>12} {'ventaja':>8} {'confiable':>10}")
for sym in args.symbols or s.symbols:
    try:
        df = download_history(conn, sym, s.timeframe, args.bars) if not conn.is_mock else conn.get_rates(sym, s.timeframe, args.bars)
        r = agent.train(sym, df, s.timeframe, epochs=args.epochs)
        print(f"{sym:8} {r.n_dir:>12} {r.dir_acc:>12.3f} {r.edge:>+8.3f} {'SÍ' if r.trusted() else 'no (ignorado)':>10}")
    except Exception as exc:
        print(f"{sym:8} ERROR: {exc}")
conn.disconnect()
print("\nModelos guardados en", s.model_dir)
