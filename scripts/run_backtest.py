"""Backtest del sistema contra histórico (MT5 descargado, CSV en data/history o sintético)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from _common import ROOT, force_mock

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "csv", "mt5"], default="synthetic")
ap.add_argument("--symbols", nargs="*")
ap.add_argument("--tf")
ap.add_argument("--bars", type=int, default=12_000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--train-frac", type=float, default=0.0, help=">0 activa el modelo temporal (walk-forward de un pliegue)")
ap.add_argument("--arch", choices=["transformer", "lstm"], default="lstm")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--no-learn", action="store_true", help="desactiva el aprendizaje de la memoria durante el backtest")
ap.add_argument("--commission", type=float, default=7.0)
ap.add_argument("--slippage", type=float, default=1.0)
ap.add_argument("--out", default="reports")
args = ap.parse_args()

from agents.economic_calendar import EconomicCalendar                                  # noqa: E402
from agents.technical_agent import TechnicalConfig                                     # noqa: E402
from backtesting.data import (history_path, load_history, download_history, specs_from_conn,   # noqa: E402
                              synthetic_dataset)
from backtesting.engine import BacktestConfig, BacktestEngine                          # noqa: E402
from backtesting.metrics import drawdown_series                                        # noqa: E402
from bot import charts                                                                 # noqa: E402
from config.settings import load_settings                                              # noqa: E402

s = load_settings()
tf = args.tf or s.timeframe
symbols = args.symbols or list(s.symbols)
if args.source == "synthetic":
    data, specs = synthetic_dataset(symbols, args.bars, tf, args.seed)
else:
    from core.mt5_connection import MT5Connection
    conn = MT5Connection(s)
    conn.connect()
    specs = specs_from_conn(conn, symbols)
    if args.source == "csv":
        data = {sym: load_history(history_path(s.data_dir, sym, tf)) for sym in symbols}
    else:
        data = {sym: download_history(conn, sym, tf, args.bars) for sym in symbols}
    conn.disconnect()

cfg = BacktestConfig(timeframe=tf, train_frac=args.train_frac, epochs=args.epochs, learn=not args.no_learn,
                     commission_per_lot=args.commission, slippage_points=args.slippage,
                     technical=TechnicalConfig(arch=args.arch), risk=s.risk)
cal = EconomicCalendar.from_json(s.data_dir / "economic_calendar.json")
print(f"Backtest {args.source} · {symbols} · {tf} · {min(len(d) for d in data.values())} velas/símbolo · ML={'sí' if args.train_frac else 'no'}")
res = BacktestEngine(data, specs, cfg, calendar=cal).run(progress=lambda k, n: print(f"  {k}/{n}", end="\r"))
print("\n" + res.markdown())

out = ROOT / args.out
out.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
(out / f"backtest_{stamp}.md").write_text(f"# Backtest {stamp}\n\nFuente: {args.source} · {symbols} · {tf}\n\n{res.markdown()}\n", encoding="utf-8")
(out / f"backtest_{stamp}.json").write_text(json.dumps({"metrics": res.metrics, "config": {k: str(v) for k, v in res.config.items()}},
                                                        indent=1, default=str), encoding="utf-8")
(out / f"backtest_{stamp}.png").write_bytes(charts.backtest_png(res.equity, drawdown_series(res.equity), f"Backtest {args.source} {tf}"))
print(f"\nInforme guardado en {out}\\backtest_{stamp}.(md|json|png)")
