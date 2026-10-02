"""Descarga histórico de MT5 (velas y, opcionalmente, ticks) a data/history/ (CSV comprimido)."""
from __future__ import annotations

import argparse
from datetime import timedelta

from _common import force_mock

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
ap.add_argument("--symbols", nargs="*")
ap.add_argument("--tf")
ap.add_argument("--bars", type=int, default=60_000)
ap.add_argument("--ticks-hours", type=float, default=0.0, help="además, descarga los ticks de las últimas N horas")
args = ap.parse_args()
if args.mock:
    force_mock()

from backtesting.data import download_history, history_path, save_history   # noqa: E402
from config.settings import load_settings                                    # noqa: E402
from core.mt5_connection import MT5Connection                                # noqa: E402

s = load_settings()
tf = args.tf or s.timeframe
conn = MT5Connection(s)
conn.connect()
for sym in args.symbols or s.symbols:
    try:
        df = download_history(conn, sym, tf, args.bars)
        save_history(df, history_path(s.data_dir, sym, tf))
        print(f"{sym} {tf}: {len(df)} velas  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  → {history_path(s.data_dir, sym, tf)}")
        if args.ticks_hours > 0:
            end = conn.server_time()
            ticks = conn.get_ticks(sym, end - timedelta(hours=args.ticks_hours), end)
            path = s.data_dir / "history" / f"{sym}_ticks.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            ticks.to_csv(path, compression="gzip")
            print(f"   ticks: {len(ticks):,} → {path}")
    except Exception as exc:
        print(f"{sym}: ERROR {exc}")
conn.disconnect()
