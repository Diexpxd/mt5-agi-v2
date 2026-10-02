"""Datos históricos para el backtest: descarga desde MT5, caché CSV comprimido y datos sintéticos."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from core.exceptions import DataUnavailableError
from core.mock_mt5 import MockMT5
from core.mt5_connection import MT5Connection
from core.synthetic import generate_ohlcv
from core.timeframes import tf_minutes
from core.types import SymbolSpec

log = logging.getLogger(__name__)


def history_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    return data_dir / "history" / f"{symbol}_{timeframe}.csv.gz"


def save_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, compression="gzip")


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "time"
    return df.sort_index()


def download_history(conn: MT5Connection, symbol: str, timeframe: str, bars: int = 50_000, chunk_days: int = 20) -> pd.DataFrame:
    """Descarga ``bars`` velas cerradas hacia atrás en bloques por rango de fechas (evita el límite de una sola llamada)."""
    end = conn.server_time()
    step = timedelta(days=chunk_days)
    frames, got, cur_end = [], 0, end
    minutes = tf_minutes(timeframe)
    while got < bars:
        start = cur_end - step
        try:
            part = conn.get_rates_range(symbol, timeframe, start, cur_end)
        except DataUnavailableError:
            break
        frames.append(part)
        got = len(pd.concat(frames)[lambda d: ~d.index.duplicated()])
        if part.index[0] >= cur_end or (cur_end - part.index[0]) < timedelta(minutes=minutes):
            break
        cur_end = part.index[0]
        if len(frames) > 400:
            break
    if not frames:
        raise DataUnavailableError(f"Sin histórico para {symbol} {timeframe}")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.iloc[-bars:]


def specs_from_conn(conn: MT5Connection, symbols: Iterable[str]) -> Dict[str, SymbolSpec]:
    return {s: conn.get_symbol_spec(s) for s in symbols}


def synthetic_dataset(symbols: Iterable[str], bars: int = 12_000, timeframe: str = "M15", seed: int = 7
                      ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, SymbolSpec]]:
    """Datos sintéticos + especificaciones de contrato (vía el broker simulado). SIN ventaja garantizada."""
    from config.settings import Settings

    symbols = list(symbols)
    minutes = tf_minutes(timeframe)
    data = {s: generate_ohlcv(s, n=bars, minutes=minutes, seed=seed) for s in symbols}
    mock = MockMT5(symbols=symbols, n_minutes=2000, seed=seed)
    conn = MT5Connection(Settings(symbols=tuple(symbols), mt5_max_retries=1), mt5=mock, sleep=lambda s: None)
    conn.connect()
    return data, specs_from_conn(conn, symbols)
