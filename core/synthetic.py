"""Generador de datos OHLCV sintéticos con regímenes de mercado."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

SYMBOL_PROFILES: Dict[str, dict] = {
    "EURUSD": dict(price=1.0850, daily_vol=0.0050, digits=5, spread=12),
    "GBPUSD": dict(price=1.2700, daily_vol=0.0060, digits=5, spread=15),
    "USDJPY": dict(price=150.00, daily_vol=0.0060, digits=3, spread=14),
    "USDCAD": dict(price=1.3600, daily_vol=0.0045, digits=5, spread=18),
    "AUDUSD": dict(price=0.6600, daily_vol=0.0055, digits=5, spread=14),
    "XAUUSD": dict(price=2350.0, daily_vol=0.0110, digits=2, spread=25),
    "BTCUSD": dict(price=65000.0, daily_vol=0.0300, digits=2, spread=1500),
    "ETHUSD": dict(price=3200.0, daily_vol=0.0350, digits=2, spread=300),
}

DEFAULT_END = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)


def _seed_for(symbol: str, seed: int) -> int:
    return (abs(hash_str(symbol)) + seed * 7919) % (2**32 - 1)


def hash_str(s: str) -> int:
    """Hash determinista (``hash()`` de Python varía entre procesos)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _session_multiplier(hours: np.ndarray) -> np.ndarray:
    m = np.full(hours.shape, 0.75)
    m[(hours >= 7) & (hours < 16)] = 1.15   # Londres
    m[(hours >= 13) & (hours < 17)] = 1.40  # solape Londres-NY
    m[(hours >= 17) & (hours < 21)] = 0.95
    return m


def generate_ohlcv(
    symbol: str,
    n: int = 20_000,
    minutes: int = 1,
    seed: int = 7,
    end: datetime | None = None,
    start_price: float | None = None,
    trend_strength: float = 0.10,
    mean_regime_bars: int = 400,
) -> pd.DataFrame:
    """Genera ``n`` barras OHLCV de ``minutes`` minutos terminando en ``end`` (UTC)."""
    prof = SYMBOL_PROFILES.get(symbol.upper(), dict(price=100.0, daily_vol=0.006, digits=3, spread=20))
    rng = np.random.default_rng(_seed_for(symbol, seed))
    end = end or DEFAULT_END
    idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC", name="time")

    bars_per_day = 1440.0 / minutes
    sigma_base = prof["daily_vol"] / np.sqrt(bars_per_day)

    regime = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        length = int(rng.geometric(1.0 / mean_regime_bars))
        length = max(length, 20)
        state = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        regime[i:i + length] = state
        i += length

    h = np.zeros(n)
    xi = rng.normal(0, 0.06, n)
    for t in range(1, n):
        h[t] = 0.985 * h[t - 1] + xi[t]
    season = _session_multiplier(idx.hour.to_numpy())
    sigma = sigma_base * np.exp(h) * season

    eps = rng.normal(0, 1, n)
    r = np.empty(n)
    prev = 0.0
    for t in range(n):
        if regime[t] == 0:
            mu = -0.15 * prev
        else:
            mu = regime[t] * trend_strength * sigma_base
        r[t] = mu + sigma[t] * eps[t]
        prev = r[t]

    close = (start_price or prof["price"]) * np.exp(np.cumsum(r))
    open_ = np.concatenate([[close[0] / np.exp(r[0])], close[:-1]])
    wick = np.abs(rng.normal(0, 0.6, (n, 2))) * sigma[:, None]
    high = np.maximum(open_, close) * np.exp(wick[:, 0])
    low = np.minimum(open_, close) * np.exp(-wick[:, 1])

    z = np.abs(r) / (sigma + 1e-12)
    vol = rng.lognormal(mean=np.log(120), sigma=0.35, size=n) * season * (1 + 0.6 * z)
    digits = prof["digits"]
    df = pd.DataFrame(
        {
            "open": np.round(open_, digits),
            "high": np.round(high, digits),
            "low": np.round(low, digits),
            "close": np.round(close, digits),
            "tick_volume": vol.astype(np.int64),
            "spread": np.full(n, prof["spread"], dtype=np.int32),
            "real_volume": np.zeros(n, dtype=np.int64),
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Reagrupa barras de menor timeframe a ``minutes`` minutos (OHLC estándar)."""
    if minutes <= 1:
        return df.copy()
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "tick_volume": "sum", "spread": "max", "real_volume": "sum",
    }
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])
