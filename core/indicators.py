"""Indicadores técnicos causales (solo usan datos hasta la barra t; sin look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial: ``EMA_t = a x_t + (1 - a) EMA_{t-1}``, con ``a = 2 / (span + 1)``."""
    return x.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Rango verdadero: ``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``."""
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR de Wilder: media exponencial de ``TR`` con ``alpha = 1/n``."""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder: ``RSI = 100 - 100 / (1 + RS)``, ``RS = media(subidas) / media(bajadas)``."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(up.notna(), np.nan)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Ratio de eficiencia de Kaufman: ``ER = |C_t - C_{t-n}| / sum_{i}|C_i - C_{i-1}|`` en [0, 1]."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)


def rolling_zscore(x: pd.Series, n: int = 50) -> pd.Series:
    """Z-score móvil: ``(x - media_n) / desv_n``."""
    m = x.rolling(n).mean()
    s = x.rolling(n).std()
    return (x - m) / s.replace(0, np.nan)
