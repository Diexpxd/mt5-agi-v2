"""Ingeniería de características causales y etiquetado para los modelos temporales."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.indicators import atr, efficiency_ratio, ema, rolling_zscore, rsi

FEATURE_NAMES = [
    "ret1", "ret4", "body", "range", "upper_wick", "lower_wick", "rsi", "ema_dist20", "ema_slope",
    "atr_pct_z", "vol_z", "er20", "dist_hi50", "dist_lo50", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
N_FEATURES = len(FEATURE_NAMES)
CLIP = 6.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de características por barra, invariantes a la escala del precio."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(df, 14)
    apct = a / c
    e20, e50 = ema(c, 20), ema(c, 50)
    hour = df.index.hour.to_numpy()
    dow = df.index.dayofweek.to_numpy()
    f = pd.DataFrame(index=df.index)
    f["ret1"] = np.log(c / c.shift(1)) / apct
    f["ret4"] = np.log(c / c.shift(4)) / (apct * 2.0)
    f["body"] = (c - o) / a
    f["range"] = (h - l) / a
    f["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / a
    f["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / a
    f["rsi"] = rsi(c, 14) / 100.0 - 0.5
    f["ema_dist20"] = (c - e20) / a
    f["ema_slope"] = (e20 - e50) / a
    f["atr_pct_z"] = rolling_zscore(apct, 100)
    f["vol_z"] = rolling_zscore(df["tick_volume"].astype(float), 50)
    f["er20"] = efficiency_ratio(c, 20)
    f["dist_hi50"] = (h.rolling(50).max() - c) / a
    f["dist_lo50"] = (c - l.rolling(50).min()) / a
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return f.replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)[FEATURE_NAMES]


def make_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Ventanas deslizantes ``(N, seq_len, F)``; la ventana ``k`` termina en la fila ``k + seq_len - 1``."""
    w = sliding_window_view(values, (seq_len, values.shape[1]))[:, 0]
    return np.ascontiguousarray(w, dtype=np.float32)


def make_labels(df: pd.DataFrame, horizon: int = 12, k: float = 0.5) -> np.ndarray:
    """Etiqueta de 3 clases por retorno futuro normalizado por volatilidad."""
    c = df["close"]
    fwd = np.log(c.shift(-horizon) / c)
    thr = k * (atr(df, 14) / c) * np.sqrt(horizon)
    y = np.full(len(df), 1, dtype=np.int64)
    y[(fwd > thr).to_numpy()] = 2
    y[(fwd < -thr).to_numpy()] = 0
    y[fwd.isna().to_numpy() | thr.isna().to_numpy()] = -1
    return y


def training_set(df: pd.DataFrame, seq_len: int, horizon: int, k: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye ``X (N, L, F)``, ``y (N,)`` y los índices de fila (fin de cada ventana) descartando NaN/sin etiqueta."""
    feats = build_features(df)
    vals = feats.to_numpy(dtype=np.float32)
    labels = make_labels(df, horizon, k)
    wins = make_windows(np.nan_to_num(vals, nan=0.0), seq_len)
    end_rows = np.arange(seq_len - 1, len(df))
    valid_feat = ~np.isnan(vals).any(axis=1)
    csum = np.concatenate([[0], np.cumsum(valid_feat.astype(int))])
    full = (csum[end_rows + 1] - csum[end_rows + 1 - seq_len]) == seq_len
    ok = full & (labels[end_rows] >= 0)
    return wins[ok], labels[end_rows][ok], end_rows[ok]
