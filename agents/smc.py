"""Detección de patrones institucionales (Smart Money Concepts + Wyckoff) sobre velas OHLCV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.indicators import atr as _atr


@dataclass
class Zone:
    """Zona de oferta/demanda (Order Block o FVG)."""

    kind: str            # "OB" | "FVG"
    direction: int       # +1 demanda (alcista), -1 oferta (bajista)
    top: float
    bottom: float
    index: int           # posición de la vela que define la zona
    formed: int          # posición de la barra que la confirma
    strength: float      # desplazamiento en múltiplos de ATR
    tested: bool = False

    def contains(self, price: float, pad: float = 0.0) -> bool:
        return self.bottom - pad <= price <= self.top + pad


def swing_points(high: np.ndarray, low: np.ndarray, left: int = 3, right: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Pivotes fractales confirmados."""
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    w = left + right + 1
    if n < w:
        return sh, sl
    hw = sliding_window_view(high, w)
    lw = sliding_window_view(low, w)
    sh[left:n - right] = (hw[:, left] > hw[:, :left].max(axis=1)) & (hw[:, left] >= hw[:, left + 1:].max(axis=1))
    sl[left:n - right] = (lw[:, left] < lw[:, :left].min(axis=1)) & (lw[:, left] <= lw[:, left + 1:].min(axis=1))
    return sh, sl


def market_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, object]:
    """Sesgo de estructura por secuencia de pivotes."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    sh, sl = swing_points(h, l, left, right)
    hi_idx, lo_idx = np.flatnonzero(sh), np.flatnonzero(sl)
    bias = 0
    if len(hi_idx) >= 2 and len(lo_idx) >= 2:
        hh = h[hi_idx[-1]] > h[hi_idx[-2]]
        hl = l[lo_idx[-1]] > l[lo_idx[-2]]
        if hh and hl:
            bias = 1
        elif (not hh) and (not hl):
            bias = -1
    return {
        "bias": bias,
        "last_swing_high": float(h[hi_idx[-1]]) if len(hi_idx) else None,
        "last_swing_low": float(l[lo_idx[-1]]) if len(lo_idx) else None,
        "swing_high_idx": hi_idx, "swing_low_idx": lo_idx,
    }


def order_blocks(df: pd.DataFrame, atr: np.ndarray | None = None, disp_mult: float = 1.6, struct_lookback: int = 10,
                 max_back: int = 6, lookback: int = 150) -> List[Zone]:
    """Detecta Order Blocks institucionales no invalidados."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(df)
    a = atr if atr is not None else _atr(df).to_numpy()
    zones: Dict[int, Zone] = {}
    for i in range(max(struct_lookback + 1, n - lookback), n):
        ai = a[i - 1]
        if not np.isfinite(ai) or ai <= 0:
            continue
        body = c[i] - o[i]
        if abs(body) < disp_mult * ai:
            continue
        if body > 0 and c[i] > h[i - struct_lookback:i].max():
            for j in range(i - 1, max(i - max_back - 1, -1), -1):
                if c[j] < o[j]:
                    zones.setdefault(j, Zone("OB", 1, float(h[j]), float(l[j]), j, i, abs(body) / ai))
                    break
        elif body < 0 and c[i] < l[i - struct_lookback:i].min():
            for j in range(i - 1, max(i - max_back - 1, -1), -1):
                if c[j] > o[j]:
                    zones.setdefault(j, Zone("OB", -1, float(h[j]), float(l[j]), j, i, abs(body) / ai))
                    break
    alive: List[Zone] = []
    for z in zones.values():
        after = slice(z.formed + 1, n)
        if z.direction == 1:
            if (c[after] < z.bottom).any():
                continue
            z.tested = bool((l[after] <= z.top).any())
        else:
            if (c[after] > z.top).any():
                continue
            z.tested = bool((h[after] >= z.bottom).any())
        alive.append(z)
    return sorted(alive, key=lambda z: z.formed)


def fair_value_gaps(df: pd.DataFrame, atr: np.ndarray | None = None, min_gap_atr: float = 0.3,
                    lookback: int = 80) -> List[Zone]:
    """Fair Value Gaps (desequilibrios de 3 velas) aún sin rellenar."""
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    n = len(df)
    a = atr if atr is not None else _atr(df).to_numpy()
    out: List[Zone] = []
    for i in range(max(2, n - lookback), n):
        ai = a[i - 1]
        if not np.isfinite(ai) or ai <= 0:
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap_atr * ai:
            z = Zone("FVG", 1, float(l[i]), float(h[i - 2]), i - 1, i, (l[i] - h[i - 2]) / ai)
            if not (l[i + 1:] <= z.bottom).any():
                z.tested = bool((l[i + 1:] <= z.top).any())
                out.append(z)
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap_atr * ai:
            z = Zone("FVG", -1, float(l[i - 2]), float(h[i]), i - 1, i, (l[i - 2] - h[i]) / ai)
            if not (h[i + 1:] >= z.top).any():
                z.tested = bool((h[i + 1:] >= z.bottom).any())
                out.append(z)
    return out


def liquidity_sweeps(df: pd.DataFrame, atr: np.ndarray | None = None, lookback: int = 24, min_wick_atr: float = 0.25,
                     max_age: int = 8) -> List[Dict[str, float]]:
    """Barridos de liquidez recientes (*stop hunts*)."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(df)
    a = atr if atr is not None else _atr(df).to_numpy()
    out: List[Dict[str, float]] = []
    for i in range(max(lookback, n - max_age), n):
        ai = a[i - 1]
        if not np.isfinite(ai) or ai <= 0:
            continue
        prior_hi, prior_lo = h[i - lookback:i].max(), l[i - lookback:i].min()
        upper, lower = h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]
        if h[i] > prior_hi and c[i] < prior_hi and upper >= min_wick_atr * ai:
            out.append({"direction": -1, "level": float(prior_hi), "extreme": float(h[i]),
                        "strength": float((h[i] - prior_hi) / ai + upper / ai), "age": n - 1 - i})
        if l[i] < prior_lo and c[i] > prior_lo and lower >= min_wick_atr * ai:
            out.append({"direction": 1, "level": float(prior_lo), "extreme": float(l[i]),
                        "strength": float((prior_lo - l[i]) / ai + lower / ai), "age": n - 1 - i})
    return out


def wyckoff_phase(df: pd.DataFrame, atr: np.ndarray | None = None, window: int = 80, recent: int = 10) -> Dict[str, object]:
    """Clasificación heurística de fase de Wyckoff."""
    n = len(df)
    if n < window + 40:
        return {"phase": "unclear", "bias": 0.0, "evidence": ["histórico insuficiente"]}
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    v = df["tick_volume"].to_numpy(dtype=float)
    a = atr if atr is not None else _atr(df).to_numpy()
    seg = slice(n - window, n)
    base = slice(n - window, n - recent)
    hi0, lo0 = h[base].max(), l[base].min()
    path = np.abs(np.diff(c[seg])).sum()
    er = abs(c[-1] - c[n - window]) / path if path > 0 else 0.0
    prior_move = c[n - window] - c[n - window - 40]
    vol_ref = np.median(v[seg]) or 1.0
    evidence: List[str] = []
    last_a = a[-1] if np.isfinite(a[-1]) and a[-1] > 0 else (h[seg] - l[seg]).mean()
    in_range = er < 0.22 and (hi0 - lo0) / last_a < 16

    if in_range:
        spring = [i for i in range(n - recent, n) if l[i] < lo0 and c[i] > lo0]
        upthr = [i for i in range(n - recent, n) if h[i] > hi0 and c[i] < hi0]
        if spring:
            vol_ok = any(v[i] > 1.3 * vol_ref for i in spring)
            evidence.append(f"spring bajo {lo0:.5g}" + (" con volumen" if vol_ok else ""))
            trend = prior_move < 0
            return {"phase": "accumulation", "bias": (0.8 if trend else 0.5) + (0.1 if vol_ok else 0.0),
                    "evidence": evidence + (["tendencia bajista previa"] if trend else [])}
        if upthr:
            vol_ok = any(v[i] > 1.3 * vol_ref for i in upthr)
            evidence.append(f"upthrust sobre {hi0:.5g}" + (" con volumen" if vol_ok else ""))
            trend = prior_move > 0
            return {"phase": "distribution", "bias": -((0.8 if trend else 0.5) + (0.1 if vol_ok else 0.0)),
                    "evidence": evidence + (["tendencia alcista previa"] if trend else [])}
        lean = 0.15 if prior_move < 0 else -0.15
        return {"phase": "range", "bias": lean, "evidence": [f"rango ER={er:.2f}"]}

    if c[-1] > hi0 and er > 0.3:
        return {"phase": "markup", "bias": 0.4, "evidence": [f"ruptura sobre rango, ER={er:.2f}"]}
    if c[-1] < lo0 and er > 0.3:
        return {"phase": "markdown", "bias": -0.4, "evidence": [f"ruptura bajo rango, ER={er:.2f}"]}
    return {"phase": "unclear", "bias": 0.0, "evidence": [f"ER={er:.2f}"]}
