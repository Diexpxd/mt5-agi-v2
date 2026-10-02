"""Métricas de rendimiento de un backtest."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from core.types import ClosedTrade


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown relativo: ``DD_t = (pico_t - equity_t) / pico_t`` con ``pico_t = max_{s<=t} equity_s``."""
    peak = equity.cummax()
    return ((peak - equity) / peak).fillna(0.0)


def compute_metrics(equity: pd.Series, trades: Sequence[ClosedTrade], exposure: float = 0.0) -> Dict[str, float]:
    """Calcula el conjunto estándar de métricas."""
    m: Dict[str, float] = {}
    if len(equity) < 3:
        return {"n_trades": len(trades)}
    e0, e1 = float(equity.iloc[0]), float(equity.iloc[-1])
    years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    r = equity.pct_change().dropna()
    ann = len(r) / years
    std = float(r.std())
    down = r[r < 0]
    dd = drawdown_series(equity)
    max_dd = float(dd.max())
    m.update(
        initial_equity=e0, final_equity=e1, total_return=e1 / e0 - 1.0, years=years,
        cagr=(e1 / e0) ** (1.0 / years) - 1.0 if e1 > 0 else -1.0,
        sharpe=(math.sqrt(ann) * float(r.mean()) / std) if std > 0 else 0.0,
        sortino=(math.sqrt(ann) * float(r.mean()) / float(down.std())) if len(down) > 1 and down.std() > 0 else 0.0,
        max_drawdown=max_dd, exposure=exposure,
    )
    m["calmar"] = m["cagr"] / max_dd if max_dd > 0 else 0.0
    n = len(trades)
    m["n_trades"] = n
    if n:
        pnl = np.array([t.profit for t in trades])
        rs = np.array([t.r_multiple for t in trades])
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        m.update(
            win_rate=float((pnl > 0).mean()), avg_win=float(wins.mean()) if len(wins) else 0.0,
            avg_loss=float(losses.mean()) if len(losses) else 0.0, expectancy=float(pnl.mean()),
            expectancy_r=float(rs.mean()), total_pnl=float(pnl.sum()),
            profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else (float("inf") if wins.sum() > 0 else 0.0),
            best_trade=float(pnl.max()), worst_trade=float(pnl.min()),
            avg_hold_hours=float(np.mean([(t.close_time - t.open_time).total_seconds() / 3600 for t in trades])),
        )
        m["t_stat_r"] = float(rs.mean() / (rs.std(ddof=1) / math.sqrt(n))) if n > 2 and rs.std(ddof=1) > 0 else 0.0
        streak = best = 0
        for x in pnl:
            streak = streak + 1 if x < 0 else 0
            best = max(best, streak)
        m["max_consecutive_losses"] = best
    return m


def per_symbol(trades: Sequence[ClosedTrade]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for sym in sorted({t.symbol for t in trades}):
        ts = [t for t in trades if t.symbol == sym]
        pnl = [t.profit for t in ts]
        out[sym] = {"n": len(ts), "pnl": sum(pnl), "win_rate": sum(1 for x in pnl if x > 0) / len(ts),
                    "avg_r": sum(t.r_multiple for t in ts) / len(ts)}
    return out


def format_markdown(m: Dict[str, float], by_symbol: Dict[str, Dict[str, float]] | None = None) -> str:
    def f(k, fmt="{:,.2f}", scale=1.0, suffix=""):
        v = m.get(k)
        return "—" if v is None else (fmt.format(v * scale) + suffix)
    rows = [
        ("Retorno total", f("total_return", "{:+.2f}", 100, " %")), ("CAGR", f("cagr", "{:+.2f}", 100, " %")),
        ("Sharpe (anualizado)", f("sharpe")), ("Sortino", f("sortino")), ("Máx. drawdown", f("max_drawdown", "{:.2f}", 100, " %")),
        ("Calmar", f("calmar")), ("Operaciones", f("n_trades", "{:.0f}")), ("Win rate", f("win_rate", "{:.1f}", 100, " %")),
        ("Profit factor", f("profit_factor")), ("Expectativa (R)", f("expectancy_r", "{:+.3f}")),
        ("t-estadístico de R", f("t_stat_r")), ("PnL total", f("total_pnl", "{:+,.2f}")), ("Exposición", f("exposure", "{:.1f}", 100, " %")),
        ("Máx. pérdidas seguidas", f("max_consecutive_losses", "{:.0f}")),
    ]
    out = "| Métrica | Valor |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    if by_symbol:
        out += "\n\n| Símbolo | Ops | PnL | Win rate | R medio |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {s} | {v['n']} | {v['pnl']:+,.2f} | {v['win_rate'] * 100:.0f} % | {v['avg_r']:+.2f} |" for s, v in by_symbol.items())
    return out
