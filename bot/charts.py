"""Gráficos para Telegram (matplotlib -> PNG en memoria)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SURFACE, INK, INK2, MUTED = "#1a1a19", "#ffffff", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"


def _fig(w: float = 9.0, h: float = 5.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=140, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _fmt_price(x: float) -> str:
    return f"{x:,.5f}" if abs(x) < 20 else f"{x:,.2f}"


def candlestick_png(df: pd.DataFrame, symbol: str, timeframe: str, zones: Sequence = (), levels: Optional[Dict[str, float]] = None,
                    n: int = 110, subtitle: str = "") -> bytes:
    """Velas con EMA20/EMA50, zonas OB/FVG (rectángulos translúcidos) y niveles (entrada/SL/TP)."""
    d = df.iloc[-n:]
    x = np.arange(len(d))
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    fig, ax = _fig()
    up = c >= o
    col = np.where(up, BLUE, RED)
    ax.vlines(x, l, h, colors=col, linewidth=0.8, zorder=2)
    ax.bar(x, np.abs(c - o) + 1e-12, bottom=np.minimum(o, c), width=0.62, color=col, zorder=3)
    ema20, ema50 = d["close"].ewm(span=20, adjust=False).mean(), d["close"].ewm(span=50, adjust=False).mean()
    full = df["close"]
    ema20, ema50 = full.ewm(span=20, adjust=False).mean().iloc[-n:], full.ewm(span=50, adjust=False).mean().iloc[-n:]
    ax.plot(x, ema20.to_numpy(), color=ORANGE, linewidth=1.6, label="EMA 20", zorder=4)
    ax.plot(x, ema50.to_numpy(), color=AQUA, linewidth=1.6, label="EMA 50", zorder=4)
    offset = len(df) - n
    for z in zones:
        x0 = max(z.index - offset, 0)
        if z.index - offset + 40 < 0:
            continue
        color = BLUE if z.direction > 0 else RED
        ax.add_patch(Rectangle((x0 - 0.5, z.bottom), len(d) - x0 + 0.5, z.top - z.bottom, facecolor=color, alpha=0.16,
                               edgecolor=color, linewidth=0.8, zorder=1))
        ax.text(len(d) - 0.3, (z.top + z.bottom) / 2, f"{z.kind} {'demanda' if z.direction > 0 else 'oferta'}",
                color=INK2, fontsize=7, va="center", ha="right")
    for name, val in (levels or {}).items():
        ax.axhline(val, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=5)
        ax.text(0.2, val, f" {name} {_fmt_price(val)}", color=INK, fontsize=7.5, va="bottom")
    ticks = np.linspace(0, len(d) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.index[i].strftime("%d-%b %H:%M") for i in ticks])
    ax.set_xlim(-1, len(d) + 0.5)
    ax.yaxis.tick_right()
    ax.grid(axis="x", visible=False)
    _title(ax, f"{symbol} · {timeframe}", subtitle or f"Último cierre {_fmt_price(float(c[-1]))} · {d.index[-1].strftime('%Y-%m-%d %H:%M')} UTC")
    ax.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=8, ncol=2)
    return _png(fig)


def equity_png(points: Sequence[Tuple[datetime, float]], title: str = "Curva de equity") -> bytes:
    """Serie única de equity (línea de 2 px; sin leyenda: el título nombra la serie) + anotación del último valor."""
    fig, ax = _fig()
    if len(points) >= 2:
        t = [p[0] for p in points]
        v = np.array([p[1] for p in points], dtype=float)
        ax.plot(t, v, color=BLUE, linewidth=2.0)
        ax.scatter([t[-1]], [v[-1]], s=28, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
        peak = np.maximum.accumulate(v)
        dd = float(((peak - v) / peak).max()) * 100
        ax.annotate(f"${v[-1]:,.2f}", (t[-1], v[-1]), textcoords="offset points", xytext=(-8, 10), ha="right", color=INK, fontsize=9)
        _title(ax, title, f"Inicio ${v[0]:,.2f} · Actual ${v[-1]:,.2f} ({(v[-1] / v[0] - 1) * 100:+.2f}%) · Drawdown máx {dd:.2f}%")
        fig.autofmt_xdate(rotation=0, ha="center")
    else:
        ax.text(0.5, 0.5, "Aún no hay suficientes datos", transform=ax.transAxes, ha="center", color=INK2)
        _title(ax, title)
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")
    return _png(fig)


def pnl_bars_png(daily: Sequence[Tuple[str, float]], title: str = "PnL por día") -> bytes:
    """Barras de PnL diario: azul = ganancia, rojo = pérdida (polaridad), ancladas a la línea base."""
    fig, ax = _fig()
    if daily:
        labels = [d[0] for d in daily]
        vals = np.array([d[1] for d in daily], dtype=float)
        ax.bar(range(len(vals)), vals, color=np.where(vals >= 0, BLUE, RED), width=0.55)
        ax.axhline(0, color=AXIS, linewidth=1.0)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+,.0f}", ha="center", va="bottom" if v >= 0 else "top", color=INK2, fontsize=7.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=0 if len(vals) <= 8 else 45, fontsize=7.5)
        _title(ax, title, f"Total {vals.sum():+,.2f} · {int((vals > 0).sum())} días verdes / {int((vals < 0).sum())} rojos")
    else:
        ax.text(0.5, 0.5, "Sin operaciones cerradas", transform=ax.transAxes, ha="center", color=INK2)
        _title(ax, title)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def sentiment_png(scores: Dict[str, float], title: str = "Sentiment Score por activo") -> bytes:
    """Barras horizontales divergentes en [-100, 100]: azul = alcista, rojo = bajista, valor rotulado."""
    fig, ax = _fig(8.0, 1.2 + 0.7 * max(len(scores), 1))
    items = sorted(scores.items(), key=lambda kv: kv[1])
    ys = np.arange(len(items))
    vals = np.array([v for _, v in items], dtype=float)
    ax.barh(ys, vals, color=np.where(vals >= 0, BLUE, RED), height=0.5)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    for y, v in zip(ys, vals):
        ax.text(v + (3 if v >= 0 else -3), y, f"{round(v):+d}" if round(v) else "0", va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in items], color=INK2, fontsize=9)
    ax.set_xlim(-115, 115)
    ax.grid(axis="y", visible=False)
    _title(ax, title, "−100 muy bajista · +100 muy alcista")
    return _png(fig)


def backtest_png(equity: pd.Series, drawdown: pd.Series, title: str = "Backtest") -> bytes:
    """Equity y drawdown en dos paneles (una escala por eje; nunca doble eje)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), dpi=140, facecolor=SURFACE, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.12})
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
    a1.plot(equity.index, equity.to_numpy(), color=BLUE, linewidth=1.8)
    a1.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    a1.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")
    a2.fill_between(drawdown.index, drawdown.to_numpy() * 100, 0, color=RED, alpha=0.55, linewidth=0)
    a2.set_ylabel("Drawdown %", color=INK2, fontsize=8)
    fig.autofmt_xdate(rotation=0, ha="center")
    return _png(fig)
