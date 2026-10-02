"""Tests del detector de patrones institucionales con velas construidas a mano (resultado conocido)."""
import numpy as np
import pandas as pd
import pytest

from agents import smc
from core.indicators import atr, efficiency_ratio, rsi


def make_df(rows, start="2026-01-05 00:00", minutes=15):
    arr = np.array([r if len(r) == 5 else (*r, 100) for r in rows], dtype=float)
    idx = pd.date_range(start, periods=len(arr), freq=f"{minutes}min", tz="UTC")
    return pd.DataFrame(arr, columns=["open", "high", "low", "close", "tick_volume"], index=idx).assign(
        spread=10, real_volume=0)


def flat(n, base=100.0):
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append((base, base + 0.3, base - 0.1, base + 0.1))
        else:
            rows.append((base + 0.1, base + 0.3, base - 0.1, base))
    return rows


def test_indicators_basic():
    df = make_df(flat(60))
    a = atr(df).dropna()
    assert (a > 0).all() and a.iloc[-1] == pytest.approx(0.4, abs=0.05)
    r = rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    trend = pd.Series(np.arange(60, dtype=float))
    assert efficiency_ratio(trend, 20).iloc[-1] == pytest.approx(1.0)


def test_swing_points_confirmed_only():
    h = np.array([1, 2, 3, 5, 3, 2, 1, 2, 3, 2, 1], dtype=float)
    l = h - 0.5
    sh, sl = smc.swing_points(h, l, left=3, right=3)
    assert list(np.flatnonzero(sh)) == [3]
    # el pivote de la última barra nunca se confirma (right=3)
    assert not sh[-3:].any() and not sl[-3:].any()


def test_bullish_order_block_detected():
    rows = flat(40)
    rows.append((100.2, 100.3, 99.7, 99.8))         # vela bajista = OB
    rows.append((99.8, 101.6, 99.8, 101.5))         # desplazamiento alcista con BOS
    rows += [(101.5, 101.8, 101.4, 101.7)] * 4
    zones = smc.order_blocks(make_df(rows))
    assert len(zones) == 1
    z = zones[0]
    assert (z.kind, z.direction, z.index) == ("OB", 1, 40)
    assert (z.bottom, z.top) == (99.7, 100.3) and z.strength > 3 and not z.tested


def test_order_block_invalidated_by_close_through():
    rows = flat(40)
    rows.append((100.2, 100.3, 99.7, 99.8))
    rows.append((99.8, 101.6, 99.8, 101.5))
    rows += [(101.5, 101.6, 99.0, 99.2)]            # cierre por debajo de la zona => invalida el OB alcista
    zones = smc.order_blocks(make_df(rows))
    assert [z for z in zones if z.direction == 1] == []
    assert [z.index for z in zones if z.direction == -1] == [41]


def test_order_block_tested_flag():
    rows = flat(40)
    rows.append((100.2, 100.3, 99.7, 99.8))
    rows.append((99.8, 101.6, 99.8, 101.5))
    rows.append((101.5, 101.6, 100.2, 100.9))       # vuelve a tocar la zona (low <= top) sin cerrar debajo
    z = smc.order_blocks(make_df(rows))[0]
    assert z.tested


def test_bearish_order_block_detected():
    rows = flat(40)
    rows.append((99.9, 100.3, 99.85, 100.2))        # vela alcista = OB de oferta
    rows.append((100.2, 100.2, 98.4, 98.5))         # desplazamiento bajista con BOS
    z = smc.order_blocks(make_df(rows))[0]
    assert z.direction == -1 and z.index == 40


def test_liquidity_sweep_bearish_and_bullish():
    rows = flat(30)
    bear = smc.liquidity_sweeps(make_df(rows + [(100.1, 100.9, 100.0, 100.15)]))
    assert len(bear) == 1 and bear[0]["direction"] == -1 and bear[0]["age"] == 0
    bull = smc.liquidity_sweeps(make_df(rows + [(100.0, 100.15, 99.3, 100.05)]))
    assert len(bull) == 1 and bull[0]["direction"] == 1


def test_no_sweep_when_close_stays_beyond():
    rows = flat(30) + [(100.1, 100.9, 100.0, 100.8)]    # cierra por encima => es ruptura, no barrido
    assert smc.liquidity_sweeps(make_df(rows)) == []


def test_fair_value_gap():
    rows = flat(20) + [(100.0, 100.2, 99.9, 100.1), (100.1, 101.6, 100.1, 101.5), (101.5, 101.9, 101.0, 101.8),
                       (101.8, 102.0, 101.6, 101.9)]
    fvg = [z for z in smc.fair_value_gaps(make_df(rows)) if z.direction == 1]
    assert len(fvg) == 1
    # hueco = [H de la vela i-2, L de la vela i] = [100.2, 101.0]
    assert (fvg[0].bottom, fvg[0].top) == (100.2, 101.0) and not fvg[0].tested
    # si el precio rellena el hueco por completo, deja de contar
    filled = rows + [(101.8, 101.9, 100.0, 100.1)]
    assert [z for z in smc.fair_value_gaps(make_df(filled)) if z.direction == 1 and z.index == 21] == []


def test_market_structure_bullish_sequence():
    # zig-zag ascendente: HH y HL
    ups = [100, 103, 101, 105, 102.5, 108, 104, 111, 106]
    rows = []
    for k in range(len(ups) - 1):
        a, b = ups[k], ups[k + 1]
        for t in np.linspace(a, b, 5)[:-1]:
            rows.append((t, t + 0.2, t - 0.2, t + (b - a) / 5))
    ms = smc.market_structure(make_df(rows), left=2, right=2)
    assert ms["bias"] == 1


def _range_with_tail(tail):
    prior = [(110 - 0.25 * i, 110.1 - 0.25 * i, 109.9 - 0.25 * i, 109.8 - 0.25 * i) for i in range(45)]
    rng = []
    for i in range(70):
        top = i % 10 < 5
        base = 100.0 if top else 99.0
        rng.append((base, base + 0.4, base - 0.4, base + (0.1 if top else -0.1)))
    return make_df(prior + rng + tail)


def test_wyckoff_spring_is_accumulation():
    df = _range_with_tail([(99.0, 99.2, 97.9, 99.1, 400)] + [(99.1, 99.4, 98.8, 99.2)] * 4)
    res = smc.wyckoff_phase(df)
    assert res["phase"] == "accumulation" and res["bias"] > 0.5


def test_wyckoff_upthrust_is_distribution():
    prior = [(90 + 0.25 * i, 90.4 + 0.25 * i, 89.9 + 0.25 * i, 90.2 + 0.25 * i) for i in range(45)]
    rng = []
    for i in range(70):
        top = i % 10 < 5
        base = 101.0 if top else 100.0
        rng.append((base, base + 0.4, base - 0.4, base + (0.1 if top else -0.1)))
    tail = [(101.0, 102.3, 100.9, 100.9, 400)] + [(100.9, 101.1, 100.5, 100.7)] * 4
    res = smc.wyckoff_phase(make_df(prior + rng + tail))
    assert res["phase"] == "distribution" and res["bias"] < -0.5


def test_wyckoff_short_history_is_unclear():
    assert smc.wyckoff_phase(make_df(flat(50)))["phase"] == "unclear"
