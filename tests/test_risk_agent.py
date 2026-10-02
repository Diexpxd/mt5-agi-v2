"""Tests del agente de riesgo con cifras verificables a mano."""
import dataclasses
import math
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from agents.economic_calendar import EconEvent, EconomicCalendar
from agents.risk_agent import (PortfolioTracker, RiskAgent, RiskContext, dynamic_kelly_scale, kelly_fraction,
                               portfolio_var, round_down_to_step, size_from_risk)
from config.settings import RiskLimits
from core.approval import ApprovalAuthority
from core.exceptions import UnapprovedOrderError
from core.types import Direction, PortfolioState, PositionInfo, SymbolSpec, TradeProposal

NOW = datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc)
EURUSD = SymbolSpec("EURUSD")                       # tick 1e-5, $1/tick/lote, contrato 100k, spread 10 pts = 1 pip
GBPUSD = SymbolSpec("GBPUSD", currency_base="GBP")


def proposal(**kw):
    base = dict(symbol="EURUSD", direction=Direction.LONG, entry=1.1000, sl=1.0980, tp=1.1040, win_prob=0.45,
                confidence=0.7)
    base.update(kw)
    return TradeProposal(**base)


def pf(equity=10_000.0, **kw):
    base = dict(equity=equity, balance=equity, free_margin=equity, day_start_equity=equity, peak_equity=equity,
                leverage=100.0)
    base.update(kw)
    return PortfolioState(**base)


@pytest.fixture()
def auth():
    return ApprovalAuthority()


@pytest.fixture()
def agent(auth):
    return RiskAgent(RiskLimits(), authority=auth)


def reasons_of(dec):
    return " | ".join(dec.reasons)


def test_kelly_known_values():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)          # p - q/b = 0.6 - 0.2
    assert kelly_fraction(1 / 3, 2.0) == pytest.approx(0.0, abs=1e-12)   # equilibrio: p = 1/(1+b)
    assert kelly_fraction(0.4, 1.0) == pytest.approx(-0.2)
    assert kelly_fraction(0.5, 0.0) == 0.0 and kelly_fraction(1.0, 2) == 0.0 and kelly_fraction(0.0, 2) == 0.0


def test_dynamic_scale_components():
    k, parts = dynamic_kelly_scale(0.25, drawdown=0.05, max_drawdown=0.10, consecutive_losses=2, memory_weight=1.0)
    assert parts["dd_scale"] == pytest.approx(0.5) and parts["streak_scale"] == pytest.approx(0.7)
    assert k == pytest.approx(0.25 * 0.5 * 0.7)
    assert dynamic_kelly_scale(0.25, 0.0, 0.10, 0, 5.0)[1]["memory"] == 1.5      # recorte superior
    assert dynamic_kelly_scale(0.25, 0.0, 0.10, 0, 0.0)[1]["memory"] == 0.25     # recorte inferior
    assert dynamic_kelly_scale(0.25, 0.10, 0.10, 0)[0] == 0.0                    # en el límite de drawdown: 0
    assert dynamic_kelly_scale(0.25, 0.0, 0.10, 99)[1]["streak_scale"] == pytest.approx(0.4)


def test_size_and_rounding():
    assert size_from_risk(10_000, 0.01, 1.1000, 1.0980, EURUSD) == pytest.approx(0.5)   # $100 / $200 por lote
    assert size_from_risk(10_000, 0.01, 1.1000, 1.1000, EURUSD) == 0.0
    assert round_down_to_step(0.5678, 0.01) == 0.56 and round_down_to_step(0.1 + 0.2, 0.01) == 0.3
    assert round_down_to_step(0.29, 0.01) == 0.29 and round_down_to_step(0.99, 0.5) == 0.5


def test_var_matches_closed_form_single_asset():
    rng = np.random.default_rng(1)
    R = pd.DataFrame({"EURUSD": rng.normal(0, 0.001, 5000)})
    v = portfolio_var({"EURUSD": 100_000.0}, R, 0.95, bars_per_day=1.0)
    param = NormalDist().inv_cdf(0.95) * 100_000 * R["EURUSD"].std()
    assert v >= param * (1 - 1e-9) and v == pytest.approx(param, rel=0.08)


def test_var_diversification_and_hedge():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 0.001, 4000)
    R = pd.DataFrame({"A": a, "B": rng.normal(0, 0.001, 4000), "C": a})          # C == A (correlación 1)
    single = portfolio_var({"A": 100_000}, R, 0.95, 1.0)
    diversified = portfolio_var({"A": 100_000, "B": 100_000}, R, 0.95, 1.0)
    hedged = portfolio_var({"A": 100_000, "C": -100_000}, R, 0.95, 1.0)
    assert diversified < 2 * single * 0.85 and diversified > single      # sub-aditivo
    assert hedged < 0.6 * single                                         # la cobertura reduce el riesgo
    assert portfolio_var({"A": 1}, R.head(30), 0.95, 1.0) is None       # historial insuficiente
    assert portfolio_var({"ZZZ": 1}, R, 0.95, 1.0) is None


def test_approves_and_signs(agent, auth):
    dec = agent.evaluate(proposal(), EURUSD, pf())
    assert dec.approved and dec.lots == pytest.approx(0.50) and dec.risk_amount == pytest.approx(100.0)
    assert dec.risk_pct == pytest.approx(0.01) and dec.kelly_full == pytest.approx(0.154, abs=0.005)
    o = dec.order
    assert (o.symbol, o.direction, o.volume, o.sl, o.tp) == ("EURUSD", 1, 0.5, 1.0980, 1.1040)
    auth.verify(o)
    with pytest.raises(UnapprovedOrderError):
        auth.verify(dataclasses.replace(o, volume=5.0))
    assert any(c.name == "límite_vinculante" for c in dec.checks)


def test_short_direction_sizes_symmetrically(agent):
    dec = agent.evaluate(proposal(direction=Direction.SHORT, entry=1.1000, sl=1.1020, tp=1.0960), EURUSD, pf())
    assert dec.approved and dec.lots == pytest.approx(0.5) and dec.order.direction == -1


def test_kelly_bound_size_scales_with_drawdown_and_memory(agent):
    p = proposal(win_prob=0.36)                                    # ventaja pequeña: Kelly (no el tope) manda
    base = agent.evaluate(p, EURUSD, pf())
    dd = agent.evaluate(p, EURUSD, pf(9_500.0, peak_equity=10_000.0, day_start_equity=9_500.0))
    weak_mem = agent.evaluate(proposal(win_prob=0.36, risk_weight=0.25), EURUSD, pf())
    assert base.approved and dd.approved and weak_mem.approved
    assert base.lots == pytest.approx(0.19, abs=0.011)             # f*=0.0154 -> x0.25 -> 0.385 % -> 0.1925 lotes
    assert dd.lots < base.lots * 0.6                                # dd 5 % con límite 10 % => x0.5
    assert weak_mem.lots < base.lots * 0.35                         # peso de memoria 0.25
    p_cap = agent.evaluate(proposal(win_prob=0.6), EURUSD, pf())
    assert p_cap.risk_pct == pytest.approx(0.01)                    # nunca supera el tope por operación


def test_losing_streak_reduces_size(agent):
    p = proposal(win_prob=0.36)
    a = agent.evaluate(p, EURUSD, pf(), RiskContext(recent_r=[1.0, 2.0]))
    b = agent.evaluate(p, EURUSD, pf(), RiskContext(recent_r=[1.0, -1.0, -1.0, -1.0]))
    assert b.lots < a.lots * 0.6


@pytest.mark.parametrize("kw,needle", [
    (dict(win_prob=0.30), "kelly"),
    (dict(direction=Direction.FLAT), "dirección"),
    (dict(sl=1.1010), "niveles"),                                   # SL del lado equivocado para un LONG
    (dict(tp=1.0990), "niveles"),
    (dict(confidence=0.30), "confianza"),
    (dict(tp=1.1010), "reward_risk"),                               # R:R 0.5
    (dict(win_prob=1.2), "win_prob"),
])
def test_rejections(agent, kw, needle):
    dec = agent.evaluate(proposal(**kw), EURUSD, pf())
    assert not dec.approved and dec.order is None and needle in reasons_of(dec)


def test_wide_spread_rejected(agent):
    wide = dataclasses.replace(EURUSD, spread_points=400)           # 40 pips de spread vs SL de 20
    assert "spread" in reasons_of(agent.evaluate(proposal(), wide, pf()))
    ok = agent.evaluate(proposal(), EURUSD, pf(), RiskContext(quote={"bid": 1.1000, "ask": 1.1001}))
    assert ok.approved
    bad = agent.evaluate(proposal(), EURUSD, pf(), RiskContext(quote={"bid": 1.1000, "ask": 1.1060}))
    assert "spread" in reasons_of(bad)


def test_daily_loss_circuit_breaker(agent):
    dec = agent.evaluate(proposal(), EURUSD, pf(9_600.0, day_start_equity=10_000.0, peak_equity=10_000.0))
    assert not dec.approved and "pérdida_diaria" in reasons_of(dec)
    assert agent.evaluate(proposal(), EURUSD, pf(9_800.0, day_start_equity=10_000.0, peak_equity=10_000.0)).approved


def test_drawdown_circuit_breaker(agent):
    dec = agent.evaluate(proposal(), EURUSD, pf(10_500.0, peak_equity=12_000.0, day_start_equity=10_500.0))
    assert not dec.approved and "drawdown" in reasons_of(dec)


def test_news_blackout(auth):
    cal = EconomicCalendar([EconEvent(NOW + timedelta(minutes=15), "USD", "high", "Non-Farm Employment Change")])
    ag = RiskAgent(RiskLimits(), authority=auth, calendar=cal)
    dec = ag.evaluate(proposal(), EURUSD, pf(), RiskContext(now=NOW))
    assert not dec.approved and "noticias" in reasons_of(dec) and "Non-Farm" in reasons_of(dec)
    assert ag.evaluate(proposal(), EURUSD, pf(), RiskContext(now=NOW - timedelta(hours=3))).approved
    assert ag.evaluate(proposal(symbol="EURJPY"), EURUSD, pf(), RiskContext(now=NOW)).approved     # NFP no afecta EUR/JPY


def test_position_count_limits(agent):
    pos = lambda i, s: PositionInfo(i, s, Direction.LONG, 0.01, 1.0, sl=0.99, price_current=1.0)
    full = pf(positions=[pos(i, f"S{i}") for i in range(6)])
    assert "posiciones_totales" in reasons_of(agent.evaluate(proposal(), EURUSD, full))
    dup = pf(positions=[pos(1, "EURUSD")])
    assert "posiciones_símbolo" in reasons_of(agent.evaluate(proposal(), EURUSD, dup))


def _gbp_long(volume, price=1.2700, sl=1.2665):
    return PositionInfo(9, "GBPUSD", Direction.LONG, volume, price, sl=sl, tp=1.28, price_current=price)


def test_open_risk_budget_trims_size(auth):
    ag = RiskAgent(RiskLimits(max_leverage=50.0, max_currency_exposure=50.0), authority=auth)
    existing = _gbp_long(3.0, price=1.2700, sl=1.2700 - 0.0116667)
    dec = ag.evaluate(proposal(), EURUSD, pf(100_000.0, positions=[existing]), RiskContext(specs={"GBPUSD": GBPUSD}))
    assert dec.approved and dec.lots == pytest.approx(2.5, abs=0.01)
    assert dec.risk_amount == pytest.approx(500.0, abs=2.0)
    assert "riesgo_abierto_total" in [c for c in dec.checks if c.name == "límite_vinculante"][0].detail.split("->")[1]
    # sin presupuesto: la cartera ya está al límite
    full = _gbp_long(3.0, price=1.2700, sl=1.2700 - 0.0133334)                       # $4,000 = 4 %
    dec2 = ag.evaluate(proposal(), EURUSD, pf(100_000.0, positions=[full]), RiskContext(specs={"GBPUSD": GBPUSD}))
    assert not dec2.approved and "riesgo_abierto_total" in reasons_of(dec2)


def test_currency_exposure_trims_size(agent):
    state = pf(positions=[_gbp_long(0.3, sl=1.2690)])
    dec = agent.evaluate(proposal(), EURUSD, state, RiskContext(specs={"GBPUSD": GBPUSD}))
    assert dec.approved and dec.lots < 0.5
    room = (6.0 * 10_000 - 0.3 * 100_000 * 1.27) / (100_000 * 1.10)
    assert dec.lots == pytest.approx(math.floor(room * 100) / 100, abs=0.011)
    detail = [c for c in dec.checks if c.name == "límite_vinculante"][0].detail
    assert "exposición_USD" in detail.split("->")[1]


def test_opposite_position_frees_currency_room(agent):
    short_gbp = PositionInfo(9, "GBPUSD", Direction.SHORT, 0.3, 1.27, sl=1.2735, tp=1.26, price_current=1.27)
    dec = agent.evaluate(proposal(), EURUSD, pf(positions=[short_gbp]), RiskContext(specs={"GBPUSD": GBPUSD}))
    assert dec.approved and dec.lots == pytest.approx(0.5)


def test_leverage_cap(auth):
    ag = RiskAgent(RiskLimits(max_leverage=3.0, max_currency_exposure=50.0), authority=auth)
    dec = ag.evaluate(proposal(win_prob=0.6), EURUSD, pf())
    assert dec.approved and dec.lots == pytest.approx(math.floor(3.0 * 10_000 / 110_000 * 100) / 100)


def test_var_bound_size(auth):
    ag = RiskAgent(RiskLimits(), authority=auth)
    rng = np.random.default_rng(3)
    R = pd.DataFrame({"EURUSD": rng.normal(0, 0.002, 3000)})                   # vol muy alta
    ctx = RiskContext(returns=R, bars_per_day=96.0)
    dec = ag.evaluate(proposal(), EURUSD, pf(), ctx)
    assert dec.approved and dec.lots < 0.2
    assert dec.var_after <= 0.03 * 10_000 * (1 + 1e-6)
    v_next = portfolio_var({"EURUSD": (dec.lots + 0.03) * 110_000.0}, R, 0.95, 96.0)
    assert v_next > 0.03 * 10_000 * 0.97                                        # un poco más de tamaño ya rompería el límite
    assert "VaR" in [c for c in dec.checks if c.name == "límite_vinculante"][0].detail.split("->")[1]


def test_var_hedge_allows_more_size(auth):
    ag = RiskAgent(RiskLimits(max_currency_exposure=50, max_leverage=50, max_total_open_risk=1.0), authority=auth)
    rng = np.random.default_rng(4)
    e = rng.normal(0, 0.002, 3000)
    R = pd.DataFrame({"EURUSD": e, "GBPUSD": e * 0.9 + rng.normal(0, 0.0003, 3000)})
    ctx = RiskContext(returns=R, bars_per_day=96.0, specs={"GBPUSD": GBPUSD})
    hedge = PositionInfo(9, "GBPUSD", Direction.SHORT, 0.05, 1.27, sl=1.30, tp=1.2, price_current=1.27)
    same = PositionInfo(9, "GBPUSD", Direction.LONG, 0.05, 1.27, sl=1.20, tp=1.3, price_current=1.27)
    a = ag.evaluate(proposal(), EURUSD, pf(positions=[hedge]), ctx)
    b = ag.evaluate(proposal(), EURUSD, pf(positions=[same]), ctx)
    assert a.approved and b.approved and a.lots > b.lots


def test_margin_rejects_when_no_free_margin(agent):
    dec = agent.evaluate(proposal(), EURUSD, pf(free_margin=200.0))
    assert not dec.approved and "tamaño" in reasons_of(dec)


def test_min_lot_rejection_on_tiny_account(agent):
    dec = agent.evaluate(proposal(), EURUSD, pf(100.0))                        # 1 % = $1 -> 0.005 lotes < 0.01
    assert not dec.approved and "no cabe el lote mínimo" in reasons_of(dec)


def test_volume_step_respected(agent):
    spec = dataclasses.replace(EURUSD, volume_step=0.1, volume_min=0.1)
    dec = agent.evaluate(proposal(win_prob=0.6), spec, pf())
    assert dec.approved and round(dec.lots * 10, 6) == pytest.approx(round(dec.lots * 10))


def test_jpy_pair_uses_broker_tick_value(agent):
    jpy = SymbolSpec("USDJPY", digits=3, point=0.001, tick_size=0.001, tick_value=0.65, currency_base="USD",
                     currency_profit="JPY", spread_points=14)
    p = proposal(symbol="USDJPY", entry=150.00, sl=149.70, tp=150.60)          # 30 pips
    dec = agent.evaluate(p, jpy, pf())
    per_lot = 0.30 / 0.001 * 0.65                                               # $195 por lote
    assert dec.approved and dec.lots == pytest.approx(math.floor(100 / per_lot * 100) / 100, abs=0.011)


def test_portfolio_tracker_day_roll_and_peak_persistence(tmp_path):
    f = tmp_path / "risk.json"
    t = PortfolioTracker(f)
    t.update(10_000, NOW)
    t.update(11_000, NOW + timedelta(hours=1))
    t.update(10_500, NOW + timedelta(hours=2))
    assert t.day_start_equity == 10_000 and t.peak_equity == 11_000
    t.update(10_400, NOW + timedelta(days=1))
    assert t.day_start_equity == 10_400 and t.peak_equity == 11_000
    t2 = PortfolioTracker(f)                                                   # reinicio: no se pierde el pico
    assert t2.peak_equity == 11_000 and t2.day_start_equity == 10_400
    acc = type("A", (), dict(equity=10_400.0, balance=10_400.0, margin_free=10_400.0, margin_level=0.0, leverage=100, currency="USD"))
    state = t2.snapshot(acc, [], NOW + timedelta(days=1))
    assert state.drawdown == pytest.approx((11_000 - 10_400) / 11_000) and state.daily_pnl == 0
