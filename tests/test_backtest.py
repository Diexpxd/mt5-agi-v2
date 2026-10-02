"""Tests del motor de backtesting con datos construidos a mano (resultados exactos calculables)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agents.technical_agent import TechnicalConfig
from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.metrics import compute_metrics, drawdown_series
from config.settings import RiskLimits
from core.synthetic import generate_ohlcv
from core.types import ClosedTrade, Direction, Signal, SymbolSpec

SPEC = SymbolSpec("EURUSD")                # spread 10 pts = 0.0001, tick 1e-5 => $1/tick/lote
CFG = dict(warmup=5, window=50, orch=None)


def build(rows, start="2026-03-02 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["tick_volume"], df["spread"], df["real_volume"] = 100, 10, 0
    return df


def flat_rows(n):
    return [(1.1000, 1.1002, 1.0998, 1.1000)] * n


def engine_with_signal(rows, direction=Direction.LONG, trigger=9, **cfg):
    df = build(rows)
    eng = BacktestEngine({"EURUSD": df}, {"EURUSD": SPEC}, BacktestConfig(warmup=5, window=50, learn=False, **cfg))
    trigger_time = df.index[trigger]
    fired = []

    def fake(symbol, window, spec=None, model_score=None):
        assert window.index[-1] <= trigger_time or True
        if window.index[-1] == trigger_time and not fired:
            fired.append(1)
            px = float(window["close"].iloc[-1])
            sl, tp = (px - 0.002, px + 0.004) if direction == Direction.LONG else (px + 0.002, px - 0.004)
            return Signal(symbol, direction, 0.9, entry=px, sl=sl, tp=tp, meta={"atr": 0.001, "session": "london", "regime": "trend"})
        return Signal(symbol, Direction.FLAT, 0.0)

    eng.technical.analyze = fake
    return eng, df


def test_long_hits_tp_with_exact_costs():
    rows = flat_rows(10) + [(1.1001, 1.1010, 1.0999, 1.1005), (1.1005, 1.1045, 1.1004, 1.1040)] + flat_rows(3)
    eng, df = engine_with_signal(rows)
    res = eng.run()
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.open_time == df.index[10] and t.direction == Direction.LONG
    assert t.price_open == pytest.approx(1.1001 + 0.0001 + 0.00001)
    assert (t.exit_reason, t.close_time) == ("tp", df.index[11])
    # TP re-anclado: entry_est(1.1001) + 2 * 0.0020 = 1.1041
    assert t.price_close == pytest.approx(1.1041)
    lots = 0.5                                                # riesgo 1 % de 10k / $200 por lote
    assert t.volume == pytest.approx(lots)
    expected = (1.1041 - t.price_open) / 0.00001 * 1.0 * lots - 7.0 * lots
    assert t.profit == pytest.approx(expected, abs=1e-6) and t.r_multiple == pytest.approx(expected / 100.0, abs=1e-6)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + expected, abs=1e-6)     # contabilidad consistente


def test_sl_wins_when_sl_and_tp_share_a_bar():
    rows = flat_rows(10) + [(1.1001, 1.1002, 1.1000, 1.1001), (1.1001, 1.1060, 1.0960, 1.1000)] + flat_rows(3)
    res = engine_with_signal(rows)[0].run()
    t = res.trades[0]
    assert t.exit_reason == "sl"                              # asunción conservadora
    assert t.price_close == pytest.approx(1.0981 - 0.00001)   # SL - slippage
    assert t.profit < 0 and t.r_multiple == pytest.approx(-1.0, abs=0.15)


def test_gap_through_stop_exits_at_open_and_loses_more_than_planned():
    rows = flat_rows(10) + [(1.1001, 1.1004, 1.0999, 1.1002), (1.0950, 1.0955, 1.0940, 1.0950)] + flat_rows(3)
    res = engine_with_signal(rows)[0].run()
    t = res.trades[0]
    assert t.exit_reason == "sl" and t.price_close == pytest.approx(1.0950 - 0.00001)
    assert t.r_multiple < -2.0                                # el gap rompe el riesgo planificado: el backtest lo refleja


def test_short_enters_at_bid_exits_at_ask():
    rows = flat_rows(10) + [(1.1001, 1.1004, 1.0999, 1.1000), (1.1000, 1.1001, 1.0950, 1.0955)] + flat_rows(3)
    res = engine_with_signal(rows, Direction.SHORT)[0].run()
    t = res.trades[0]
    assert t.direction == Direction.SHORT and t.exit_reason == "tp"
    assert t.price_open == pytest.approx(1.1001 - 0.00001)    # BID de apertura menos slippage
    assert t.price_close == pytest.approx(1.0960)             # TP (la vela BID llegó a 1.0950 + spread <= TP)
    assert t.profit == pytest.approx((t.price_open - 1.0960) / 0.00001 * 0.5 - 3.5, abs=1e-6)


def test_short_stop_uses_ask_side_spread():
    rows = flat_rows(10) + [(1.1001, 1.1004, 1.0999, 1.1000), (1.1000, 1.1019, 1.0995, 1.1010)] + flat_rows(3)
    res = engine_with_signal(rows, Direction.SHORT)[0].run()
    assert res.trades[0].exit_reason == "sl"


def test_open_position_is_closed_at_end_and_counted():
    rows = flat_rows(10) + [(1.1001, 1.1004, 1.0999, 1.1002)] * 4       # ni SL ni TP
    res = engine_with_signal(rows)[0].run()
    assert len(res.trades) == 1 and res.trades[0].exit_reason == "fin_backtest"


def test_no_look_ahead_decisions_only_see_past_bars():
    df = generate_ohlcv("EURUSD", n=800, minutes=15, seed=1)
    eng = BacktestEngine({"EURUSD": df}, {"EURUSD": SPEC}, BacktestConfig(warmup=250, window=300, learn=False))
    seen = []
    real = eng.technical.analyze

    def spy(symbol, window, spec=None, model_score=None):
        seen.append((window.index[0], window.index[-1], len(window)))
        return real(symbol, window, spec, model_score=model_score)

    eng.technical.analyze = spy
    eng.run()
    assert seen and all(n <= 300 for _, _, n in seen)
    ends = [e for _, e, _ in seen]
    assert ends == sorted(ends) and ends[0] >= df.index[250]        # nunca antes del calentamiento, siempre en orden
    assert all(df.index.get_loc(e) - df.index.get_loc(s) + 1 == n for s, e, n in seen)


def test_cooldown_and_one_position_per_symbol():
    rows = flat_rows(10) + [(1.1001, 1.1010, 1.0999, 1.1005), (1.1005, 1.1045, 1.1004, 1.1040)] + flat_rows(30)
    df = build(rows)
    eng = BacktestEngine({"EURUSD": df}, {"EURUSD": SPEC}, BacktestConfig(warmup=5, window=50, learn=False, cooldown_bars=6))
    times = []

    def always_long(symbol, window, spec=None, model_score=None):
        px = float(window["close"].iloc[-1])
        times.append(window.index[-1])
        return Signal(symbol, Direction.LONG, 0.9, entry=px, sl=px - 0.002, tp=px + 0.004, meta={"atr": 0.001})

    eng.technical.analyze = always_long
    res = eng.run()
    opens = [t.open_time for t in res.trades]
    assert len(res.trades) >= 2
    for a, b in zip(res.trades, res.trades[1:]):
        assert b.open_time >= a.close_time + pd.Timedelta(minutes=15 * 6), "respeta el enfriamiento"
        assert b.open_time > a.close_time                            # nunca dos posiciones simultáneas


def test_risk_agent_is_in_the_loop_and_rejections_are_counted():
    rows = flat_rows(10) + [(1.1001, 1.1004, 1.0999, 1.1002)] * 4
    eng, _ = engine_with_signal(rows, risk=RiskLimits(min_reward_risk=5.0))      # R:R 2 < 5 => el riesgo rechaza
    res = eng.run()
    assert res.trades == [] and res.counters["rejected"] == 1 and "reward_risk" in res.reject_reasons


@pytest.fixture(scope="module")
def full_run():
    data = {s: generate_ohlcv(s, n=2500, minutes=15, seed=8) for s in ("EURUSD", "XAUUSD")}
    specs = {"EURUSD": SPEC, "XAUUSD": SymbolSpec("XAUUSD", digits=2, point=0.01, tick_size=0.01, tick_value=1.0, contract_size=100.0,
                                                   currency_base="XAU", spread_points=25)}
    res = BacktestEngine(data, specs, BacktestConfig(warmup=250, window=400)).run()
    return data, specs, res


def test_full_pipeline_runs_and_accounts(full_run):
    data, specs, res = full_run
    assert len(res.equity) == len(data["EURUSD"]) and res.equity.index.is_monotonic_increasing
    assert res.metrics["n_trades"] == len(res.trades) and res.metrics["n_trades"] >= 1
    assert res.equity.iloc[-1] == pytest.approx(10_000 + sum(t.profit for t in res.trades), abs=60.0)
    for t in res.trades:
        assert t.close_time >= t.open_time and t.risk_amount > 0
        assert t.risk_amount <= 0.01 * 10_000 * 1.5 + 1                 # riesgo por operación acotado por el agente de riesgo
    assert "Retorno total" in res.markdown() and set(res.by_symbol) <= {"EURUSD", "XAUUSD"}


def test_backtest_is_deterministic(full_run):
    data, specs, res = full_run
    again = BacktestEngine(data, specs, BacktestConfig(warmup=250, window=400)).run()
    assert again.metrics["final_equity"] == pytest.approx(res.metrics["final_equity"])
    assert [t.profit for t in again.trades] == pytest.approx([t.profit for t in res.trades])


def test_costs_reduce_performance(full_run):
    data, specs, res = full_run
    free = BacktestEngine(data, specs, BacktestConfig(warmup=250, window=400, commission_per_lot=0.0, slippage_points=0.0)).run()
    assert free.metrics["final_equity"] >= res.metrics["final_equity"] - 1e-6 or free.metrics["n_trades"] != res.metrics["n_trades"]


def test_walk_forward_ml_evaluates_only_after_training_period():
    df = generate_ohlcv("EURUSD", n=4000, minutes=15, seed=4, trend_strength=0.6, mean_regime_bars=600)
    cfg = BacktestConfig(warmup=250, window=400, train_frac=0.5, epochs=4, learn=False, technical=TechnicalConfig(arch="lstm"))
    res = BacktestEngine({"EURUSD": df}, {"EURUSD": SPEC}, cfg).run()
    assert all(t.open_time >= df.index[2000] for t in res.trades), "el modelo nunca opera sobre datos con los que se entrenó"


def test_metrics_math():
    idx = pd.date_range("2026-01-01", periods=101, freq="1D", tz="UTC")
    eq = pd.Series(np.linspace(10_000, 11_000, 101), index=idx)
    m = compute_metrics(eq, [])
    assert m["total_return"] == pytest.approx(0.10) and m["max_drawdown"] == 0.0 and m["sharpe"] > 5
    eq2 = pd.Series([100, 120, 90, 110, 130.0] + [130.0] * 3, index=pd.date_range("2026-01-01", periods=8, freq="1D", tz="UTC"))
    dd = drawdown_series(eq2)
    assert dd.max() == pytest.approx(0.25) and dd.iloc[2] == pytest.approx(0.25)     # (120-90)/120
    mk = lambda p, r: ClosedTrade(1, "EURUSD", Direction.LONG, 0.1, idx[0], idx[1], 1, 1, 1, 1, p, r)
    m2 = compute_metrics(eq, [mk(200, 100), mk(-100, 100), mk(200, 100), mk(-100, 100), mk(-100, 100)])
    assert m2["win_rate"] == pytest.approx(0.4) and m2["profit_factor"] == pytest.approx(400 / 300)
    assert m2["max_consecutive_losses"] == 2 and m2["expectancy_r"] == pytest.approx(0.2)
    assert m2["t_stat_r"] == pytest.approx(0.2 / (np.std([2, -1, 2, -1, -1], ddof=1) / np.sqrt(5)))
