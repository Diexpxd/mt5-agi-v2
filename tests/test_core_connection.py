"""Tests de la capa de conexión: seguridad DEMO, reconexión, datos y firma de órdenes."""
import dataclasses
import time
from datetime import timedelta

import pytest

from core.approval import ApprovalAuthority
from core.exceptions import (
    MT5ConnectionError,
    OrderRejectedError,
    RealAccountBlockedError,
    UnapprovedOrderError,
)
from core.mock_mt5 import MockMT5
from core.mt5_connection import MT5Connection
from core.types import ApprovedOrder, Direction


def _order(conn, authority, symbol="EURUSD", direction=Direction.LONG, volume=0.10, rr=2.0, dist=0.0020):
    q = conn.get_quote(symbol)
    price = q["ask"] if direction == Direction.LONG else q["bid"]
    sl = price - dist * direction
    tp = price + dist * rr * direction
    o = ApprovedOrder(proposal_id="t1", symbol=symbol, direction=int(direction), volume=volume, price=price,
                      sl=round(sl, 5), tp=round(tp, 5), issued_at=time.time())
    return authority.sign(o)


def test_connect_demo_ok(conn):
    assert conn.is_alive()
    assert conn.assert_demo_account().trade_mode == 0


def test_real_account_blocked_on_connect(settings, authority):
    real = MockMT5(symbols=["EURUSD"], n_minutes=2000, account_mode=MockMT5.ACCOUNT_TRADE_MODE_REAL, server="Broker-Live")
    c = MT5Connection(settings, mt5=real, sleep=lambda s: None, authority=authority)
    with pytest.raises(RealAccountBlockedError):
        c.connect()
    assert not real.connected, "debe cerrar el terminal al detectar cuenta real"


def test_blocked_error_is_assertion_error_and_survives_optimize():
    assert issubclass(RealAccountBlockedError, AssertionError)


def test_contest_account_also_blocked(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, account_mode=MockMT5.ACCOUNT_TRADE_MODE_CONTEST)
    with pytest.raises(RealAccountBlockedError):
        MT5Connection(settings, mt5=m, sleep=lambda s: None, authority=authority).connect()


def test_live_looking_server_blocked_even_if_mode_demo(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, account_mode=0, server="ICMarkets-Live07")
    with pytest.raises(RealAccountBlockedError):
        MT5Connection(settings, mt5=m, sleep=lambda s: None, authority=authority).connect()


def test_blocked_error_is_not_retried(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, account_mode=2)
    sleeps = []
    c = MT5Connection(settings, mt5=m, sleep=sleeps.append, authority=authority)
    with pytest.raises(RealAccountBlockedError):
        c.connect()
    assert sleeps == []


def test_account_switching_to_real_after_connect_blocks_orders(conn, mock_mt5, authority):
    order = _order(conn, authority)
    mock_mt5.account_mode = MockMT5.ACCOUNT_TRADE_MODE_REAL     # el usuario cambia de cuenta con el bot en marcha
    with pytest.raises(RealAccountBlockedError):
        conn.send_order(order)
    assert mock_mt5.order_log == [], "no debe llegar NINGUNA orden al broker"


def test_connect_retries_with_backoff(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, fail_initialize_times=2)
    sleeps = []
    c = MT5Connection(settings, mt5=m, sleep=sleeps.append, authority=authority)
    c.connect()
    assert len(sleeps) == 2 and c.is_alive()


def test_connect_gives_up_after_max_retries(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, fail_initialize_times=99)
    c = MT5Connection(settings, mt5=m, sleep=lambda s: None, authority=authority)
    with pytest.raises(MT5ConnectionError):
        c.connect()


def test_auto_reconnect_on_dropped_terminal(conn, mock_mt5):
    mock_mt5.simulate_disconnect()
    df = conn.get_rates("EURUSD", "M15", 50)
    assert len(df) == 50
    assert conn.reconnects == 1


def test_rates_shape_and_excludes_forming_bar(conn, mock_mt5):
    closed = conn.get_rates("EURUSD", "M15", 100)
    with_current = conn.get_rates("EURUSD", "M15", 100, include_current=True)
    assert list(closed.columns) == ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    assert closed.index.tz is not None and closed.index.is_monotonic_increasing
    assert with_current.index[-1] > closed.index[-1]
    assert (closed["high"] >= closed[["open", "close"]].max(axis=1)).all()


def test_ticks_history_chunked(conn):
    end = conn.server_time()
    ticks = conn.get_ticks("EURUSD", end - timedelta(hours=14), end, chunk_hours=6)
    assert {"bid", "ask"} <= set(ticks.columns)
    assert ticks.index.is_monotonic_increasing and (ticks["ask"] >= ticks["bid"]).all()


def test_symbol_spec_pnl_convention(conn):
    spec = conn.get_symbol_spec("EURUSD")
    # 1 lote EURUSD, +10 pips = +100 USD
    assert spec.pnl(1, 1.1000, 1.1010, 1.0) == pytest.approx(100.0, rel=1e-6)
    assert spec.risk_per_lot(1.1000, 1.0980) == pytest.approx(200.0, rel=1e-6)


def test_latency_is_tracked(conn):
    conn.get_rates("EURUSD", "M15", 10)
    assert conn.latency.p95() >= 0.0
    assert "rates:EURUSD" in conn.latency.summary()["ewma_ms"]


def test_unsigned_order_never_reaches_broker(conn, mock_mt5):
    q = conn.get_quote("EURUSD")
    raw = ApprovedOrder("x", "EURUSD", 1, 0.1, q["ask"], q["ask"] - 0.002, q["ask"] + 0.004, time.time())
    with pytest.raises(UnapprovedOrderError):
        conn.send_order(raw)
    assert mock_mt5.order_log == []


def test_tampered_order_rejected(conn, mock_mt5, authority):
    good = _order(conn, authority)
    bad = dataclasses.replace(good, volume=50.0)
    with pytest.raises(UnapprovedOrderError):
        conn.send_order(bad)
    assert mock_mt5.order_log == []


def test_expired_approval_rejected(settings, mock_mt5):
    now = [1000.0]
    auth = ApprovalAuthority(ttl_seconds=30, clock=lambda: now[0])
    c = MT5Connection(settings, mt5=mock_mt5, sleep=lambda s: None, authority=auth)
    c.connect()
    q = c.get_quote("EURUSD")
    o = auth.sign(ApprovedOrder("e", "EURUSD", 1, 0.1, q["ask"], q["ask"] - 0.002, q["ask"] + 0.004, now[0]))
    now[0] += 31
    with pytest.raises(UnapprovedOrderError):
        c.send_order(o)


def test_signed_order_open_and_close(conn, mock_mt5, authority):
    res = conn.send_order(_order(conn, authority))
    assert res.ok and res.ticket > 0
    pos = conn.get_positions()
    assert len(pos) == 1 and pos[0].direction == Direction.LONG and pos[0].volume == pytest.approx(0.10)
    conn.close_position(res.ticket)
    assert conn.get_positions() == []
    deals = conn.get_deals(conn.server_time() - timedelta(days=1), conn.server_time() + timedelta(minutes=5))
    assert set(deals["entry"]) == {0, 1}


def test_invalid_volume_rejected_by_broker(conn, authority):
    with pytest.raises(OrderRejectedError):
        conn.send_order(_order(conn, authority, volume=0.005))


def test_sl_triggered_updates_history(conn, mock_mt5, authority):
    balance_before = mock_mt5.balance
    conn.send_order(_order(conn, authority, dist=0.0003))   # SL/TP a 3 y 6 pips: se toca con certeza práctica
    mock_mt5.advance(3000)
    assert conn.get_positions() == [], "SL/TP debe haberse ejecutado"
    deals = conn.get_deals(conn.server_time() - timedelta(days=5), conn.server_time() + timedelta(minutes=5))
    outs = deals[deals["entry"] == 1]
    assert len(outs) == 1 and outs.iloc[0]["comment"] in {"sl", "tp"}
    assert abs(outs.iloc[0]["profit"]) == pytest.approx(3.0, abs=0.5) or abs(outs.iloc[0]["profit"]) == pytest.approx(6.0, abs=0.5)
    assert mock_mt5.balance != balance_before


def test_ipc_timeout_error_carries_actionable_hint(settings, authority):
    m = MockMT5(symbols=["EURUSD"], n_minutes=2000, fail_initialize_times=99)     # el mock falla con código -10005
    with pytest.raises(MT5ConnectionError) as exc:
        MT5Connection(settings, mt5=m, sleep=lambda s: None, authority=authority).connect()
    assert "IPC timeout" in str(exc.value) and "MT5_PATH" in str(exc.value)
