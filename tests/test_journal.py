"""Tests de la memoria: el sistema debe APRENDER de sus errores (p. ej. perder siempre en NFP)."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.types import ClosedTrade, Direction
from memory.embeddings import HashingEmbedder
from memory.trade_journal import (MIN_BUCKET_N, TradeJournal, bucket_weight, context_tags, describe_context,
                                  stats_of)
from memory.vector_store import NumpyVectorStore

T0 = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)


def make_trade(i, symbol="EURUSD", r=1.0, event=None, session="london", regime="trend", direction=Direction.LONG,
               risk=100.0):
    ctx = {"session": session, "regime": regime, "event_tags": [event] if event else [], "sentiment": 30,
           "structure": 1, "wyckoff": "accumulation"}
    opened = T0 + timedelta(hours=i * 5)
    return ClosedTrade(ticket=1000 + i, symbol=symbol, direction=direction, volume=0.1, open_time=opened,
                       close_time=opened + timedelta(hours=2), price_open=1.1, price_close=1.1, sl=1.09, tp=1.12,
                       profit=r * risk, risk_amount=risk, exit_reason="tp" if r > 0 else "sl", context=ctx)


@pytest.fixture()
def journal(tmp_path):
    store = NumpyVectorStore("trades_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path)
    return TradeJournal(tmp_path / "j.sqlite", store)


def test_bucket_weight_math():
    w, info = bucket_weight([1, -1, 2, -1, 1, 1, -1, 2])          # positivo pero ruidoso
    assert 1.0 < w <= 1.5 and info["n"] == 8
    m = 4 / 8
    assert w == pytest.approx(1 + 8 / 16 * m)                       # 1 + n/(n+8) * m
    assert bucket_weight([1, 1])[0] == 1.0                          # muestra insuficiente => sin efecto
    w_neg, info = bucket_weight([-1, -1, -1, -1, -1, -1, 2, -1, -1])
    assert w_neg <= 0.4 and info["significant"] == 1.0
    assert bucket_weight([-5.0] * 40)[0] == 0.25                    # suelo del peso
    assert bucket_weight([9.0] * 40)[0] == 1.5                      # techo del peso


def test_stats_of():
    s = stats_of([2.0, -1.0, 2.0, -1.0, -1.0])
    assert s["n"] == 5 and s["wins"] == 2 and s["win_rate"] == pytest.approx(0.4)
    assert s["avg_r"] == pytest.approx(0.2) and s["profit_factor"] == pytest.approx(4 / 3)
    assert stats_of([])["n"] == 0 and stats_of([1.0])["profit_factor"] == float("inf")


def test_context_description_has_no_outcome():
    t = make_trade(0, r=-1.0, event="NFP")
    text = describe_context(t.symbol, t.direction, t.context, t.open_time)
    assert "event:NFP" in text and "session:london" in text
    assert "loss" not in text.lower() and "win" not in text.lower()
    assert "event:NFP" in context_tags("EURUSD", 1, t.context, t.open_time)


def test_record_roundtrip_and_upsert(journal, tmp_path):
    t = make_trade(0, r=-1.0, event="NFP")
    journal.record(t)
    journal.record(t)                                               # idempotente por ticket
    assert journal.count() == 1 and journal.has(1000) and not journal.has(5)
    back = journal.trades()[0]
    assert back.r_multiple == pytest.approx(-1.0) and back.context["event_tags"] == ["NFP"]
    assert back.direction == Direction.LONG and back.close_time == t.close_time
    j2 = TradeJournal(tmp_path / "j.sqlite")                        # persiste entre reinicios
    assert j2.count() == 1


def test_recent_r_order(journal):
    for i, r in enumerate([1, -1, 2, -1, -1]):
        journal.record(make_trade(i, r=r))
    assert journal.recent_r(3) == pytest.approx([2, -1, -1])


def _seed_nfp_history(journal):
    rng_r = [2, -1, 2, 2, -1, 2, -1, 2, -1, 2, -1, 2]              # 7/12 ganadas, muy positivo
    for i, r in enumerate(rng_r):
        journal.record(make_trade(i, r=r))
    nfp = [2, -1, -1, -1, -1, -1, -1, -1, -1]
    for j, r in enumerate(nfp):
        journal.record(make_trade(100 + j, r=r, event="NFP"))


def test_learns_to_reduce_risk_on_nfp(journal):
    _seed_nfp_history(journal)
    when = T0 + timedelta(days=60)
    nfp_ctx = {"session": "overlap", "regime": "trend", "event_tags": ["NFP"]}
    calm_ctx = {"session": "london", "regime": "trend", "event_tags": []}
    a_nfp = journal.advise("EURUSD", Direction.LONG, nfp_ctx, when)
    a_calm = journal.advise("EURUSD", Direction.LONG, calm_ctx, when)
    assert a_nfp.risk_weight <= 0.4, a_nfp.buckets
    assert a_calm.risk_weight >= 1.0
    assert any("NFP" in l and "reduzco" in l for l in a_nfp.lessons)
    assert a_nfp.risk_weight < a_calm.risk_weight
    # otro símbolo sin historial propio hereda la cautela global sobre NFP
    other = journal.advise("GBPUSD", Direction.LONG, nfp_ctx, when)
    assert other.risk_weight < 1.0 and any("*/event:NFP" in k for k in other.buckets)


def test_no_history_means_neutral(journal):
    a = journal.advise("EURUSD", Direction.LONG, {"session": "asia"}, T0)
    assert a.risk_weight == 1.0 and a.hist_win_rate is None and a.lessons == []


def test_small_samples_do_not_move_weights(journal):
    for i in range(MIN_BUCKET_N - 1):
        journal.record(make_trade(i, r=-1.0, event="CPI"))
    assert journal.advise("EURUSD", Direction.LONG, {"event_tags": ["CPI"]}, T0).risk_weight == 1.0


def test_lessons_text_and_ordering(journal):
    _seed_nfp_history(journal)
    ls = journal.lessons()
    assert ls and "NFP" in ls[0] and "Pierdo" in ls[0]
    assert all("ALL" not in l for l in ls)


def test_rag_similar_trades_win_rate(journal):
    _seed_nfp_history(journal)
    ctx = {"session": "london", "regime": "trend", "event_tags": ["NFP"], "sentiment": 30, "structure": 1, "wyckoff": "accumulation"}
    a = journal.advise("EURUSD", Direction.LONG, ctx, T0 + timedelta(days=90))
    assert a.n_similar >= 8 and a.hist_win_rate is not None
    calm = journal.advise("EURUSD", Direction.LONG, {"session": "london", "regime": "trend", "event_tags": [],
                                                     "sentiment": 30, "structure": 1, "wyckoff": "accumulation"}, T0)
    assert a.hist_win_rate < calm.hist_win_rate


def test_stats_by_tag(journal):
    _seed_nfp_history(journal)
    assert journal.stats("EURUSD")["n"] == 21
    nfp = journal.stats("EURUSD", "event:NFP")
    assert nfp["n"] == 9 and nfp["wins"] == 1 and nfp["avg_r"] < 0
