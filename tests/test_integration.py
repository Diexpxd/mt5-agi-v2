"""Integración extremo a extremo sobre el mercado simulado: agentes + riesgo + broker + memoria + interfaz + bot."""
import asyncio
import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from agents.economic_calendar import EconEvent, EconomicCalendar
from agents.interface_agent import Reply, classify, resolve_period, resolve_symbol
from agents.orchestrator import OrchestratorConfig
from bot.telegram_bot import TradingBot, build_async_bot, deliver, split_text
from config.settings import Settings
from core.exceptions import RealAccountBlockedError
from core.mock_mt5 import MockMT5
from core.types import Direction
from system import build_system

ROOT = Path(__file__).resolve().parent.parent
SYMS = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")


def make_settings(tmp_path, mode="paper", symbols=SYMS, **kw):
    (tmp_path / "data").mkdir(exist_ok=True)
    shutil.copy(ROOT / "data" / "sample_news.json", tmp_path / "data" / "sample_news.json")
    return Settings(symbols=symbols, mt5_max_retries=2, mt5_retry_base_delay=0.0, execution_mode=mode,
                    data_dir=tmp_path / "data", state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
                    chroma_dir=tmp_path / "state" / "chroma", model_dir=tmp_path / "state" / "models", **kw)


def make_system(tmp_path, mode="paper", symbols=SYMS, seed=5, n_minutes=60_000, **kw):
    s = make_settings(tmp_path, mode, symbols, **kw)
    mock = MockMT5(symbols=list(symbols), seed=seed, n_minutes=n_minutes)
    sysm = build_system(s, mt5=mock, mode=mode, vector_backend="numpy", orch_config=OrchestratorConfig(async_news=False))
    sysm.conn.connect()
    return sysm, mock


def run_cycles(sysm, mock, n, minutes=15):
    reps = []
    for _ in range(n):
        mock.advance(minutes)
        reps.append(sysm.orchestrator.run_cycle())
    return reps


@pytest.fixture(scope="module")
def paper_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paper")
    sysm, mock = make_system(tmp, "paper", seed=5, n_minutes=90_000)
    reps = run_cycles(sysm, mock, 260)
    return sysm, mock, reps


def test_paper_run_has_no_errors_and_trades_happen(paper_run):
    sysm, mock, reps = paper_run
    assert all(not r.errors for r in reps), [e for r in reps for e in r.errors][:3]
    opened = sum(len(r.opened) for r in reps)
    closed = sum(len(r.closed) for r in reps)
    assert opened >= 1 and closed >= 1
    assert sysm.journal.count() == closed                      # cada cierre reconciliado queda en la memoria
    assert mock.order_log == [], "el modo paper NUNCA envía órdenes al broker"


def test_paper_run_respects_risk_invariants(paper_run):
    sysm, _, reps = paper_run
    lim = sysm.settings.risk
    for t in sysm.journal.trades():
        assert t.risk_amount > 0
        assert t.risk_amount <= lim.max_risk_per_trade * 10_000 * 1.6 + 1, "riesgo por operación acotado (equity puede haber crecido)"
        assert t.context.get("session") and "memory" in t.context
    assert max(len(sysm.broker.positions()), 0) <= lim.max_positions
    assert sysm.orchestrator.tracker.peak_equity >= sysm.orchestrator.status()["equity"] - 1e-6
    # el equity del broker es consistente con el balance más el flotante
    acc = sysm.broker.account()
    assert acc.equity == pytest.approx(acc.balance + sum(p.profit for p in sysm.broker.positions()))


def test_journal_pnl_matches_balance_change(paper_run):
    sysm, _, _ = paper_run
    total = sum(t.profit for t in sysm.journal.trades())
    entry_comm = sum(7.0 * p.volume / 2 for p in sysm.broker.positions())        # comisión de entrada ya cobrada de posiciones abiertas
    assert sysm.broker.balance == pytest.approx(10_000 + total - entry_comm, abs=0.05)


def test_no_duplicate_signals_on_same_bar(tmp_path):
    sysm, mock = make_system(tmp_path, symbols=("EURUSD",))
    mock.advance(15)
    r1 = sysm.orchestrator.run_cycle()
    r2 = sysm.orchestrator.run_cycle()                            # sin nueva vela: no se re-analiza
    assert r1.analyzed == ["EURUSD"] and r2.analyzed == []


def test_pause_blocks_new_trades_and_resume_restores(tmp_path):
    sysm, mock = make_system(tmp_path, seed=5, n_minutes=90_000)
    sysm.orchestrator.pause()
    reps = run_cycles(sysm, mock, 40)
    assert all(not r.opened and "sistema en pausa" in r.notes for r in reps)
    sysm.orchestrator.resume()
    assert not sysm.orchestrator.paused


def test_events_are_emitted_and_alert_formatted(paper_run):
    sysm, mock, reps = paper_run
    got = []
    sysm.orchestrator.subscribe(lambda k, p: got.append((k, p)))
    sysm.orchestrator._emit("trade_opened", symbol="EURUSD", direction="BUY", lots=0.5, price=1.1, sl=1.09, tp=1.12,
                            risk_amount=100.0, risk_pct=0.01, rationale=["OB"], lessons=["cuidado NFP"], win_prob=0.4)
    text = sysm.interface.format_event(*got[0])
    assert "ABIERTA EURUSD BUY" in text and "cuidado NFP" in text


def test_veto_when_sentiment_strongly_opposes(tmp_path):
    from core.types import Signal
    from agents.fundamental_agent import SentimentResult
    sysm, mock = make_system(tmp_path, symbols=("EURUSD",), seed=5, n_minutes=90_000)
    o = sysm.orchestrator
    fake_signal = lambda *a, **k: Signal("EURUSD", Direction.LONG, 0.9, entry=1.1, sl=1.09, tp=1.12, meta={"atr": 0.001})
    o.technical.analyze = fake_signal
    o.fundamental.sentiment = lambda sym, now=None: SentimentResult(sym, -80.0, 0.9, ["USD hawkish"])
    mock.advance(15)
    rep = o.run_cycle()
    assert rep.vetoed == ["EURUSD"] and not rep.opened
    assert "veto fundamental" in list(o.decision_log)[-1]["summary"]


def test_risk_rejection_reaches_log_with_reasons(tmp_path):
    from core.types import Signal
    from agents.fundamental_agent import SentimentResult
    sysm, mock = make_system(tmp_path, symbols=("EURUSD",), seed=5, n_minutes=90_000)
    o = sysm.orchestrator
    # SL a 0.4 pips: el spread (>15 % del SL) debe hacer rechazar la operación
    o.technical.analyze = lambda *a, **k: Signal("EURUSD", Direction.LONG, 0.9, entry=1.1, sl=1.1 - 0.00004, tp=1.1 + 0.00008, meta={"atr": 0.001})
    o.fundamental.sentiment = lambda sym, now=None: SentimentResult(sym, 10.0, 0.5)
    mock.advance(15)
    rep = o.run_cycle()
    assert rep.rejected and "spread" in rep.rejected[0] and not rep.opened


def _force_signal(o, sentiment=10.0):
    from core.types import Signal
    from agents.fundamental_agent import SentimentResult
    o.technical.analyze = lambda *a, **k: Signal("EURUSD", Direction.LONG, 0.9, entry=1.1, sl=1.09, tp=1.12, meta={"atr": 0.001})
    o.fundamental.sentiment = lambda sym, now=None: SentimentResult(sym, sentiment, 0.5)


def test_news_blackout_blocks_trades_in_orchestrator(tmp_path):
    (tmp_path / "c").mkdir(); (tmp_path / "t").mkdir()
    control, mock_c = make_system(tmp_path / "c", symbols=("EURUSD",), seed=5, n_minutes=90_000)
    _force_signal(control.orchestrator)
    mock_c.advance(15)
    rc = control.orchestrator.run_cycle()
    assert rc.opened and not rc.rejected, rc.rejected                 # el control demuestra que la señal es válida

    sysm, mock = make_system(tmp_path / "t", symbols=("EURUSD",), seed=5, n_minutes=90_000)
    _force_signal(sysm.orchestrator)
    mock.advance(15)
    now = sysm.orchestrator.now()
    sysm.orchestrator.calendar.merge([EconEvent(now + timedelta(minutes=15), "USD", "high", "Non-Farm Employment Change")])
    rep = sysm.orchestrator.run_cycle()
    assert not rep.opened and rep.rejected and "noticias" in rep.rejected[0] and "Non-Farm" in rep.rejected[0]


def test_state_persists_across_restart(tmp_path):
    sysm, mock = make_system(tmp_path, seed=5, n_minutes=90_000)
    run_cycles(sysm, mock, 200)
    bal, n_jr = sysm.broker.balance, sysm.journal.count()
    sysm2 = build_system(sysm.settings, mt5=mock, mode="paper", vector_backend="numpy")
    assert sysm2.broker.balance == pytest.approx(bal) and sysm2.journal.count() == n_jr
    assert len(sysm2.orchestrator.equity_history) > 10


def test_demo_mode_sends_signed_orders_to_mt5_and_reconciles(tmp_path):
    sysm, mock = make_system(tmp_path, "demo", seed=5, n_minutes=90_000)
    reps = run_cycles(sysm, mock, 260)
    assert all(not r.errors for r in reps)
    assert len(mock.order_log) >= 1
    assert all(o["magic"] == sysm.settings.magic_number and o["sl"] > 0 and o["tp"] > 0 for o in mock.order_log if "position" not in o)
    assert sysm.journal.count() >= 1
    open_comm = sum(7.0 * p.volume / 2 for p in mock._positions.values())
    assert mock.balance == pytest.approx(10_000 + sum(t.profit for t in sysm.journal.trades()) - open_comm, abs=0.05)


def test_real_account_halts_everything(tmp_path):
    s = make_settings(tmp_path, "demo")
    real = MockMT5(symbols=list(SYMS), n_minutes=4000, account_mode=2, server="Broker-Live")
    sysm = build_system(s, mt5=real, mode="demo", vector_backend="numpy")
    with pytest.raises(RealAccountBlockedError):
        sysm.start()
    assert real.order_log == []


def test_close_all_and_reconcile(tmp_path):
    sysm, mock = make_system(tmp_path, seed=5, n_minutes=90_000)
    for _ in range(300):
        mock.advance(15)
        sysm.orchestrator.run_cycle()
        if sysm.broker.positions():
            break
    assert sysm.broker.positions(), "el escenario debe abrir alguna posición"
    n = sysm.orchestrator.close_all("test")
    assert n >= 1 and sysm.broker.positions() == []
    assert any(t.exit_reason == "test" for t in sysm.journal.trades())


def test_server_utc_offset_is_applied_to_data_and_ranges(tmp_path):
    s = make_settings(tmp_path, mt5_utc_offset_hours=3.0)
    mock = MockMT5(symbols=["EURUSD"], n_minutes=6000)
    from core.mt5_connection import MT5Connection
    conn = MT5Connection(s, mt5=mock, sleep=lambda x: None)
    conn.connect()
    raw_last = mock.copy_rates_from_pos("EURUSD", mock.TIMEFRAME_M15, 1, 1)[0]["time"]
    df = conn.get_rates("EURUSD", "M15", 5)
    assert int(df.index[-1].timestamp()) == raw_last - 3 * 3600          # UTC = hora servidor - 3 h
    start, end = df.index[0], df.index[-1]
    rng = conn.get_rates_range("EURUSD", "M15", start.to_pydatetime(), end.to_pydatetime())
    assert list(rng.index) == list(df.index)                             # ida y vuelta consistente
    assert conn.server_time().timestamp() == pytest.approx(mock._now_ts() - 3 * 3600)


@pytest.mark.parametrize("text,intent", [
    ("¿Cuál es nuestro PnL hoy?", "pnl"), ("cuánto llevamos ganado", "pnl"), ("Resume el mercado del Oro", "market"),
    ("dame un gráfico del euro", "chart"), ("¿qué posiciones tenemos abiertas?", "positions"),
    ("¿por qué no has operado?", "decisions"), ("qué has aprendido de tus errores", "lessons"),
    ("¿cómo está el riesgo?", "risk"), ("pausa el bot", "pause"), ("cierra todo", "close_all"),
    ("show me the sentiment for bitcoin", "sentiment"), ("próximos eventos del calendario", "calendar"),
    ("ayuda", "help"), ("blablabla", "unknown"),
])
def test_intent_classification(text, intent):
    assert classify(text) == intent


def test_symbol_and_period_resolution():
    known = SYMS
    assert resolve_symbol("Resume el mercado del Oro", known) == "XAUUSD"
    assert resolve_symbol("gráfico EUR/USD", known) == "EURUSD" and resolve_symbol("cómo va el yen", known) == "USDJPY"
    assert resolve_symbol("qué tal el eurusd", known) == "EURUSD" and resolve_symbol("nada", known) is None
    assert resolve_period("pnl de esta semana") == "week" and resolve_period("este mes") == "month"
    assert resolve_period("pnl") == "today" and resolve_period("histórico") == "all"


def test_interface_answers_with_real_data(paper_run):
    sysm, _, _ = paper_run
    ia = sysm.interface
    pnl = ia.handle("¿Cuál es nuestro PnL hoy?")
    assert "PnL HOY" in pnl.text and "Equity actual" in pnl.text
    market = ia.handle("Resume el mercado del Oro")
    assert "XAUUSD" in market.text and "Sentimiento" in market.text
    assert market.images and market.images[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert market.buttons
    assert "SIN SETUP" in ia.handle("/signal EURUSD").text or "EURUSD" in ia.handle("/signal EURUSD").text
    assert "Riesgo abierto" in ia.handle("/risk").text and "Estado del sistema" in ia.handle("/status").text
    assert ia.handle("/equity").images and ia.handle("gráfico del euro").images
    assert "Memoria" in ia.handle("/lessons").text and ia.handle("/trades").text.startswith("🧾")
    week = ia.handle("pnl de la semana")
    assert week.images and "últimos 7 días" in week.text
    assert "Panorama" in ia.handle("/market").text and ia.handle("/market").images
    assert "PnL" in ia.handle_callback("cmd:pnl:week").text
    assert "Elige un activo" in ia.handle("/chart").text


def test_admin_actions_require_admin_and_confirmation(paper_run):
    sysm, _, _ = paper_run
    ia, o = sysm.interface, sysm.orchestrator
    assert "Solo un administrador" in ia.handle("pausa el bot", is_admin=False).text and not o.paused
    assert "Solo un administrador" in ia.handle_callback("confirm:close_all", is_admin=False).text
    r = ia.handle("cierra todo", is_admin=True)
    assert r.buttons and r.buttons[0][0][1] == "confirm:close_all"                  # pide confirmación
    ia.handle("pausa el bot", is_admin=True)
    assert o.paused
    ia.handle("/resume", is_admin=True)
    assert not o.paused
    assert "Cancelado" in ia.handle_callback("cancel", is_admin=True).text


def test_interface_never_raises(paper_run):
    sysm, _, _ = paper_run
    sysm.orchestrator.conn.get_rates = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    r = sysm.interface.handle("resume el mercado del oro")
    assert "No pude completar" in r.text


class FakeSender:
    def __init__(self):
        self.texts, self.photos = [], []

    async def send_text(self, chat_id, text, buttons=None):
        self.texts.append((chat_id, text, buttons))

    async def send_photo(self, chat_id, png, caption=""):
        self.photos.append((chat_id, png, caption))


def test_bot_access_control(paper_run, tmp_path):
    sysm, _, _ = paper_run
    s = make_settings(tmp_path, telegram_admin_ids=(111,))
    bot = TradingBot(s, sysm.interface, allowed_ids=[222])
    assert bot.role(111) == "admin" and bot.role(222) == "viewer" and bot.role(999) == "none"
    assert "Tu id de Telegram es 999" in bot.on_text(999, "¿PnL hoy?").text
    assert "PnL HOY" in bot.on_text(222, "¿PnL hoy?").text
    assert "Solo un administrador" in bot.on_text(222, "/pause").text        # el viewer no puede pausar
    assert "Acceso no autorizado" in bot.on_callback(999, "cmd:pnl:today").text


def test_bot_delivery_photos_text_and_splitting():
    async def go():
        fs = FakeSender()
        await deliver(fs, 1, Reply("hola", [b"png1"]))
        assert fs.photos == [(1, b"png1", "hola")] and fs.texts == []       # texto corto = pie de foto
        fs2 = FakeSender()
        await deliver(fs2, 1, Reply("x" * 9000, [b"a", b"b"], buttons=[[("ok", "cb")]]))
        assert len(fs2.photos) == 2 and len(fs2.texts) == 3 and fs2.texts[-1][2] == [[("ok", "cb")]]
        assert all(len(t[1]) <= 4000 for t in fs2.texts)
    asyncio.run(go())
    assert split_text("a\n" * 3000, 100)[0].count("\n") <= 100 and "".join(split_text("abc", 2)) == "abc"


def test_async_telebot_builds_with_handlers(paper_run, tmp_path):
    sysm, _, _ = paper_run
    s = make_settings(tmp_path, telegram_token="123456:ABC-DEF_fake", telegram_admin_ids=(1,))
    bot, sender = build_async_bot(s, TradingBot(s, sysm.interface))
    assert len(bot.message_handlers) >= 2 and len(bot.callback_query_handlers) == 1
    markup = sender._markup([[("A", "a"), ("B", "b")]])
    assert [b.callback_data for b in markup.keyboard[0]] == ["a", "b"]
