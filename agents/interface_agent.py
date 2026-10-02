"""Agente de Interfaz: convierte lenguaje natural (ES/EN) y comandos en respuestas con datos REALES del sistema."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from bot import charts
from config.settings import Settings
from core.llm import LLMClient
from core.types import Direction

from . import smc
from .risk_agent import pair_legs, signed_notional

log = logging.getLogger(__name__)

Button = Tuple[str, str]                                   # (texto, callback_data)


@dataclass
class Reply:
    text: str
    images: List[bytes] = field(default_factory=list)
    buttons: Optional[List[List[Button]]] = None


ALIASES = {
    "oro": "XAUUSD", "gold": "XAUUSD", "xau": "XAUUSD", "euro": "EURUSD", "libra": "GBPUSD", "cable": "GBPUSD",
    "sterling": "GBPUSD", "yen": "USDJPY", "bitcoin": "BTCUSD", "btc": "BTCUSD", "ethereum": "ETHUSD", "eth": "ETHUSD",
    "aussie": "AUDUSD", "loonie": "USDCAD", "petroleo": "USOIL", "oil": "USOIL",
}

INTENTS: List[Tuple[str, str]] = [
    ("close_all", r"cerrar (todo|todas)|close all|liquidar|cierra todo"),
    ("pause", r"\b(pausa|pausar|pausa el|detener|detente|para el bot|stop trading)\b"),
    ("resume", r"\b(reanudar|reanuda|continuar|reactivar)\b|^resume$|\bresume (trading|el bot|el sistema|el trading|operaciones|operando)\b"),
    ("help", r"\b(ayuda|help|que puedes hacer|comandos|menu)\b|^/start"),
    ("decisions", r"por que no|why not|rechaz|decisiones|no has (operado|entrado)"),
    ("lessons", r"aprend|leccion|memoria|errores|lesson|que has aprendido"),
    ("calendar", r"calendario|agenda|eventos|proximas noticias|nfp|fomc|cpi"),
    ("risk", r"\briesgo\b|\brisk\b|\bvar\b|drawdown|exposicion|apalancamiento"),
    ("positions", r"posicion|abiertas|open positions|operaciones abiertas|que tenemos abierto"),
    ("trades", r"ultimas operaciones|historial|trades|operaciones cerradas|ultimos trades"),
    ("equity_chart", r"curva|equity|evolucion del capital|capital"),
    ("chart", r"grafico|grafica|chart|plot|velas|candles"),
    ("sentiment", r"sentimiento|sentiment|noticias de|que dicen las noticias"),
    ("signal", r"\bsenal\b|\bsignal\b|comprar|vender|entrar|entrarias|opinas|deberia (comprar|vender)"),
    ("market", r"resum|analiza|analisis|como esta|que pasa con|mercado|panorama|market"),
    ("pnl", r"pnl|ganancia|perdida|beneficio|profit|resultado|cuanto llevamos|como vamos|cuanto hemos"),
    ("status", r"estado|status|conectad|latencia|como estas|funcionando"),
]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c)).replace("¿", "").replace("?", "").strip()


def resolve_symbol(text: str, known: Tuple[str, ...]) -> Optional[str]:
    """Extrae el símbolo mencionado: alias (oro, bitcoin...), ticker exacto o forma ``EUR/USD``."""
    t = normalize(text)
    compact = re.sub(r"[\s/_\-]", "", t)
    for sym in known:
        if sym.lower() in compact:
            return sym
    words = set(re.findall(r"[a-z0-9]+", t))
    for alias, sym in ALIASES.items():
        if alias in words and (sym in known or not known):
            return sym
    m = re.search(r"\b([a-z]{3})\s*/\s*([a-z]{3})\b", t)
    if m:
        cand = (m.group(1) + m.group(2)).upper()
        return cand if cand in known else None
    return None


def resolve_period(text: str) -> str:
    t = normalize(text)
    if re.search(r"\bsemana|week|7 dias", t):
        return "week"
    if re.search(r"\bmes\b|month|30 dias", t):
        return "month"
    if re.search(r"\btodo\b|total|historico|all time|desde el principio", t):
        return "all"
    return "today"


def classify(text: str) -> str:
    t = normalize(text)
    for name, pat in INTENTS:
        if re.search(pat, t):
            return name
    return "unknown"


def _sc(x: float) -> str:
    r = round(x)
    return "0" if r == 0 else f"{r:+d}"


def _money(x: float) -> str:
    return f"{'+' if x > 0 else ''}{x:,.2f}"


class InterfaceAgent:
    """Traduce consultas a acciones/lecturas sobre el orquestador."""

    name = "interface"

    def __init__(self, settings: Settings, orchestrator, llm: LLMClient) -> None:
        self.settings = settings
        self.o = orchestrator
        self.llm = llm

    def handle(self, text: str, is_admin: bool = False) -> Reply:
        text = text.strip()
        if text.startswith("/"):
            parts = text[1:].split()
            cmd = parts[0].split("@")[0].lower() if parts else ""
            return self.handle_command(cmd, parts[1:], is_admin)
        intent = classify(text)
        sym = resolve_symbol(text, self.settings.symbols)
        if intent == "unknown" and sym:
            intent = "market"                                   # "¿oro?" => resumen del oro
        return self._dispatch(intent, sym, resolve_period(text), text, is_admin)

    def handle_command(self, cmd: str, args: List[str], is_admin: bool = False) -> Reply:
        table = {"start": "help", "help": "help", "menu": "help", "pnl": "pnl", "positions": "positions", "posiciones": "positions",
                 "status": "status", "estado": "status", "market": "market", "mercado": "market", "chart": "chart",
                 "grafico": "chart", "sentiment": "sentiment", "sentimiento": "sentiment", "signal": "signal", "senal": "signal",
                 "risk": "risk", "riesgo": "risk", "lessons": "lessons", "lecciones": "lessons", "calendar": "calendar",
                 "calendario": "calendar", "trades": "trades", "why": "decisions", "porque": "decisions",
                 "equity": "equity_chart", "pause": "pause", "pausa": "pause", "resume": "resume", "reanudar": "resume",
                 "closeall": "close_all", "cerrartodo": "close_all"}
        intent = table.get(cmd)
        if intent is None:
            return Reply(f"Comando /{cmd} no reconocido.\n\n" + self._help_text(), buttons=self.main_menu(is_admin))
        arg_text = " ".join(args)
        sym = resolve_symbol(arg_text, self.settings.symbols) if args else None
        return self._dispatch(intent, sym, resolve_period(arg_text), arg_text, is_admin)

    def handle_callback(self, data: str, is_admin: bool = False) -> Reply:
        kind, _, rest = data.partition(":")
        if kind == "cancel":
            return Reply("Cancelado.", buttons=self.main_menu(is_admin))
        if kind == "confirm" and rest == "close_all":
            if not is_admin:
                return Reply("⛔ Solo un administrador puede cerrar todas las posiciones.")
            n = self.o.close_all("telegram")
            return Reply(f"✅ Cerradas {n} posiciones.", buttons=self.main_menu(is_admin))
        if kind == "cmd":
            intent, _, arg = rest.partition(":")
            sym = arg if arg in self.settings.symbols else None
            period = arg if arg in ("today", "week", "month", "all") else "today"
            return self._dispatch(intent, sym, period, arg, is_admin)
        return Reply("Botón no reconocido.")

    def main_menu(self, is_admin: bool = False) -> List[List[Button]]:
        rows = [
            [("💰 PnL hoy", "cmd:pnl:today"), ("📂 Posiciones", "cmd:positions")],
            [("📊 Mercado", "cmd:pick_market"), ("🧠 Sentimiento", "cmd:sentiment")],
            [("🛡 Riesgo", "cmd:risk"), ("📅 Calendario", "cmd:calendar")],
            [("📈 Equity", "cmd:equity_chart"), ("🧾 Trades", "cmd:trades")],
            [("🎓 Lecciones", "cmd:lessons"), ("🔎 ¿Por qué no?", "cmd:decisions")],
            [("ℹ️ Estado", "cmd:status")],
        ]
        if is_admin:
            rows.append([("⏸ Pausar", "cmd:pause"), ("▶️ Reanudar", "cmd:resume"), ("🛑 Cerrar todo", "cmd:close_all")])
        return rows

    def _symbol_menu(self, intent: str) -> Reply:
        syms = list(self.settings.symbols)
        rows = [[(s, f"cmd:{intent}:{s}") for s in syms[i:i + 3]] for i in range(0, len(syms), 3)]
        return Reply("Elige un activo:", buttons=rows)

    def _help_text(self) -> str:
        return ("🤖 MT5 AGI V2 — pregúntame en lenguaje natural o usa los comandos:\n"
                "• ¿Cuál es nuestro PnL hoy? · /pnl [today|week|month|all]\n"
                "• Resume el mercado del Oro · /market XAUUSD\n"
                "• Gráfico del euro · /chart EURUSD\n"
                "• Sentimiento de las noticias · /sentiment [símbolo]\n"
                "• ¿Comprarías el yen? · /signal USDJPY\n"
                "• Posiciones · /positions   Riesgo · /risk   Estado · /status\n"
                "• Calendario · /calendar   Lecciones aprendidas · /lessons\n"
                "• ¿Por qué no has operado? · /why\n"
                "Admin: /pause · /resume · /closeall")

    def _dispatch(self, intent: str, sym: Optional[str], period: str, text: str, is_admin: bool) -> Reply:
        try:
            if intent in ("market", "chart", "signal") and sym is None and text and intent != "market":
                return self._symbol_menu(intent)
            if intent == "pick_market":
                return self._symbol_menu("market")
            h = getattr(self, f"_h_{intent}", None)
            if h is None:
                return self._unknown(text)
            return h(sym=sym, period=period, is_admin=is_admin, text=text)
        except Exception as exc:                               # el bot nunca debe quedarse mudo
            log.exception("interface: error en %s", intent)
            return Reply(f"⚠️ No pude completar la consulta ({type(exc).__name__}: {exc}).")

    def _unknown(self, text: str) -> Reply:
        if not self.llm.is_mock and text:
            facts = json.dumps(self._facts(), default=str, ensure_ascii=False)
            ans = self.llm.generate(f"Estado del sistema de trading (JSON): {facts}\n\nPregunta del usuario: {text}\n"
                                    "Responde en español, en ≤5 líneas, usando solo estos datos. Si no puedes, dilo.",
                                    system="Eres el asistente de un fondo cuantitativo. No inventes cifras ni des consejos de inversión.")
            if ans.strip():
                return Reply(ans.strip(), buttons=self.main_menu())
        return Reply("No he entendido la consulta. Prueba con:\n" + self._help_text(), buttons=self.main_menu())

    def _facts(self) -> Dict[str, Any]:
        st = self.o.status()
        return {"modo": st["mode"], "equity": round(st["equity"], 2), "pnl_dia": round(st["daily_pnl"], 2),
                "posiciones": len(st["positions"]), "pausado": st["paused"]}

    def _h_help(self, is_admin: bool, **_) -> Reply:
        return Reply(self._help_text(), buttons=self.main_menu(is_admin))

    def _h_status(self, **_) -> Reply:
        st = self.o.status()
        lat = st["latency"]
        return Reply(
            f"ℹ️ Estado del sistema\n• Modo: {st['mode'].upper()} ({st['backend']})  {'⏸ PAUSADO' if st['paused'] else '▶️ activo'}\n"
            f"• Conexión MT5: {'OK' if st['connected'] else 'CAÍDA'} · reconexiones {st['reconnects']}\n"
            f"• Latencia: ping terminal {lat['ping_terminal_ms']} ms · p95 {lat['p95_ms']} ms\n"
            f"• Equity ${st['equity']:,.2f} · balance ${st['balance']:,.2f}\n• PnL del día {_money(st['daily_pnl'])} · drawdown {st['drawdown'] * 100:.2f}%\n"
            f"• Posiciones abiertas: {len(st['positions'])}\n• Último ciclo: {st['last_cycle'] or '—'}\n"
            f"• Offset hora servidor: {st['utc_offset_h']:+.1f} h · LLM: {self.llm.name}")

    def _h_positions(self, **_) -> Reply:
        pos = self.o.broker.positions()
        if not pos:
            return Reply("📂 No hay posiciones abiertas.")
        lines = ["📂 Posiciones abiertas:"]
        for p in pos:
            lines.append(f"• {p.symbol} {p.direction.label} {p.volume:.2f} lotes @ {p.price_open:g} → {p.price_current:g} | "
                         f"PnL {_money(p.profit)} | SL {p.sl:g} TP {p.tp:g}")
        lines.append(f"Flotante total: {_money(sum(p.profit for p in pos))}")
        return Reply("\n".join(lines))

    def _h_pnl(self, period: str, **_) -> Reply:
        st = self.o.status()
        now = self.o.now()
        trades = self.o.journal.trades()
        days = {"today": 1, "week": 7, "month": 30, "all": 36500}[period]
        cutoff = (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)) if period != "all" else None
        sel = [t for t in trades if cutoff is None or t.close_time >= cutoff]
        realized = sum(t.profit for t in sel)
        floating = sum(p.profit for p in st["positions"])
        wins = sum(1 for t in sel if t.profit > 0)
        name = {"today": "HOY", "week": "últimos 7 días", "month": "últimos 30 días", "all": "histórico"}[period]
        lines = [f"💰 PnL {name}", f"• Realizado: {_money(realized)} ({len(sel)} operaciones, {wins} ganadas"
                 + (f", {wins / len(sel) * 100:.0f}%)" if sel else ")")]
        if period == "today":
            lines.append(f"• Flotante: {_money(floating)}\n• Variación de equity del día: {_money(st['daily_pnl'])}")
        if sel:
            lines.append(f"• R medio {sum(t.r_multiple for t in sel) / len(sel):+.2f} · mejor {_money(max(t.profit for t in sel))}"
                         f" · peor {_money(min(t.profit for t in sel))}")
        lines.append(f"• Equity actual ${st['equity']:,.2f} (drawdown {st['drawdown'] * 100:.2f}%)")
        images = []
        if period != "today" and sel:
            by_day: Dict[str, float] = {}
            for t in sel:
                by_day[t.close_time.strftime("%m-%d")] = by_day.get(t.close_time.strftime("%m-%d"), 0.0) + t.profit
            images.append(charts.pnl_bars_png(list(by_day.items()), f"PnL por día — {name}"))
        return Reply("\n".join(lines), images)

    def _h_equity_chart(self, **_) -> Reply:
        pts = list(self.o.equity_history)
        return Reply("📈 Curva de equity" if len(pts) >= 2 else "Aún no hay historial suficiente de equity.",
                     [charts.equity_png(pts)] if len(pts) >= 2 else [])

    def _h_trades(self, **_) -> Reply:
        ts = self.o.journal.trades(limit=8)
        if not ts:
            return Reply("🧾 Aún no hay operaciones cerradas.")
        return Reply("🧾 Últimas operaciones:\n" + "\n".join(
            f"• {t.close_time.strftime('%m-%d %H:%M')} {t.symbol} {t.direction.label} {_money(t.profit)} ({t.r_multiple:+.2f}R, {t.exit_reason})"
            for t in reversed(ts)))

    def _analysis(self, sym: str) -> Dict[str, Any]:
        o = self.o
        df = o.conn.get_rates(sym, self.settings.timeframe, 400)
        spec = o.conn.get_symbol_spec(sym)
        sig = o.technical.analyze(sym, df, spec)
        sent = o.fundamental.sentiment(sym, o.now())
        return {"df": df, "signal": sig, "sentiment": sent, "spec": spec}

    def _h_signal(self, sym: Optional[str], **_) -> Reply:
        if not sym:
            return self._symbol_menu("signal")
        a = self._analysis(sym)
        s, se = a["signal"], a["sentiment"]
        head = f"🎯 {sym}: {s.direction.label if s.direction != Direction.FLAT else 'SIN SETUP'} (confianza {s.confidence:.0%})"
        body = ["• " + r for r in s.rationale[:6]] or ["• Sin confluencias técnicas suficientes"]
        if s.direction != Direction.FLAT:
            body.append(f"• Niveles sugeridos: entrada {s.entry:g} · SL {s.sl:g} · TP {s.tp:g}")
        body.append(f"• Sentimiento fundamental: {_sc(se.score)} ({se.label()})")
        body.append("El agente de riesgo decide el tamaño y si se ejecuta al cierre de la vela.")
        return Reply(head + "\n" + "\n".join(body))

    def _h_market(self, sym: Optional[str], **_) -> Reply:
        if not sym:
            syms = list(self.settings.symbols)
            scores = {s: self.o.fundamental.sentiment(s, self.o.now()).score for s in syms}
            return Reply("📊 Panorama: sentimiento por activo (elige uno para el detalle).", [charts.sentiment_png(scores)],
                         buttons=self._symbol_menu("market").buttons)
        a = self._analysis(sym)
        df, sig, se = a["df"], a["signal"], a["sentiment"]
        o = self.o
        now = o.now()
        bars_day = int(1440 / max(1, (df.index[1] - df.index[0]).total_seconds() / 60))
        chg = (df["close"].iloc[-1] / df["close"].iloc[-min(bars_day, len(df))] - 1) * 100
        from .economic_calendar import symbol_currencies
        events = o.calendar.upcoming(now, 24, "medium", symbol_currencies(sym)) if o.calendar else []
        facts = {"simbolo": sym, "precio": float(df["close"].iloc[-1]), "cambio_24h_pct": round(float(chg), 2),
                 "atr": sig.meta.get("atr"), "señal": sig.direction.label, "confianza": round(sig.confidence, 2),
                 "motivos": sig.rationale[:4], "wyckoff": sig.meta.get("wyckoff"), "regimen": sig.meta.get("regime"),
                 "sentimiento": round(se.score, 1), "drivers": se.drivers[:3],
                 "eventos_24h": [f"{e.time:%H:%M} {e.currency} {e.title}" for e in events[:4]]}
        text = self._narrate(facts)
        zones = smc.order_blocks(df)[-3:] + smc.fair_value_gaps(df)[-2:]
        levels = {"SL": sig.sl, "TP": sig.tp, "Entrada": sig.entry} if sig.direction != Direction.FLAT else None
        return Reply(text, [charts.candlestick_png(df, sym, self.settings.timeframe, zones, levels)],
                     buttons=[[("📈 Gráfico", f"cmd:chart:{sym}"), ("🎯 Señal", f"cmd:signal:{sym}"), ("🧠 Sentimiento", f"cmd:sentiment:{sym}")]])

    def _narrate(self, f: Dict[str, Any]) -> str:
        if not self.llm.is_mock:
            txt = self.llm.generate("Redacta un resumen de mercado en español (≤6 líneas, tono profesional, sin promesas de rentabilidad) "
                                    "usando EXCLUSIVAMENTE estos datos:\n" + json.dumps(f, ensure_ascii=False, default=str),
                                    system="Eres analista cuantitativo. No inventes datos que no estén en el JSON.")
            if txt.strip():
                return f"📊 {f['simbolo']}\n{txt.strip()}"
        ev = "; ".join(f["eventos_24h"]) or "sin eventos relevantes"
        mot = "\n".join("  · " + m for m in f["motivos"]) or "  · sin confluencias"
        return (f"📊 {f['simbolo']} — {f['precio']:g} ({f['cambio_24h_pct']:+.2f}% 24h)\n"
                f"• Señal técnica: {f['señal']} (confianza {f['confianza']:.0%}), régimen {f['regimen']}, Wyckoff {f['wyckoff']}\n{mot}\n"
                f"• Sentimiento noticias: {_sc(f['sentimiento'])}" + (f" — {f['drivers'][0]}" if f["drivers"] else "") + "\n"
                f"• Próximos eventos (24h): {ev}")

    def _h_chart(self, sym: Optional[str], **_) -> Reply:
        if not sym:
            return self._symbol_menu("chart")
        df = self.o.conn.get_rates(sym, self.settings.timeframe, 400)
        zones = smc.order_blocks(df)[-3:]
        levels = None
        pos = [p for p in self.o.broker.positions() if p.symbol == sym]
        if pos:
            p = pos[0]
            levels = {"Entrada": p.price_open, "SL": p.sl, "TP": p.tp}
        return Reply(f"📈 {sym} · {self.settings.timeframe}", [charts.candlestick_png(df, sym, self.settings.timeframe, zones, levels)])

    def _h_sentiment(self, sym: Optional[str], **_) -> Reply:
        now = self.o.now()
        if sym:
            r = self.o.fundamental.sentiment(sym, now)
            lines = [f"🧠 Sentimiento {sym}: {_sc(r.score)} ({r.label()}) · confianza {r.confidence:.0%} · {r.n_items} titulares · método {r.method}"]
            lines += ["• " + d for d in r.drivers]
            return Reply("\n".join(lines), [charts.sentiment_png(r.per_entity, f"Sentimiento por divisa/activo ({sym})")])
        scores = {s: self.o.fundamental.sentiment(s, now).score for s in self.settings.symbols}
        return Reply("🧠 Sentimiento por activo", [charts.sentiment_png(scores)])

    def _h_risk(self, **_) -> Reply:
        o, lim = self.o, self.settings.risk
        st = o.status()
        eq = st["equity"] or 1.0
        pos = st["positions"]
        ccy: Dict[str, float] = {}
        open_risk = 0.0
        notional = 0.0
        for p in pos:
            spec = o.conn.get_symbol_spec(p.symbol)
            n = signed_notional(int(p.direction), p.volume, p.price_current or p.price_open, spec)
            b, q = pair_legs(p.symbol)
            ccy[b] = ccy.get(b, 0.0) + n
            ccy[q] = ccy.get(q, 0.0) - n
            notional += abs(n)
            open_risk += o.risk.position_risk(p, spec, eq)
        exp = ", ".join(f"{k} {v / eq:+.1f}x" for k, v in sorted(ccy.items(), key=lambda kv: -abs(kv[1]))[:5]) or "—"
        return Reply(
            "🛡 Estado de riesgo\n"
            f"• Riesgo abierto (a SL): {open_risk / eq * 100:.2f}% / máx {lim.max_total_open_risk * 100:.1f}%\n"
            f"• Apalancamiento nocional: {notional / eq:.1f}x / máx {lim.max_leverage:.0f}x\n• Exposición neta por divisa: {exp} (máx {lim.max_currency_exposure:.0f}x)\n"
            f"• PnL del día {_money(st['daily_pnl'])} ({st['daily_pnl'] / eq * 100:+.2f}%) — corte a −{lim.max_daily_loss * 100:.1f}%\n"
            f"• Drawdown {st['drawdown'] * 100:.2f}% — corte a {lim.max_drawdown * 100:.0f}%\n"
            f"• Por operación: máx {lim.max_risk_per_trade * 100:.1f}% · Kelly ×{lim.kelly_fraction} · VaR{lim.var_confidence * 100:.0f}% ≤ {lim.max_var_pct * 100:.1f}%\n"
            f"• Posiciones: {len(pos)}/{lim.max_positions}")

    def _h_lessons(self, **_) -> Reply:
        j = self.o.journal
        ls = j.lessons(8)
        head = f"🎓 Memoria: {j.count()} operaciones registradas."
        return Reply(head + ("\n" + "\n".join("• " + l for l in ls) if ls else "\nAún no hay patrones estadísticamente relevantes."))

    def _h_calendar(self, **_) -> Reply:
        now = self.o.now()
        evs = self.o.calendar.upcoming(now, 48, "medium") if self.o.calendar else []
        if not evs:
            return Reply("📅 Sin eventos de impacto medio/alto en las próximas 48 h (o el calendario está vacío: scripts/update_calendar.py).")
        return Reply("📅 Próximos eventos (48 h):\n" + "\n".join(
            f"• {e.time:%a %H:%M} UTC {'🔴' if e.impact == 'high' else '🟠'} {e.currency} {e.title}" + (f" (prev. {e.forecast})" if e.forecast else "")
            for e in evs[:12]))

    def _h_decisions(self, **_) -> Reply:
        d = list(self.o.decision_log)[-6:]
        if not d:
            return Reply("🔎 Todavía no se ha evaluado ninguna propuesta (sin setups desde el arranque).")
        return Reply("🔎 Últimas decisiones:\n" + "\n".join(f"• {x['time'][11:16]} {x['symbol']} {x['direction']}: {x['summary']}" for x in reversed(d)))

    def _h_pause(self, is_admin: bool, **_) -> Reply:
        if not is_admin:
            return Reply("⛔ Solo un administrador puede pausar el sistema.")
        self.o.pause()
        return Reply("⏸ Sistema en pausa: no se abrirán operaciones nuevas (las abiertas siguen con su SL/TP).", buttons=self.main_menu(True))

    def _h_resume(self, is_admin: bool, **_) -> Reply:
        if not is_admin:
            return Reply("⛔ Solo un administrador puede reanudar el sistema.")
        self.o.resume()
        return Reply("▶️ Sistema reanudado.", buttons=self.main_menu(True))

    def _h_close_all(self, is_admin: bool, **_) -> Reply:
        if not is_admin:
            return Reply("⛔ Solo un administrador puede cerrar todas las posiciones.")
        n = len(self.o.broker.positions())
        return Reply(f"⚠️ ¿Cerrar las {n} posiciones abiertas a mercado?",
                     buttons=[[("✅ Confirmar", "confirm:close_all"), ("❌ Cancelar", "cancel")]])

    def format_event(self, kind: str, p: Dict[str, Any]) -> Optional[str]:
        if kind == "trade_opened":
            why = "\n".join("  · " + r for r in p.get("rationale", [])[:4])
            les = ("\n🎓 " + p["lessons"][0]) if p.get("lessons") else ""
            return (f"🟢 ABIERTA {p['symbol']} {p['direction']} {p['lots']:.2f} lotes @ {p['price']:g}\nSL {p['sl']:g} · TP {p['tp']:g} · "
                    f"riesgo ${p['risk_amount']:,.2f} ({p['risk_pct'] * 100:.2f}%) · p={p['win_prob']:.2f}\n{why}{les}")
        if kind == "trade_closed":
            return f"{'✅' if p['profit'] > 0 else '🔴'} CERRADA {p['symbol']} {_money(p['profit'])} ({p['r']:+.2f}R, {p['reason']})"
        if kind == "halt":
            return f"🚨 SISTEMA DETENIDO: {p.get('reason')}"
        if kind == "connection_lost":
            return f"⚠️ Conexión con MT5 perdida: {p.get('error')}"
        return None
