"""Conexión resiliente y SEGURA a MetaTrader 5."""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings, load_settings
from .approval import DEFAULT_AUTHORITY, ApprovalAuthority
from .exceptions import (
    DataUnavailableError,
    MT5ConnectionError,
    OrderRejectedError,
    RealAccountBlockedError,
)
from .timeframes import TF_MT5_CONST
from .types import ApprovedOrder, Direction, OrderResult, PositionInfo, SymbolSpec

log = logging.getLogger(__name__)

_LIVE_SERVER_TOKENS = ("live", "real", "prod")


def _epoch_to_utc(seconds) -> pd.DatetimeIndex:
    return pd.to_datetime(seconds, unit="s", utc=True)


class LatencyTracker:
    """Registra la latencia de las llamadas al terminal."""

    def __init__(self, maxlen: int = 200, alpha: float = 0.2) -> None:
        self._samples: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._ewma: Dict[str, float] = {}
        self.alpha = alpha
        self.terminal_ping_ms: float = 0.0

    def record(self, op: str, seconds: float) -> None:
        ms = seconds * 1000.0
        self._samples[op].append(ms)
        prev = self._ewma.get(op)
        self._ewma[op] = ms if prev is None else self.alpha * ms + (1 - self.alpha) * prev

    def p95(self, op: str | None = None) -> float:
        vals = list(self._samples[op]) if op else [v for d in self._samples.values() for v in d]
        if not vals:
            return 0.0
        return float(np.percentile(vals, 95))

    def summary(self) -> Dict[str, Any]:
        return {
            "ping_terminal_ms": round(self.terminal_ping_ms, 1),
            "p95_ms": round(self.p95(), 1),
            "ewma_ms": {k: round(v, 1) for k, v in self._ewma.items()},
        }


class MT5Connection:
    """Fachada del terminal MT5 con reconexión, control de latencia y bloqueo de cuentas reales."""

    def __init__(
        self,
        settings: Settings | None = None,
        mt5: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        authority: ApprovalAuthority | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._sleep = sleep
        self._authority = authority or DEFAULT_AUTHORITY
        self._mt5 = mt5 if mt5 is not None else self._load_backend()
        self._connected = False
        self._lock = threading.RLock()
        self.latency = LatencyTracker()
        self._spec_cache: Dict[str, tuple[float, SymbolSpec]] = {}
        self.reconnects = 0
        self.utc_offset = timedelta(0)
        self.offset_calibrated = False
        self._offset_file = self.settings.state_dir / "mt5_utc_offset.json"

    @property
    def authority(self) -> ApprovalAuthority:
        return self._authority

    def calibrate_time_offset(self) -> timedelta:
        s = self.settings
        if s.mt5_utc_offset_hours is not None:
            self.utc_offset, self.offset_calibrated = timedelta(hours=s.mt5_utc_offset_hours), True
            return self.utc_offset
        if self.is_mock:
            self.offset_calibrated = True
            return self.utc_offset
        now = datetime.now(timezone.utc)
        open_market = now.weekday() < 4 or (now.weekday() == 4 and now.hour < 20)
        if open_market:
            offsets = []
            for sym in list(s.symbols)[:3]:
                try:
                    tick = self._mt5.symbol_info_tick(sym)
                    if tick is not None and tick.time:
                        raw = (datetime.fromtimestamp(tick.time, tz=timezone.utc) - now).total_seconds()
                        if abs(raw) <= 15 * 3600:
                            offsets.append(round(raw / 1800.0) * 1800.0)
                except Exception:
                    continue
            if len(offsets) >= 2 and max(offsets) - min(offsets) <= 1800:
                self.utc_offset = timedelta(seconds=sorted(offsets)[len(offsets) // 2])
                self.offset_calibrated = True
                try:
                    self._offset_file.parent.mkdir(parents=True, exist_ok=True)
                    self._offset_file.write_text(json.dumps({"seconds": self.utc_offset.total_seconds()}), encoding="utf-8")
                except OSError:
                    pass
                log.info("Offset hora-servidor calibrado: %+.1f h respecto a UTC", self.utc_offset.total_seconds() / 3600)
                return self.utc_offset
        if not self.offset_calibrated and self._offset_file.exists():
            try:
                self.utc_offset = timedelta(seconds=json.loads(self._offset_file.read_text())["seconds"])
                log.info("Offset hora-servidor (persistido): %+.1f h", self.utc_offset.total_seconds() / 3600)
            except Exception:
                pass
        return self.utc_offset

    def _from_server(self, idx):
        return idx - self.utc_offset

    def _to_server(self, dt: datetime) -> datetime:
        return dt + self.utc_offset

    def _load_backend(self) -> Any:
        if self.settings.use_mock_mt5:
            from .mock_mt5 import MockMT5
            log.warning("USE_MOCK_MT5=1: usando backend MT5 simulado")
            return MockMT5(symbols=list(self.settings.symbols))
        try:
            import MetaTrader5 as mt5  # type: ignore
            return mt5
        except Exception as exc:  # paquete ausente (p. ej. Linux/Mac)
            from .mock_mt5 import MockMT5
            log.warning("MetaTrader5 no disponible (%s): usando backend MT5 simulado", exc)
            return MockMT5(symbols=list(self.settings.symbols))

    @property
    def backend(self) -> Any:
        return self._mt5

    @property
    def is_mock(self) -> bool:
        return bool(getattr(self._mt5, "is_mock", False))

    def assert_demo_account(self) -> Any:
        mt5 = self._mt5
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError(f"account_info() vacío: {mt5.last_error()}")
        demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
        server = str(getattr(info, "server", "") or "").lower()
        looks_live = any(t in server for t in _LIVE_SERVER_TOKENS) and "demo" not in server
        if info.trade_mode != demo_mode or looks_live:
            try:
                mt5.shutdown()
            finally:
                self._connected = False
            raise RealAccountBlockedError(
                f"CUENTA REAL/NO-DEMO DETECTADA (login={getattr(info, 'login', '?')}, "
                f"server={getattr(info, 'server', '?')!r}, trade_mode={info.trade_mode}). "
                "Este sistema solo opera en DEMO. Ejecución detenida."
            )
        return info

    def connect(self) -> None:
        s = self.settings
        kwargs: Dict[str, Any] = {"timeout": s.mt5_init_timeout_ms}
        if s.mt5_path:
            kwargs["path"] = s.mt5_path
        if s.mt5_login:
            kwargs.update(login=s.mt5_login, password=s.mt5_password, server=s.mt5_server)
        last_err: Any = None
        with self._lock:
            for attempt in range(1, s.mt5_max_retries + 1):
                t0 = time.perf_counter()
                ok = self._mt5.initialize(**kwargs)
                self.latency.record("initialize", time.perf_counter() - t0)
                if ok:
                    self.assert_demo_account()          # nunca se reintenta un bloqueo de seguridad
                    self._connected = True
                    self._refresh_ping()
                    self.calibrate_time_offset()
                    log.info("MT5 conectado (%s) intento %d/%d", "MOCK" if self.is_mock else "REAL-TERMINAL",
                             attempt, s.mt5_max_retries)
                    return
                last_err = self._mt5.last_error()
                delay = min(s.mt5_retry_base_delay * 2 ** (attempt - 1), 60.0) * (0.75 + 0.5 * random.random())
                log.warning("initialize() falló (%s). Reintento %d/%d en %.1fs", last_err, attempt,
                            s.mt5_max_retries, delay)
                self._sleep(delay)
        hint = ""
        if last_err and last_err[0] == -10005:
            hint = (" | Ayuda (IPC timeout): 1) cierra TODOS los terminales MT5; 2) ábrelo normalmente desde el menú Inicio y "
                    "espera a que muestre la cuenta DEMO conectada (barra inferior con KB/s); 3) ejecuta este programa desde una "
                    "terminal con el MISMO nivel de privilegios que el terminal (ambos normales o ambos administrador); "
                    "4) si tienes varias instalaciones, fija MT5_PATH en .env.")
        raise MT5ConnectionError(f"No se pudo conectar a MT5 tras {s.mt5_max_retries} intentos: {last_err}{hint}")

    def disconnect(self) -> None:
        with self._lock:
            try:
                self._mt5.shutdown()
            finally:
                self._connected = False

    def _refresh_ping(self) -> None:
        ti = self._mt5.terminal_info()
        if ti is not None:
            self.latency.terminal_ping_ms = float(getattr(ti, "ping_last", 0) or 0) / 1000.0

    def is_alive(self) -> bool:
        ti = self._mt5.terminal_info()
        return bool(ti is not None and getattr(ti, "connected", False) and self._mt5.account_info() is not None)

    def ensure_connected(self) -> None:
        with self._lock:
            if self._connected and self.is_alive():
                return
            log.warning("Conexión MT5 perdida; reconectando…")
            self._connected = False
            self.reconnects += 1
            self.connect()

    def _call(self, op: str, fn: Callable[[], Any], retries: int = 3, ok: Callable[[Any], bool] | None = None) -> Any:
        ok = ok or (lambda r: r is not None)
        for attempt in range(1, retries + 1):
            self.ensure_connected()
            t0 = time.perf_counter()
            try:
                result = fn()
            except (RealAccountBlockedError, MT5ConnectionError):
                raise
            except Exception as exc:  # errores del IPC
                log.warning("%s lanzó %r (intento %d/%d)", op, exc, attempt, retries)
                result = None
            self.latency.record(op, time.perf_counter() - t0)
            if ok(result):
                return result
            err = self._mt5.last_error()
            log.warning("%s sin datos (%s) intento %d/%d", op, err, attempt, retries)
            if not self.is_alive():
                self._connected = False
            self._sleep(min(0.5 * 2 ** (attempt - 1), 5.0))
        return None

    def _tf(self, timeframe: str) -> int:
        name = timeframe.upper()
        const = getattr(self._mt5, f"TIMEFRAME_{name}", None)
        if const is None:
            const = TF_MT5_CONST.get(name)
        if const is None:
            raise ValueError(f"Timeframe no soportado: {timeframe}")
        return const

    def _rates_to_df(self, rates: Any) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = self._from_server(_epoch_to_utc(df["time"]))
        df = df.set_index("time").sort_index()
        return df[~df.index.duplicated(keep="last")]

    def select_symbol(self, symbol: str) -> None:
        if not self._mt5.symbol_select(symbol, True):
            raise DataUnavailableError(f"Símbolo no disponible en el broker: {symbol}")

    def get_rates(self, symbol: str, timeframe: str = "M15", count: int = 500,
                  include_current: bool = False) -> pd.DataFrame:
        self.select_symbol(symbol)
        pos = 0 if include_current else 1
        rates = self._call(f"rates:{symbol}", lambda: self._mt5.copy_rates_from_pos(
            symbol, self._tf(timeframe), pos, count), ok=lambda r: r is not None and len(r) > 0)
        if rates is None:
            raise DataUnavailableError(f"Sin velas para {symbol} {timeframe}")
        return self._rates_to_df(rates)

    def get_rates_range(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.select_symbol(symbol)
        rates = self._call(f"rates_range:{symbol}", lambda: self._mt5.copy_rates_range(
            symbol, self._tf(timeframe), self._to_server(start), self._to_server(end)),
            ok=lambda r: r is not None and len(r) > 0)
        if rates is None:
            raise DataUnavailableError(f"Sin velas para {symbol} {timeframe} en [{start}, {end}]")
        return self._rates_to_df(rates)

    def get_ticks(self, symbol: str, start: datetime, end: datetime, chunk_hours: int = 6) -> pd.DataFrame:
        self.select_symbol(symbol)
        flags = getattr(self._mt5, "COPY_TICKS_ALL", -1)
        frames: List[pd.DataFrame] = []
        cur = start
        step = timedelta(hours=chunk_hours)
        while cur < end:
            nxt = min(cur + step, end)
            ticks = self._call(f"ticks:{symbol}", lambda a=cur, b=nxt: self._mt5.copy_ticks_range(
                symbol, self._to_server(a), self._to_server(b), flags),
                               retries=2, ok=lambda r: r is not None)
            if ticks is not None and len(ticks):
                d = pd.DataFrame(ticks)
                d["time"] = self._from_server(pd.to_datetime(d["time_msc"], unit="ms", utc=True))
                frames.append(d.set_index("time")[["bid", "ask", "last", "volume", "flags"]])
            cur = nxt
        if not frames:
            raise DataUnavailableError(f"Sin ticks para {symbol} en [{start}, {end}]")
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]

    def get_symbol_spec(self, symbol: str, max_age: float = 300.0) -> SymbolSpec:
        cached = self._spec_cache.get(symbol)
        if cached and time.time() - cached[0] < max_age:
            return cached[1]
        self.select_symbol(symbol)
        i = self._call(f"symbol_info:{symbol}", lambda: self._mt5.symbol_info(symbol))
        if i is None:
            raise DataUnavailableError(f"symbol_info vacío para {symbol}")
        spec = SymbolSpec(
            symbol=symbol, digits=i.digits, point=i.point, tick_size=i.trade_tick_size or i.point,
            tick_value=i.trade_tick_value, contract_size=i.trade_contract_size, volume_min=i.volume_min,
            volume_max=i.volume_max, volume_step=i.volume_step, currency_base=i.currency_base,
            currency_profit=i.currency_profit, spread_points=float(i.spread),
            stops_level_points=float(getattr(i, "trade_stops_level", 0)), filling_mode=int(getattr(i, "filling_mode", 1)),
        )
        self._spec_cache[symbol] = (time.time(), spec)
        return spec

    def get_quote(self, symbol: str) -> Dict[str, float]:
        self.select_symbol(symbol)
        t = self._call(f"tick:{symbol}", lambda: self._mt5.symbol_info_tick(symbol))
        if t is None:
            raise DataUnavailableError(f"Sin tick para {symbol}")
        return {"bid": float(t.bid), "ask": float(t.ask), "time": float(t.time)}

    def server_time(self) -> datetime:
        try:
            sym = self.settings.symbols[0]
            q = self.get_quote(sym)
            return datetime.fromtimestamp(q["time"], tz=timezone.utc) - self.utc_offset
        except Exception:
            return datetime.now(timezone.utc)

    def get_account(self) -> Any:
        info = self._call("account_info", lambda: self._mt5.account_info())
        if info is None:
            raise DataUnavailableError("account_info vacío")
        return info

    def get_positions(self) -> List[PositionInfo]:
        raw = self._call("positions_get", lambda: self._mt5.positions_get(), ok=lambda r: r is not None)
        out: List[PositionInfo] = []
        for p in raw or ():
            out.append(PositionInfo(
                ticket=int(p.ticket), symbol=p.symbol,
                direction=Direction.LONG if p.type == getattr(self._mt5, "POSITION_TYPE_BUY", 0) else Direction.SHORT,
                volume=float(p.volume), price_open=float(p.price_open), sl=float(p.sl), tp=float(p.tp),
                price_current=float(p.price_current), profit=float(p.profit),
                open_time=datetime.fromtimestamp(p.time, tz=timezone.utc) - self.utc_offset, magic=int(p.magic),
                comment=p.comment,
            ))
        return out

    def get_deals(self, start: datetime, end: datetime) -> pd.DataFrame:
        raw = self._call("history_deals", lambda: self._mt5.history_deals_get(
            self._to_server(start), self._to_server(end)), ok=lambda r: r is not None)
        cols = ["ticket", "order", "time", "type", "entry", "position_id", "volume", "price", "commission",
                "swap", "profit", "symbol", "magic", "comment", "reason"]
        if not raw:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([{c: getattr(d, c, None) for c in cols} for d in raw])
        df["time"] = self._from_server(_epoch_to_utc(df["time"]))
        return df

    def _filling(self, symbol: str) -> int:
        mt5 = self._mt5
        mode = self.get_symbol_spec(symbol).filling_mode
        if mode & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
            return getattr(mt5, "ORDER_FILLING_IOC", 1)
        if mode & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
            return getattr(mt5, "ORDER_FILLING_FOK", 0)
        return getattr(mt5, "ORDER_FILLING_RETURN", 2)

    def send_order(self, order: ApprovedOrder) -> OrderResult:
        self._authority.verify(order)
        with self._lock:
            self.ensure_connected()
            self.assert_demo_account()
            mt5 = self._mt5
            buy = order.direction == Direction.LONG
            retry_codes = {10004, 10020, 10021, 10031}   # requote / precio cambió / sin precios / conexión
            last: Any = None
            for attempt in range(1, 4):
                q = self.get_quote(order.symbol)
                price = q["ask"] if buy else q["bid"]
                request = {
                    "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
                    "symbol": order.symbol,
                    "volume": float(order.volume),
                    "type": getattr(mt5, "ORDER_TYPE_BUY", 0) if buy else getattr(mt5, "ORDER_TYPE_SELL", 1),
                    "price": price,
                    "sl": float(order.sl),
                    "tp": float(order.tp),
                    "deviation": self.settings.max_slippage_points,
                    "magic": self.settings.magic_number,
                    "comment": order.comment[:31],
                    "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
                    "type_filling": self._filling(order.symbol),
                }
                chk = mt5.order_check(request)
                if chk is not None and chk.retcode not in (0, getattr(mt5, "TRADE_RETCODE_DONE", 10009)):
                    raise OrderRejectedError(f"order_check rechazó la orden: {chk.retcode} {chk.comment}")
                t0 = time.perf_counter()
                last = mt5.order_send(request)
                self.latency.record("order_send", time.perf_counter() - t0)
                if last is None:
                    log.warning("order_send devolvió None (%s)", mt5.last_error())
                    self._sleep(0.3 * attempt)
                    continue
                done = {getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)}
                if last.retcode in done:
                    log.info("ORDEN OK %s %s %.2f @ %.5f ticket=%s", order.symbol, "BUY" if buy else "SELL",
                             order.volume, last.price, last.order)
                    return OrderResult(ok=True, ticket=int(last.order), price=float(last.price),
                                       volume=float(last.volume), retcode=int(last.retcode), comment=str(last.comment))
                if last.retcode in retry_codes:
                    self._sleep(0.3 * attempt)
                    continue
                break
            code = getattr(last, "retcode", None)
            raise OrderRejectedError(f"Orden rechazada por el broker: retcode={code} {getattr(last, 'comment', '')}")

    def close_position(self, ticket: int, comment: str = "AGI close") -> OrderResult:
        with self._lock:
            self.ensure_connected()
            self.assert_demo_account()
            mt5 = self._mt5
            pos = self._call("positions_get", lambda: mt5.positions_get(ticket=ticket), ok=lambda r: r is not None)
            if not pos:
                raise OrderRejectedError(f"No existe la posición {ticket}")
            p = pos[0]
            buy_pos = p.type == getattr(mt5, "POSITION_TYPE_BUY", 0)
            q = self.get_quote(p.symbol)
            request = {
                "action": getattr(mt5, "TRADE_ACTION_DEAL", 1), "symbol": p.symbol, "volume": float(p.volume),
                "type": getattr(mt5, "ORDER_TYPE_SELL", 1) if buy_pos else getattr(mt5, "ORDER_TYPE_BUY", 0),
                "position": int(ticket), "price": q["bid"] if buy_pos else q["ask"],
                "deviation": self.settings.max_slippage_points, "magic": self.settings.magic_number,
                "comment": comment[:31], "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
                "type_filling": self._filling(p.symbol),
            }
            res = mt5.order_send(request)
            if res is None or res.retcode != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                raise OrderRejectedError(f"Cierre rechazado: {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}")
            return OrderResult(ok=True, ticket=int(ticket), price=float(res.price), volume=float(p.volume),
                               retcode=int(res.retcode), comment=str(res.comment))
