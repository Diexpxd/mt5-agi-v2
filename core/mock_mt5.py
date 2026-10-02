"""Backend MetaTrader5 simulado (misma superficie que el paquete ``MetaTrader5``)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .synthetic import SYMBOL_PROFILES, generate_ohlcv, resample_ohlcv
from .timeframes import TF_MINUTES, TF_MT5_CONST

_RATES_DTYPE = np.dtype([
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
    ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8"),
])
_TICKS_DTYPE = np.dtype([
    ("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"), ("volume", "<u8"),
    ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8"),
])

_CONTRACT = {"XAUUSD": 100.0, "BTCUSD": 1.0, "ETHUSD": 1.0}


class MockMT5:
    """Simula el módulo ``MetaTrader5`` con un mercado sintético avanzable en el tiempo."""

    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_STOPS = 10016
    COPY_TICKS_ALL = -1
    COPY_TICKS_INFO = 1
    COPY_TICKS_TRADE = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    for _n, _v in TF_MT5_CONST.items():
        locals()[f"TIMEFRAME_{_n}"] = _v
    del _n, _v

    is_mock = True

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        seed: int = 7,
        n_minutes: int = 60_000,
        account_mode: int = 0,
        server: str = "MockBroker-Demo",
        balance: float = 10_000.0,
        leverage: int = 100,
        end: Optional[datetime] = None,
        fail_initialize_times: int = 0,
        commission_per_lot: float = 7.0,
        future_minutes: Optional[int] = None,
    ) -> None:
        self.symbols = [s.upper() for s in (symbols or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])]
        self.seed = seed
        self.account_mode = account_mode
        self.server = server
        self.balance = float(balance)
        self.leverage = leverage
        self.commission_per_lot = commission_per_lot
        self.connected = False
        self._fail_init = fail_initialize_times
        self._last_error = (1, "Success")
        self._m1: Dict[str, pd.DataFrame] = {
            s: generate_ohlcv(s, n=n_minutes, minutes=1, seed=seed, end=end) for s in self.symbols
        }
        self._cache: Dict[tuple, pd.DataFrame] = {}
        future = n_minutes // 4 if future_minutes is None else future_minutes
        self._ptr = n_minutes - 1 - future             # índice M1 del "ahora"
        self._positions: Dict[int, SimpleNamespace] = {}
        self._deals: List[SimpleNamespace] = []
        self._next_ticket = 100_000
        self._selected = set(self.symbols)
        self.order_log: List[dict] = []

    def initialize(self, *args, **kwargs) -> bool:
        if self._fail_init > 0:
            self._fail_init -= 1
            self._last_error = (-10005, "IPC timeout")
            return False
        self.connected = True
        self._last_error = (1, "Success")
        return True

    def shutdown(self) -> None:
        self.connected = False

    def last_error(self):
        return self._last_error

    def simulate_disconnect(self) -> None:
        self.connected = False
        self._last_error = (-10004, "No IPC connection")

    def terminal_info(self):
        if not self.connected:
            return None
        return SimpleNamespace(connected=True, trade_allowed=True, ping_last=25_000, name="MockTerminal",
                               company="Mock Broker", build=5000)

    def _now_ts(self) -> int:
        return int(self._m1[self.symbols[0]].index[self._ptr].timestamp())

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._now_ts(), tz=timezone.utc)

    def _price(self, symbol: str) -> float:
        return float(self._m1[symbol]["close"].iloc[self._ptr])

    def _spec(self, symbol: str) -> SimpleNamespace:
        prof = SYMBOL_PROFILES.get(symbol, dict(price=100.0, digits=3, spread=20))
        digits = prof["digits"]
        point = 10 ** -digits
        contract = _CONTRACT.get(symbol, 100_000.0)
        price = self._price(symbol)
        quote = symbol[3:6]
        tick_value = contract * point / (price if quote == "JPY" else 1.0)
        return SimpleNamespace(
            name=symbol, visible=True, digits=digits, point=point, spread=prof["spread"],
            trade_tick_size=point, trade_tick_value=tick_value, trade_contract_size=contract,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            currency_base=symbol[:3], currency_profit=quote, currency_margin=symbol[:3],
            filling_mode=3, trade_stops_level=0, trade_mode=4, bid=price, ask=price + prof["spread"] * point,
        )

    def symbols_get(self, *args, **kwargs):
        return tuple(SimpleNamespace(name=s) for s in self.symbols)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        if symbol.upper() not in self._m1:
            return False
        self._selected.add(symbol.upper())
        return True

    def symbol_info(self, symbol: str):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        return self._spec(symbol.upper())

    def symbol_info_tick(self, symbol: str):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        s = self._spec(symbol.upper())
        return SimpleNamespace(time=self._now_ts(), bid=s.bid, ask=s.ask, last=s.bid, volume=0,
                               time_msc=self._now_ts() * 1000, flags=6, volume_real=0.0)

    def _frame(self, symbol: str, tf: int) -> pd.DataFrame:
        minutes = {v: TF_MINUTES[k] for k, v in TF_MT5_CONST.items()}.get(tf, tf)
        m1 = self._m1[symbol]
        key = (symbol, minutes)
        full = self._cache.get(key)
        if full is None:
            full = resample_ohlcv(m1, minutes)
            self._cache[key] = full
        now = m1.index[self._ptr]
        upto = full.loc[:now].copy()
        if len(upto) and minutes > 1:
            last_start = upto.index[-1]
            part = m1.loc[last_start:now]
            upto.iloc[-1, upto.columns.get_loc("high")] = part["high"].max()
            upto.iloc[-1, upto.columns.get_loc("low")] = part["low"].min()
            upto.iloc[-1, upto.columns.get_loc("close")] = part["close"].iloc[-1]
        return upto

    @staticmethod
    def _to_rates(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=_RATES_DTYPE)
        out["time"] = _epoch_seconds(df.index)
        for c in ("open", "high", "low", "close"):
            out[c] = df[c].to_numpy()
        out["tick_volume"] = df["tick_volume"].to_numpy()
        out["spread"] = df["spread"].to_numpy()
        out["real_volume"] = df["real_volume"].to_numpy()
        return out

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        df = self._frame(symbol.upper(), timeframe)
        end = len(df) - start_pos
        if end <= 0:
            return None
        return self._to_rates(df.iloc[max(0, end - count):end])

    def copy_rates_range(self, symbol: str, timeframe: int, date_from, date_to):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        df = self._frame(symbol.upper(), timeframe)
        a = pd.Timestamp(date_from).tz_convert("UTC") if pd.Timestamp(date_from).tzinfo else pd.Timestamp(date_from, tz="UTC")
        b = pd.Timestamp(date_to).tz_convert("UTC") if pd.Timestamp(date_to).tzinfo else pd.Timestamp(date_to, tz="UTC")
        sub = df.loc[a:b]
        return self._to_rates(sub) if len(sub) else None

    def _ticks_from_bars(self, symbol: str, bars: pd.DataFrame) -> np.ndarray:
        spread = SYMBOL_PROFILES.get(symbol, dict(spread=20, digits=3))
        point = 10 ** -spread["digits"]
        sp = spread["spread"] * point
        n = len(bars)
        ts = _epoch_seconds(bars.index)
        o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        up = c >= o
        p1, p4 = o, c
        p2 = np.where(up, l, h)
        p3 = np.where(up, h, l)
        prices = np.stack([p1, p2, p3, p4], axis=1).reshape(-1)
        out = np.zeros(n * 4, dtype=_TICKS_DTYPE)
        out["time"] = np.repeat(ts, 4)
        out["time_msc"] = np.repeat(ts, 4) * 1000 + np.tile([0, 15_000, 30_000, 45_000], n)
        out["bid"] = prices
        out["ask"] = prices + sp
        out["last"] = prices
        out["flags"] = 6
        return out

    def copy_ticks_range(self, symbol: str, date_from, date_to, flags: int = -1):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        m1 = self._m1[symbol.upper()].iloc[: self._ptr + 1]
        a = pd.Timestamp(date_from)
        b = pd.Timestamp(date_to)
        a = a if a.tzinfo else a.tz_localize("UTC")
        b = b if b.tzinfo else b.tz_localize("UTC")
        sub = m1.loc[a:b]
        return self._ticks_from_bars(symbol.upper(), sub) if len(sub) else None

    def copy_ticks_from(self, symbol: str, date_from, count: int, flags: int = -1):
        if not self.connected or symbol.upper() not in self._m1:
            return None
        m1 = self._m1[symbol.upper()].iloc[: self._ptr + 1]
        a = pd.Timestamp(date_from)
        a = a if a.tzinfo else a.tz_localize("UTC")
        sub = m1.loc[a:].iloc[: max(1, count // 4 + 1)]
        t = self._ticks_from_bars(symbol.upper(), sub)
        return t[:count] if len(t) else None

    def _floating(self) -> float:
        return sum(self._position_profit(p) for p in self._positions.values())

    def _margin_used(self) -> float:
        total = 0.0
        for p in self._positions.values():
            s = self._spec(p.symbol)
            total += p.volume * s.trade_contract_size * p.price_open / self.leverage if s.currency_profit == "USD" \
                and s.currency_base != "USD" else p.volume * s.trade_contract_size / self.leverage
        return total

    def account_info(self):
        if not self.connected:
            return None
        equity = self.balance + self._floating()
        margin = self._margin_used()
        return SimpleNamespace(
            login=99999999, trade_mode=self.account_mode, leverage=self.leverage, balance=self.balance,
            equity=equity, profit=self._floating(), margin=margin, margin_free=equity - margin,
            margin_level=(equity / margin * 100.0) if margin > 0 else 0.0, currency="USD",
            server=self.server, name="Mock Trader", company="Mock Broker", trade_allowed=True,
        )

    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float):
        s = self.symbol_info(symbol)
        if s is None:
            return None
        if s.currency_base == "USD":
            return volume * s.trade_contract_size / self.leverage
        return volume * s.trade_contract_size * price / self.leverage

    def _position_profit(self, p: SimpleNamespace) -> float:
        s = self._spec(p.symbol)
        px = s.bid if p.type == self.POSITION_TYPE_BUY else s.ask
        sign = 1 if p.type == self.POSITION_TYPE_BUY else -1
        return sign * (px - p.price_open) / s.trade_tick_size * s.trade_tick_value * p.volume

    def _pos_view(self, p: SimpleNamespace) -> SimpleNamespace:
        s = self._spec(p.symbol)
        cur = s.bid if p.type == self.POSITION_TYPE_BUY else s.ask
        return SimpleNamespace(**{**vars(p), "price_current": cur, "profit": self._position_profit(p)})

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None, group=None):
        if not self.connected:
            return None
        out = [self._pos_view(p) for p in self._positions.values()
               if (symbol is None or p.symbol == symbol.upper()) and (ticket is None or p.ticket == ticket)]
        return tuple(out)

    def history_deals_get(self, date_from=None, date_to=None, group=None, position=None, ticket=None):
        if not self.connected:
            return None
        a = pd.Timestamp(date_from).timestamp() if date_from is not None else 0
        b = pd.Timestamp(date_to).timestamp() if date_to is not None else 10**12
        out = [d for d in self._deals if a <= d.time <= b and (position is None or d.position_id == position)]
        return tuple(out)

    def order_check(self, request: dict):
        return self._validate(request)

    def _validate(self, req: dict) -> SimpleNamespace:
        ok = lambda rc, msg: SimpleNamespace(retcode=rc, comment=msg, request=req)  # noqa: E731
        if not self.connected:
            return ok(-1, "no connection")
        sym = req["symbol"].upper()
        s = self.symbol_info(sym)
        if s is None:
            return ok(self.TRADE_RETCODE_REJECT, "unknown symbol")
        vol = float(req["volume"])
        if vol < s.volume_min - 1e-9 or vol > s.volume_max + 1e-9 or abs(vol / s.volume_step - round(vol / s.volume_step)) > 1e-6:
            return ok(self.TRADE_RETCODE_INVALID_VOLUME, "invalid volume")
        if req.get("action") == self.TRADE_ACTION_DEAL and "position" not in req:
            price = s.ask if req["type"] == self.ORDER_TYPE_BUY else s.bid
            margin = self.order_calc_margin(req["action"], sym, vol, price)
            if margin is not None and margin > self.account_info().margin_free:
                return ok(self.TRADE_RETCODE_NO_MONEY, "no money")
            sl, tp = req.get("sl", 0.0), req.get("tp", 0.0)
            if req["type"] == self.ORDER_TYPE_BUY and ((sl and sl >= price) or (tp and tp <= price)):
                return ok(self.TRADE_RETCODE_INVALID_STOPS, "invalid stops")
            if req["type"] == self.ORDER_TYPE_SELL and ((sl and sl <= price) or (tp and tp >= price)):
                return ok(self.TRADE_RETCODE_INVALID_STOPS, "invalid stops")
        return ok(0, "Done")

    def order_send(self, request: dict):
        self.order_log.append(dict(request))
        chk = self._validate(request)
        if chk.retcode != 0:
            return SimpleNamespace(retcode=chk.retcode, comment=chk.comment, order=0, deal=0, volume=0.0,
                                   price=0.0, request=request)
        sym = request["symbol"].upper()
        s = self.symbol_info(sym)
        if request["action"] == self.TRADE_ACTION_SLTP:
            p = self._positions.get(request["position"])
            if p is None:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, comment="no position", order=0, deal=0,
                                       volume=0.0, price=0.0, request=request)
            p.sl, p.tp = request.get("sl", p.sl), request.get("tp", p.tp)
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="Done", order=0, deal=0,
                                   volume=p.volume, price=0.0, request=request)
        if "position" in request:                       # cierre
            p = self._positions.get(request["position"])
            if p is None:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT, comment="no position", order=0, deal=0,
                                       volume=0.0, price=0.0, request=request)
            px = s.bid if p.type == self.POSITION_TYPE_BUY else s.ask
            return self._close(p, px, request.get("comment", "close"), reason=0)
        typ = request["type"]
        price = s.ask if typ == self.ORDER_TYPE_BUY else s.bid
        self._next_ticket += 1
        t = self._next_ticket
        pos = SimpleNamespace(
            ticket=t, symbol=sym, type=typ, volume=float(request["volume"]), price_open=price,
            sl=float(request.get("sl", 0.0)), tp=float(request.get("tp", 0.0)), time=self._now_ts(),
            magic=int(request.get("magic", 0)), comment=request.get("comment", ""), identifier=t,
        )
        self._positions[t] = pos
        comm = -self.commission_per_lot * pos.volume / 2
        self._deals.append(SimpleNamespace(ticket=self._next_ticket + 500_000, order=t, time=self._now_ts(), type=typ,
                                           entry=self.DEAL_ENTRY_IN, position_id=t, volume=pos.volume, price=price,
                                           commission=comm, swap=0.0, profit=0.0, symbol=sym, magic=pos.magic,
                                           comment=pos.comment, reason=0))
        self.balance += comm
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="Done", order=t, deal=t + 500_000,
                               volume=pos.volume, price=price, request=request)

    def _close(self, p: SimpleNamespace, price: float, comment: str, reason: int):
        s = self._spec(p.symbol)
        sign = 1 if p.type == self.POSITION_TYPE_BUY else -1
        profit = sign * (price - p.price_open) / s.trade_tick_size * s.trade_tick_value * p.volume
        comm = -self.commission_per_lot * p.volume / 2
        self._next_ticket += 1
        self._deals.append(SimpleNamespace(
            ticket=self._next_ticket + 500_000, order=self._next_ticket, time=self._now_ts(),
            type=self.DEAL_TYPE_SELL if p.type == self.POSITION_TYPE_BUY else self.DEAL_TYPE_BUY,
            entry=self.DEAL_ENTRY_OUT, position_id=p.ticket, volume=p.volume, price=price, commission=comm,
            swap=0.0, profit=profit, symbol=p.symbol, magic=p.magic, comment=comment, reason=reason))
        self.balance += profit + comm
        self._positions.pop(p.ticket, None)
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="Done", order=self._next_ticket,
                               deal=self._next_ticket + 500_000, volume=p.volume, price=price, request={})

    def advance(self, minutes: int = 1) -> None:
        for _ in range(int(minutes)):
            if self._ptr >= len(self._m1[self.symbols[0]]) - 1:
                return
            self._ptr += 1
            for p in list(self._positions.values()):
                bar = self._m1[p.symbol].iloc[self._ptr]
                hi, lo = float(bar["high"]), float(bar["low"])
                if p.type == self.POSITION_TYPE_BUY:
                    if p.sl and lo <= p.sl:
                        self._close(p, p.sl, "sl", reason=4)
                    elif p.tp and hi >= p.tp:
                        self._close(p, p.tp, "tp", reason=5)
                else:
                    sp = self._spec(p.symbol).ask - self._spec(p.symbol).bid
                    if p.sl and hi + sp >= p.sl:
                        self._close(p, p.sl, "sl", reason=4)
                    elif p.tp and lo + sp <= p.tp:
                        self._close(p, p.tp, "tp", reason=5)


def _epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1), dtype=np.int64)
