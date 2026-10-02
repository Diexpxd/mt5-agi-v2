"""Agente de Análisis Técnico (Quant)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from core.indicators import atr, ema, rsi, efficiency_ratio
from core.timeutils import session_name
from core.types import Direction, Signal, SymbolSpec

from . import smc
from .features import build_features, make_windows, training_set
from .models import TrainedModel, TrainReport, train_classifier

log = logging.getLogger(__name__)


@dataclass
class TechnicalConfig:
    seq_len: int = 48
    horizon: int = 12
    arch: str = "transformer"
    min_score: float = 0.40
    min_confluence: int = 2
    target_rr: float = 2.0
    min_model_edge: float = 0.03
    sl_min_atr: float = 1.0
    sl_max_atr: float = 3.0
    weights: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = dict(structure=0.15, trend=0.10, ob=0.35, sweep=0.30, fvg=0.10, wyckoff=0.20, ml=0.35)


class TechnicalAgent:
    """Analiza un DataFrame OHLCV y emite un ``Signal``."""

    name = "technical"

    def __init__(self, settings: Settings, config: TechnicalConfig | None = None) -> None:
        self.settings = settings
        self.cfg = config or TechnicalConfig()
        self.models: Dict[str, TrainedModel] = {}

    def _model_path(self, symbol: str, timeframe: str) -> Path:
        return self.settings.model_dir / f"{symbol}_{timeframe}_{self.cfg.arch}.pt"

    def train(self, symbol: str, df: pd.DataFrame, timeframe: str | None = None, epochs: int = 12,
              save: bool = True) -> TrainReport:
        X, y, _ = training_set(df, self.cfg.seq_len, self.cfg.horizon)
        tm = train_classifier(X, y, arch=self.cfg.arch, epochs=epochs, gap=self.cfg.horizon)
        self.models[symbol] = tm
        if save:
            tm.save(self._model_path(symbol, timeframe or self.settings.timeframe))
        return tm.report

    def load(self, symbol: str, timeframe: str | None = None) -> bool:
        path = self._model_path(symbol, timeframe or self.settings.timeframe)
        if not path.exists():
            return False
        try:
            self.models[symbol] = TrainedModel.load(path)
            return True
        except Exception as exc:  # archivo corrupto / versión distinta
            log.warning("No se pudo cargar el modelo %s: %s", path, exc)
            return False

    def model_trusted(self, symbol: str) -> bool:
        tm = self.models.get(symbol)
        return bool(tm and tm.report.trusted(self.cfg.min_model_edge))

    def model_score_series(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        tm = self.models.get(symbol)
        out = pd.Series(np.nan, index=df.index)
        if tm is None:
            return out
        feats = build_features(df)
        vals = feats.to_numpy(dtype=np.float32)
        ok = ~np.isnan(vals).any(axis=1)
        L = tm.seq_len
        if len(df) < L:
            return out
        wins = make_windows(np.nan_to_num(vals, nan=0.0), L)
        end_rows = np.arange(L - 1, len(df))
        csum = np.concatenate([[0], np.cumsum(ok.astype(int))])
        valid = (csum[end_rows + 1] - csum[end_rows + 1 - L]) == L
        p = tm.predict_proba(wins[valid])
        out.iloc[end_rows[valid]] = p[:, 2] - p[:, 0]
        return out

    def _model_score(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        tm = self.models.get(symbol)
        if tm is None or len(df) < tm.seq_len + 120:
            return None
        vals = build_features(df).to_numpy(dtype=np.float32)[-tm.seq_len:]
        if np.isnan(vals).any():
            return None
        p = tm.predict_proba(vals)
        return float(p[2] - p[0])

    def analyze(self, symbol: str, df: pd.DataFrame, spec: SymbolSpec | None = None,
                model_score: float | None = None) -> Signal:
        cfg = self.cfg
        if len(df) < 220:
            return Signal(symbol, Direction.FLAT, 0.0, source=self.name, rationale=["histórico insuficiente (<220 velas)"])
        d = df.iloc[-600:]
        c = d["close"]
        a_ser = atr(d, 14)
        a = a_ser.to_numpy()
        last_a = float(a[-1])
        price = float(c.iloc[-1])
        if not np.isfinite(last_a) or last_a <= 0:
            return Signal(symbol, Direction.FLAT, 0.0, source=self.name, rationale=["ATR no disponible"])

        e20, e50 = ema(c, 20), ema(c, 50)
        ms = smc.market_structure(d)
        obs = smc.order_blocks(d, a)
        fvgs = smc.fair_value_gaps(d, a)
        sweeps = smc.liquidity_sweeps(d, a)
        wy = smc.wyckoff_phase(d, a)
        r = float(rsi(c, 14).iloc[-1])
        er = float(efficiency_ratio(c, 20).iloc[-1])
        n = len(d)
        low_now, high_now = float(d["low"].iloc[-1]), float(d["high"].iloc[-1])

        comp: Dict[str, float] = {}
        why: List[str] = []

        comp["structure"] = float(ms["bias"])
        if ms["bias"]:
            why.append("estructura " + ("alcista (HH/HL)" if ms["bias"] > 0 else "bajista (LH/LL)"))
        trend = 0.0
        if e20.iloc[-1] > e50.iloc[-1] and price > e20.iloc[-1]:
            trend = 1.0
        elif e20.iloc[-1] < e50.iloc[-1] and price < e20.iloc[-1]:
            trend = -1.0
        comp["trend"] = trend
        if trend:
            why.append("tendencia " + ("alcista (EMA20>EMA50, precio>EMA20)" if trend > 0 else "bajista (EMA20<EMA50, precio<EMA20)"))

        # Order Block: la zona más reciente que el precio está tocando / muy cerca
        ob_score, ob_used = 0.0, None
        for z in sorted(obs, key=lambda z: -z.formed):
            touching = (low_now <= z.top + 0.3 * last_a) and (high_now >= z.bottom - 0.3 * last_a)
            if touching:
                s = z.direction * min(z.strength / 3.0, 1.0) * (0.7 if z.tested else 1.0)
                ob_score, ob_used = s, z
                break
        comp["ob"] = ob_score
        if ob_used:
            why.append(f"precio en OB {'demanda' if ob_used.direction > 0 else 'oferta'} "
                       f"[{ob_used.bottom:.5g}-{ob_used.top:.5g}]{' (ya testeado)' if ob_used.tested else ''}")

        sw = [s for s in sweeps if s["age"] <= 6]
        sw_score, sw_used = 0.0, None
        if sw:
            sw_used = max(sw, key=lambda s: s["strength"] * (1 - s["age"] / 8.0))
            sw_score = sw_used["direction"] * min(sw_used["strength"] / 1.5, 1.0) * (1 - sw_used["age"] / 8.0)
            why.append(f"barrido de liquidez {'de mínimos' if sw_used['direction'] > 0 else 'de máximos'} "
                       f"(hace {int(sw_used['age'])} velas)")
        comp["sweep"] = sw_score

        fvg_score = 0.0
        for z in sorted(fvgs, key=lambda z: -z.formed):
            if z.contains(price, 0.2 * last_a):
                fvg_score = float(z.direction)
                why.append(f"precio en FVG {'alcista' if z.direction > 0 else 'bajista'}")
                break
        comp["fvg"] = fvg_score

        comp["wyckoff"] = float(wy["bias"])
        if wy["phase"] not in ("unclear",):
            why.append(f"Wyckoff: {wy['phase']} ({'; '.join(wy['evidence'])})")

        ml_note = "modelo no disponible"
        ml = model_score if model_score is not None else self._model_score(symbol, d)
        if ml is not None and self.model_trusted(symbol):
            comp["ml"] = float(ml)
            ml_note = f"modelo {cfg.arch} p_up-p_down={ml:+.2f}"
            why.append(ml_note)
        elif ml is not None:
            ml_note = "modelo sin ventaja validada (ignorado)"

        w = cfg.weights
        score = float(np.clip(sum(w.get(k, 0.0) * v for k, v in comp.items()), -1.0, 1.0))
        direction = Direction.LONG if score > 0 else Direction.SHORT
        agree = sum(1 for k, v in comp.items() if abs(v) > 0.15 and np.sign(v) == np.sign(score))
        # el ATR y la fase de tendencia/rango describen el régimen para la memoria
        regime = "trend" if er > 0.35 else "range"
        meta = dict(
            atr=last_a, rsi=r, er=er, score=score, components={k: round(v, 3) for k, v in comp.items()},
            wyckoff=wy["phase"], structure=int(ms["bias"]), n_order_blocks=len(obs), ml_note=ml_note,
            regime=regime, session=session_name(d.index[-1].to_pydatetime()), agree=agree,
        )
        if abs(score) < cfg.min_score or agree < cfg.min_confluence:
            return Signal(symbol, Direction.FLAT, min(1.0, abs(score) / 0.8), entry=price, source=self.name,
                          rationale=why + [f"sin setup (|S|={abs(score):.2f}, confluencia={agree})"], meta=meta)

        sl, tp = self._levels(direction, price, last_a, ms, ob_used, sw_used)
        return Signal(symbol, direction, min(1.0, abs(score) / 0.8), entry=price, sl=sl, tp=tp, source=self.name,
                      rationale=why, meta=meta)

    def _levels(self, direction: Direction, price: float, a: float, ms: dict, ob, sw) -> tuple[float, float]:
        cfg = self.cfg
        cands: List[float] = []
        if direction == Direction.LONG:
            for lvl in (ms.get("last_swing_low"), ob.bottom if ob is not None and ob.direction > 0 else None,
                        sw["extreme"] if sw is not None and sw["direction"] > 0 else None):
                if lvl is not None and lvl < price - 0.3 * a:
                    cands.append(lvl)
            base = max(cands) if cands else price - 1.5 * a
            dist = np.clip(price - (base - 0.25 * a), cfg.sl_min_atr * a, cfg.sl_max_atr * a)
            sl = price - dist
            tp = price + cfg.target_rr * dist
        else:
            for lvl in (ms.get("last_swing_high"), ob.top if ob is not None and ob.direction < 0 else None,
                        sw["extreme"] if sw is not None and sw["direction"] < 0 else None):
                if lvl is not None and lvl > price + 0.3 * a:
                    cands.append(lvl)
            base = min(cands) if cands else price + 1.5 * a
            dist = np.clip((base + 0.25 * a) - price, cfg.sl_min_atr * a, cfg.sl_max_atr * a)
            sl = price + dist
            tp = price - cfg.target_rr * dist
        return float(sl), float(tp)
