"""Tests del agente técnico: causalidad de features, modelos y señales."""
import numpy as np
import pandas as pd
import pytest

from agents.features import build_features, make_labels, training_set, N_FEATURES
from agents.models import TrainedModel, TrainReport, train_classifier
from agents.technical_agent import TechnicalAgent, TechnicalConfig
from core.synthetic import generate_ohlcv
from core.types import Direction


@pytest.fixture(scope="module")
def m15():
    return generate_ohlcv("EURUSD", n=9000, minutes=15, seed=3)


def test_synthetic_is_deterministic_and_valid():
    a = generate_ohlcv("EURUSD", n=500, minutes=15, seed=5)
    b = generate_ohlcv("EURUSD", n=500, minutes=15, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all() and (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert a.index.tz is not None and (a["close"] > 0).all()


def test_features_are_causal_no_lookahead(m15):
    df = m15.iloc[:1500]
    full = build_features(df)
    cut = 900
    part = build_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-9, atol=1e-9)
    assert full.shape[1] == N_FEATURES


def test_labels_use_future_and_mark_tail(m15):
    y = make_labels(m15.iloc[:600], horizon=12)
    assert (y[-12:] == -1).all()                     # sin futuro disponible
    assert (y[:13] == -1).all()                      # ATR aún sin calentar
    assert set(np.unique(y[100:-12])) == {0, 1, 2}   # el resto tiene las 3 clases


def test_training_windows_align_with_labels(m15):
    X, y, rows = training_set(m15.iloc[:2000], seq_len=48, horizon=12)
    assert X.shape[1:] == (48, N_FEATURES) and len(X) == len(y) == len(rows)
    assert not np.isnan(X).any()
    f = build_features(m15.iloc[:2000]).to_numpy(dtype=np.float32)
    np.testing.assert_allclose(X[10, -1], f[rows[10]], rtol=1e-5)


@pytest.mark.parametrize("arch", ["transformer", "lstm"])
def test_model_trains_saves_loads_predicts(m15, arch, tmp_path):
    X, y, _ = training_set(m15, 48, 12)
    tm = train_classifier(X[:3000], y[:3000], arch=arch, epochs=3, seed=1)
    assert isinstance(tm.report, TrainReport) and tm.report.n_val > 300
    assert 0.0 <= tm.report.val_acc <= 1.0
    p = tm.predict_proba(X[:5])
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    path = tmp_path / "m.pt"
    tm.save(path)
    tm2 = TrainedModel.load(path)
    np.testing.assert_allclose(tm2.predict_proba(X[:5]), p, atol=1e-5)


def test_training_is_reproducible(m15):
    X, y, _ = training_set(m15, 48, 12)
    a = train_classifier(X[:1500], y[:1500], arch="lstm", epochs=2, seed=9)
    b = train_classifier(X[:1500], y[:1500], arch="lstm", epochs=2, seed=9)
    assert a.report.val_acc == pytest.approx(b.report.val_acc)


def test_purged_split_leaves_gap():
    X = np.random.default_rng(0).normal(size=(1000, 8, 3)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, 1000)
    tm = train_classifier(X, y, arch="lstm", epochs=1, gap=12, val_frac=0.25)
    assert tm.report.n_train + tm.report.n_val <= 1000 - 12 + 1


def test_untrained_short_history_returns_flat(settings, m15):
    ag = TechnicalAgent(settings)
    sig = ag.analyze("EURUSD", m15.iloc[:100])
    assert sig.direction == Direction.FLAT


def test_analyze_produces_consistent_levels(settings, m15):
    ag = TechnicalAgent(settings, TechnicalConfig(min_score=0.05, min_confluence=1))
    seen = 0
    for end in range(1000, 8000, 37):
        sig = ag.analyze("EURUSD", m15.iloc[:end])
        if sig.direction == Direction.FLAT:
            continue
        seen += 1
        risk = abs(sig.entry - sig.sl)
        assert risk > 0 and abs(sig.tp - sig.entry) == pytest.approx(2.0 * risk, rel=1e-6)
        if sig.direction == Direction.LONG:
            assert sig.sl < sig.entry < sig.tp
        else:
            assert sig.tp < sig.entry < sig.sl
        assert 0.0 <= sig.confidence <= 1.0 and sig.rationale
        assert sig.meta["atr"] > 0 and (1.0 * sig.meta["atr"] - 1e-9) <= risk <= (3.0 * sig.meta["atr"] + 1e-9)
    assert seen > 5, "con umbrales laxos debe emitir señales en datos con regímenes"


def test_untrusted_model_is_ignored(settings, m15):
    ag = TechnicalAgent(settings)
    X, y, _ = training_set(m15, 48, 12)
    tm = train_classifier(X[:1500], y[:1500], arch="transformer", epochs=1, seed=2)
    tm.report.edge = -0.05                                        # sin ventaja
    ag.models["EURUSD"] = tm
    sig = ag.analyze("EURUSD", m15.iloc[:3000])
    assert "ml" not in sig.meta["components"] and "ignorado" in sig.meta["ml_note"]
    tm.report.edge, tm.report.n_dir = 0.40, 1000                  # ventaja validada
    sig2 = ag.analyze("EURUSD", m15.iloc[:3000])
    assert "ml" in sig2.meta["components"]


def test_model_score_series_matches_single_inference(settings, m15):
    ag = TechnicalAgent(settings)
    X, y, _ = training_set(m15, 48, 12)
    ag.models["EURUSD"] = train_classifier(X[:1500], y[:1500], arch="lstm", epochs=1, seed=2)
    ser = ag.model_score_series("EURUSD", m15.iloc[:1200])
    single = ag._model_score("EURUSD", m15.iloc[:1200])
    assert ser.iloc[-1] == pytest.approx(single, abs=1e-5)
    assert ser.iloc[:100].isna().all() and not ser.iloc[-1:].isna().any()


def test_trust_requires_statistical_significance():
    base = dict(arch="lstm", n_train=1000, n_val=1000, val_acc=0.4, val_loss=1.0, baseline_acc=0.6, epochs_run=3)
    weak = TrainReport(**base, edge=0.06, n_dir=1900, overlap=12)       # SE_efectivo = 0.5/sqrt(158) = 0.040 -> exige 0.08
    strong = TrainReport(**base, edge=0.16, n_dir=1900, overlap=12)
    tiny = TrainReport(**base, edge=0.30, n_dir=100, overlap=12)         # muestra insuficiente
    assert not weak.trusted() and strong.trusted() and not tiny.trusted()


def test_model_learns_when_signal_exists():
    df = generate_ohlcv("EURUSD", n=12000, minutes=15, seed=21, trend_strength=0.6, mean_regime_bars=600)
    X, y, _ = training_set(df, 48, 12)
    tm = train_classifier(X, y, arch="lstm", epochs=10, seed=5)
    assert tm.report.edge > 0.15 and tm.report.trusted()


def test_no_edge_on_pure_noise_is_not_trusted():
    df = generate_ohlcv("EURUSD", n=12000, minutes=15, seed=21, trend_strength=0.0)
    X, y, _ = training_set(df, 48, 12)
    tm = train_classifier(X, y, arch="lstm", epochs=6, seed=5)
    assert not tm.report.trusted()
