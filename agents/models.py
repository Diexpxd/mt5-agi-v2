"""Modelos predictivos temporales (PyTorch): Transformer encoder y LSTM + entrenador con validación purgada."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


class TemporalTransformer(nn.Module):
    """Encoder Transformer sobre ventanas de ``seq_len`` barras."""

    def __init__(self, n_features: int, seq_len: int, d_model: int = 32, nhead: int = 4, layers: int = 2,
                 ff: int = 64, dropout: float = 0.1, n_classes: int = 3) -> None:
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.enc(self.inp(x) + self.pos))
        return self.head(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))


class LSTMClassifier(nn.Module):
    """LSTM de ``layers`` capas; clasifica con el último estado oculto."""

    def __init__(self, n_features: int, seq_len: int, hidden: int = 48, layers: int = 1, dropout: float = 0.1,
                 n_classes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def build_model(arch: str, n_features: int, seq_len: int) -> nn.Module:
    if arch == "transformer":
        return TemporalTransformer(n_features, seq_len)
    if arch == "lstm":
        return LSTMClassifier(n_features, seq_len)
    raise ValueError(f"Arquitectura desconocida: {arch}")


@dataclass
class TrainReport:
    arch: str
    n_train: int
    n_val: int
    val_acc: float
    val_loss: float
    baseline_acc: float          # exactitud 3 clases de predecir siempre la clase mayoritaria (informativo)
    edge: float                  # dir_acc - dir_baseline: ventaja DIRECCIONAL sobre la línea base
    epochs_run: int
    class_dist_val: List[float] = field(default_factory=list)
    trained_at: str = ""
    dir_acc: float = 0.0
    dir_baseline: float = 0.5    # 0.5 (apostar siempre a un lado)
    n_dir: int = 0               # muestras de validación con movimiento direccional
    overlap: int = 12            # solape de etiquetas (horizonte): reduce la muestra efectiva

    def trusted(self, min_edge: float = 0.03, min_dir: int = 300) -> bool:
        eff_n = max(self.n_dir / max(self.overlap, 1), 1.0)
        se = 0.5 / math.sqrt(eff_n)
        return self.n_dir >= min_dir and self.edge >= max(min_edge, 2.0 * se)


@dataclass
class TrainedModel:
    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    report: TrainReport
    seq_len: int
    n_features: int
    arch: str

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        single = X.ndim == 2
        if single:
            X = X[None]
        self.model.eval()
        out = []
        for i in range(0, len(X), batch):
            xb = (torch.tensor(X[i:i + batch], dtype=torch.float32) - self.mean) / self.std   # tensor() copia: X puede ser de solo lectura
            out.append(torch.softmax(self.model(xb), dim=-1).numpy())
        p = np.concatenate(out) if out else np.zeros((0, 3))
        return p[0] if single else p

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.model.state_dict(), "mean": self.mean, "std": self.std, "seq_len": self.seq_len,
            "n_features": self.n_features, "arch": self.arch, "report": asdict(self.report),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(blob["arch"], blob["n_features"], blob["seq_len"])
        model.load_state_dict(blob["state"])
        model.eval()
        return cls(model, blob["mean"], blob["std"], TrainReport(**blob["report"]), blob["seq_len"],
                   blob["n_features"], blob["arch"])


def train_classifier(X: np.ndarray, y: np.ndarray, arch: str = "transformer", epochs: int = 12, batch: int = 128,
                     lr: float = 2e-3, patience: int = 3, val_frac: float = 0.25, gap: int = 12,
                     seed: int = 42) -> TrainedModel:
    """Entrena con partición temporal purgada (walk-forward de un solo pliegue)."""
    if len(X) < 400:
        raise ValueError(f"Muestras insuficientes para entrenar ({len(X)} < 400)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(X)
    n_train = int(n * (1 - val_frac))
    tr = slice(0, n_train)
    va = slice(min(n_train + gap, n - 1), n)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    flat = Xtr.reshape(-1, X.shape[-1])
    mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(flat.std(axis=0) + 1e-6, dtype=torch.float32)
    xtr = (torch.as_tensor(Xtr) - mean) / std
    xva = (torch.as_tensor(Xva) - mean) / std
    ytr_t, yva_t = torch.as_tensor(ytr), torch.as_tensor(yva)

    counts = np.bincount(ytr, minlength=3).astype(float)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    model = build_model(arch, X.shape[-1], X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best, best_state, bad, epochs_run = math.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(xva), yva_t))
        epochs_run = ep + 1
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(xva)
        acc = float((logits.argmax(-1) == yva_t).float().mean())
        vloss = float(nn.functional.cross_entropy(logits, yva_t))
    majority = int(np.argmax(counts))
    baseline = float((yva == majority).mean())
    probs = torch.softmax(logits, dim=-1).numpy()
    moved = yva != 1
    n_dir = int(moved.sum())
    dir_acc, dir_base = 0.5, 0.5
    if n_dir:
        pred_up = probs[moved, 2] > probs[moved, 0]
        true_up = yva[moved] == 2
        if true_up.any() and (~true_up).any():
            dir_acc = float((pred_up[true_up].mean() + (~pred_up[~true_up]).mean()) / 2.0)
        else:
            dir_acc = float((pred_up == true_up).mean())
            dir_base = float(max(true_up.mean(), 1 - true_up.mean()))
    report = TrainReport(
        arch=arch, n_train=len(Xtr), n_val=len(Xva), val_acc=acc, val_loss=vloss, baseline_acc=baseline,
        edge=dir_acc - dir_base, epochs_run=epochs_run,
        class_dist_val=[float((yva == k).mean()) for k in range(3)],
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dir_acc=dir_acc, dir_baseline=dir_base, n_dir=n_dir, overlap=gap,
    )
    log.info("Modelo %s: dir_acc=%.3f base=%.3f edge=%+.3f n_dir=%d (%d épocas)", arch, dir_acc, dir_base,
             report.edge, n_dir, epochs_run)
    return TrainedModel(model, mean, std, report, X.shape[1], X.shape[-1], arch)
