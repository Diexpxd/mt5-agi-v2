"""Embeddings de texto locales, deterministas y sin descargas (funcionan 100 % offline)."""
from __future__ import annotations

import logging
import math
import re
import zlib
from collections import Counter
from typing import List, Protocol

import numpy as np

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-záéíóúñü0-9_\-\.\+%]+", re.IGNORECASE)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Embedder léxico determinista (no requiere modelo ni red)."""

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> Counter:
        words = [w.strip(".-") for w in _TOKEN.findall(text.lower())]
        words = [w for w in words if w]
        feats: Counter = Counter()
        for w in words:
            feats["w:" + w] += 1
            padded = f"<{w}>"
            for i in range(len(padded) - 2):
                feats["c:" + padded[i:i + 3]] += 0.35
        for a, b in zip(words, words[1:]):
            feats[f"b:{a}_{b}"] += 1
        return feats

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for f, tf in self._features(text).items():
                h = zlib.crc32(f.encode("utf-8"))
                out[r, h % self.dim] += (1.0 if (h >> 31) & 1 else -1.0) * (1.0 + math.log(1.0 + tf))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class GeminiEmbedder:
    """Embeddings de Gemini (opcional)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768) -> None:
        from google import genai  # import diferido: solo si se usa

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim
        self._fallback = HashingEmbedder(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        try:
            res = self._client.models.embed_content(model=self._model, contents=texts,
                                                    config={"output_dimensionality": self.dim})
            arr = np.array([e.values for e in res.embeddings], dtype=np.float32)
            n = np.linalg.norm(arr, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return arr / n
        except Exception as exc:
            log.warning("Gemini embeddings falló (%s); usando hashing local", exc)
            return self._fallback.embed(texts)
