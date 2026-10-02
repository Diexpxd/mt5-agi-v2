"""Base de datos vectorial local: ChromaDB (persistente) con fallback NumPy."""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import Embedder, HashingEmbedder

log = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float            # similitud coseno en [-1, 1]


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:60]


def _match(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, ref in cond.items():
                if val is None:
                    return False
                if op == "$gte" and not val >= ref: return False
                if op == "$lte" and not val <= ref: return False
                if op == "$gt" and not val > ref: return False
                if op == "$lt" and not val < ref: return False
                if op == "$ne" and not val != ref: return False
                if op == "$in" and val not in ref: return False
        elif val != cond:
            return False
    return True


class NumpyVectorStore:
    """Almacén en memoria con persistencia opcional en JSON (embeddings incluidos)."""

    backend = "numpy"

    def __init__(self, name: str, embedder: Embedder, path: Path | None = None) -> None:
        self.embedder = embedder
        self._path = (path / f"{_safe_name(name)}.json") if path else None
        self._lock = threading.RLock()
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._vecs = np.zeros((0, embedder.dim), dtype=np.float32)
        if self._path and self._path.exists():
            try:
                blob = json.loads(self._path.read_text(encoding="utf-8"))
                self._ids, self._texts, self._metas = blob["ids"], blob["texts"], blob["metas"]
                self._vecs = np.array(blob["vecs"], dtype=np.float32).reshape(-1, embedder.dim)
            except Exception as exc:
                log.warning("Almacén NumPy corrupto (%s); se reinicia", exc)

    def _persist(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"ids": self._ids, "texts": self._texts, "metas": self._metas,
                                              "vecs": self._vecs.round(6).tolist()}), encoding="utf-8")

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not ids:
            return
        vecs = self.embedder.embed(texts)
        with self._lock:
            for i, id_ in enumerate(ids):
                if id_ in self._ids:                       # upsert
                    k = self._ids.index(id_)
                    self._texts[k], self._metas[k], self._vecs[k] = texts[i], metadatas[i], vecs[i]
                else:
                    self._ids.append(id_)
                    self._texts.append(texts[i])
                    self._metas.append(metadatas[i])
                    self._vecs = np.vstack([self._vecs, vecs[i][None]])
            self._persist()

    def query(self, text: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        with self._lock:
            if not self._ids:
                return []
            q = self.embedder.embed([text])[0]
            sims = self._vecs @ q
            order = np.argsort(-sims)
            hits: List[Hit] = []
            for i in order:
                if _match(self._metas[i], where):
                    hits.append(Hit(self._ids[i], self._texts[i], self._metas[i], float(sims[i])))
                    if len(hits) >= k:
                        break
            return hits

    def get_all(self, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        with self._lock:
            return [Hit(i, t, m, 0.0) for i, t, m in zip(self._ids, self._texts, self._metas) if _match(m, where)]

    def has(self, id_: str) -> bool:
        return id_ in self._ids

    def count(self) -> int:
        return len(self._ids)

    def delete(self, ids: List[str]) -> None:
        with self._lock:
            keep = [i for i, x in enumerate(self._ids) if x not in set(ids)]
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._metas = [self._metas[i] for i in keep]
            self._vecs = self._vecs[keep] if keep else np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._persist()


class ChromaVectorStore:
    """Colección ChromaDB persistente con métrica coseno (embeddings calculados localmente)."""

    backend = "chroma"

    def __init__(self, name: str, embedder: Embedder, path: Path | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embedder = embedder
        cs = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=cs) if path else chromadb.EphemeralClient(settings=cs)
        self._col = self._client.get_or_create_collection(
            name=_safe_name(f"{name}_{embedder.name}{embedder.dim}"), metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _clean(meta: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            out[k] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
        return out

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not ids:
            return
        self._col.upsert(ids=ids, documents=texts, embeddings=self.embedder.embed(texts).tolist(),
                         metadatas=[self._clean(m) or {"_": 1} for m in metadatas])

    def query(self, text: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=self.embedder.embed([text]).tolist(), n_results=min(k, n),
                              where=where or None, include=["documents", "metadatas", "distances"])
        hits = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(id_, doc, meta or {}, 1.0 - float(dist)))
        return hits

    def get_all(self, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        res = self._col.get(where=where or None, include=["documents", "metadatas"])
        return [Hit(i, d, m or {}, 0.0) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def has(self, id_: str) -> bool:
        return bool(self._col.get(ids=[id_])["ids"])

    def count(self) -> int:
        return self._col.count()

    def delete(self, ids: List[str]) -> None:
        self._col.delete(ids=ids)


def open_store(name: str, chroma_dir: Path | None, embedder: Embedder | None = None, backend: str = "auto"):
    """Abre un almacén vectorial."""
    embedder = embedder or HashingEmbedder()
    if backend in ("auto", "chroma"):
        try:
            return ChromaVectorStore(name, embedder, chroma_dir)
        except Exception as exc:
            if backend == "chroma":
                raise
            log.warning("ChromaDB no disponible (%s): usando almacén NumPy local", exc)
    return NumpyVectorStore(name, embedder, chroma_dir)
