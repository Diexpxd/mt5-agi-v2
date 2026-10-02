"""Tests de embeddings y almacenes vectoriales (Chroma y NumPy comparten contrato)."""
import uuid

import numpy as np
import pytest

from memory.embeddings import HashingEmbedder
from memory.vector_store import ChromaVectorStore, NumpyVectorStore, open_store

DOCS = [
    ("n1", "Fed signals rate hike as US inflation surges, dollar rallies", {"ts": 100, "src": "reuters"}),
    ("n2", "ECB holds rates steady, euro slips on weak eurozone growth", {"ts": 200, "src": "bloomberg"}),
    ("n3", "Gold jumps as safe haven demand rises amid geopolitical tension", {"ts": 300, "src": "forexlive"}),
    ("n4", "Bitcoin ETF inflows hit record as crypto rally continues", {"ts": 400, "src": "reuters"}),
]


def test_embedder_deterministic_normalized():
    e = HashingEmbedder(128)
    a, b = e.embed(["Fed hike"]), e.embed(["Fed hike"])
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert e.embed([""])[0].sum() == 0.0                     # texto vacío no rompe


def test_embedder_similarity_ordering():
    e = HashingEmbedder()
    v = e.embed(["gold safe haven demand", "gold jumps on safe haven demand", "central bank raises interest rates"])
    assert v[0] @ v[1] > v[0] @ v[2]


@pytest.fixture(params=["numpy", "chroma"])
def store(request, tmp_path):
    name = "t_" + uuid.uuid4().hex[:8]
    cls = NumpyVectorStore if request.param == "numpy" else ChromaVectorStore
    s = cls(name, HashingEmbedder(), tmp_path / request.param)
    s.add([d[0] for d in DOCS], [d[1] for d in DOCS], [d[2] for d in DOCS])
    return s


def test_query_returns_most_relevant(store):
    hits = store.query("gold safe haven", k=2)
    assert hits[0].id == "n3" and hits[0].score >= hits[1].score
    assert store.query("bitcoin crypto ETF", k=1)[0].id == "n4"


def test_where_filter_and_count(store):
    assert store.count() == 4
    hits = store.query("rates inflation", k=4, where={"ts": {"$gte": 200}})
    assert {h.id for h in hits} <= {"n2", "n3", "n4"} and "n1" not in {h.id for h in hits}
    assert {h.id for h in store.get_all(where={"src": "reuters"})} == {"n1", "n4"}


def test_upsert_delete_has(store):
    store.add(["n1"], ["Fed cuts rates unexpectedly"], [{"ts": 101, "src": "reuters"}])
    assert store.count() == 4 and "cuts" in store.query("Fed cuts rates", k=1)[0].text
    store.delete(["n2"])
    assert store.count() == 3 and not store.has("n2") and store.has("n1")


def test_empty_store_query_is_empty(tmp_path):
    for cls in (NumpyVectorStore, ChromaVectorStore):
        s = cls("empty_" + uuid.uuid4().hex[:6], HashingEmbedder(), tmp_path / cls.__name__)
        assert s.query("anything") == []


def test_numpy_persistence_roundtrip(tmp_path):
    s1 = NumpyVectorStore("p", HashingEmbedder(), tmp_path)
    s1.add(["a"], ["bitcoin rally"], [{"x": 1}])
    s2 = NumpyVectorStore("p", HashingEmbedder(), tmp_path)
    assert s2.count() == 1 and s2.query("bitcoin", 1)[0].id == "a"


def test_chroma_persistence_roundtrip(tmp_path):
    name = "persist_" + uuid.uuid4().hex[:6]
    s1 = ChromaVectorStore(name, HashingEmbedder(), tmp_path)
    s1.add(["a"], ["bitcoin rally"], [{"x": 1, "tags": ["a", "b"]}])     # las listas se serializan
    s1 = None
    s2 = ChromaVectorStore(name, HashingEmbedder(), tmp_path)
    assert s2.count() == 1 and s2.query("bitcoin", 1)[0].metadata["x"] == 1


def test_open_store_falls_back_to_numpy(monkeypatch, tmp_path):
    import memory.vector_store as vs

    def boom(*a, **k):
        raise RuntimeError("chroma roto")

    monkeypatch.setattr(vs, "ChromaVectorStore", boom)
    s = open_store("x", tmp_path)
    assert s.backend == "numpy"
    with pytest.raises(RuntimeError):
        open_store("x", tmp_path, backend="chroma")
