"""Spike API: one endpoint, dense retrieval over the in-memory sample index.

Throwaway. See issue #12. Not the real backend.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from sentence_transformers import SentenceTransformer

MODEL = "nomic-ai/nomic-embed-text-v1.5"
INDEX = Path(__file__).parent / "index"

app = FastAPI(title="Recipe retrieval spike", version="0.1.0")

_model: SentenceTransformer | None = None
_vectors: np.ndarray | None = None
_docs: list[dict] = []


@app.on_event("startup")
def load() -> None:
    global _model, _vectors, _docs
    if not (INDEX / "vectors.npy").exists():
        raise RuntimeError(f"no index at {INDEX} — run build_index.py first")
    _vectors = np.load(INDEX / "vectors.npy")
    _docs = [json.loads(line) for line in (INDEX / "docs.jsonl").read_text().splitlines()]
    _model = SentenceTransformer(MODEL, trust_remote_code=True)
    print(f"loaded {_vectors.shape[0]:,} vectors (dim {_vectors.shape[1]}) on {_model.device}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _vectors is not None else "no index",
        "documents": 0 if _vectors is None else int(_vectors.shape[0]),
        "dim": 0 if _vectors is None else int(_vectors.shape[1]),
        "device": str(_model.device) if _model else None,
    }


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=50)) -> dict:
    """Dense retrieval only. Cosine similarity over the whole sample — a brute-force
    scan, which is correct and fast enough at this scale and keeps the spike honest
    about what it is not (no ANN index, no BM25, no fusion, no reranking).
    """
    if _model is None or _vectors is None:
        raise HTTPException(503, "index not loaded")

    t0 = time.perf_counter()
    # Query prefix differs from the document prefix — see build_index.py.
    qv = _model.encode(
        [f"search_query: {q}"], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]

    scores = _vectors @ qv  # both sides L2-normalised, so this is cosine similarity
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    took_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": q,
        "arm": "dense",
        "took_ms": round(took_ms, 1),
        "results": [
            {
                "rank": rank,
                "score": round(float(scores[i]), 4),
                "title": _docs[i]["title"],
                "category": _docs[i].get("category"),
                "total_time": _docs[i].get("total_time"),
                "rating": _docs[i].get("rating"),
                "url": _docs[i].get("url"),
                "ingredients": (_docs[i].get("ingredients") or "")[:200],
            }
            for rank, i in enumerate(top, start=1)
        ],
    }
