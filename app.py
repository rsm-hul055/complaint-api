# app.py
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.responses import Response

# --------------------
# Paths & App
# --------------------
BASE = Path(__file__).parent
STORE = BASE / "index_store"   # expects: meta.parquet, faiss.index, config.json

app = FastAPI(title="Complaint Search API", version="0.1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("complaint-api")

# --------------------
# Health & Warmup
# --------------------
@app.get("/", include_in_schema=False)
def root():
    # return 200 directly (avoid 307 redirects)
    return {"status": "ok", "message": "Complaint API is running."}

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

@app.get("/healthz", include_in_schema=False)
def healthz():
    # lightweight health (no heavy work)
    return {"status": "healthy"}

@app.get("/warmup", include_in_schema=False)
def warmup():
    # preload resources once; helpful after cold start
    try:
        _ = get_resources()
        return {"status": "warmed"}
    except Exception as e:
        logger.exception("Warmup failed: %s", e)
        raise HTTPException(status_code=500, detail=f"warmup failed: {e}")

# --------------------
# Lazy resources (load once)
# --------------------
class _Resources:
    def __init__(self, meta: pd.DataFrame, index: faiss.Index, model: SentenceTransformer):
        self.meta = meta
        self.index = index
        self.model = model

@lru_cache(maxsize=1)
def get_resources() -> _Resources:
    try:
        # 1) metadata (expects columns: text, name, city, review_stars, categories)
        meta_path = STORE / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found")
        meta = pd.read_parquet(meta_path)

        # 2) FAISS index
        index_path = STORE / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found")
        index = faiss.read_index(str(index_path))

        # 3) sentence-transformers model (should already be cached on disk)
        cfg_path = STORE / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_name_or_path = cfg.get("model", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name_or_path)

        logger.info("Resources loaded: meta=%s rows, index ntotal=%s, model=%s",
                    len(meta), getattr(index, 'ntotal', 'n/a'), model_name_or_path)
        return _Resources(meta=meta, index=index, model=model)
    except Exception as e:
        # surface as 500 instead of silent 502
        raise RuntimeError(f"Resource loading failed: {e}")

# --------------------
# Pydantic response models
# --------------------
class SearchItem(BaseModel):
    business: str
    city: str
    stars: float
    text: str
    topic: Optional[int] = None  # reserved for future use

class SearchResponse(BaseModel):
    items: List[SearchItem]

# --------------------
# Search endpoint
# --------------------
@app.get("/search_reviews", response_model=SearchResponse, summary="Search reviews")
def search_reviews(
    query: str = Query(..., description="What you want to find, e.g. 'waiting time'"),
    city: Optional[str] = Query(None, description="Filter by city (optional)"),
    category: Optional[str] = Query(None, description="Filter by category keyword (optional)"),
    top_k: int = Query(5, ge=1, le=50, description="How many results to return (1–50)"),
):
    try:
        R = get_resources()

        # encode query (normalize for cosine via dot)
        qv = R.model.encode([query], normalize_embeddings=True).astype(np.float32)
        search_k = max(top_k * 10, 50)  # oversample for later filtering
        D, I = R.index.search(qv, search_k)

        # guard: empty index
        idxs = I[0]
        if idxs is None or len(idxs) == 0:
            return SearchResponse(items=[])

        cand = R.meta.iloc[idxs].copy()

        if city:
            cand = cand[cand.get("city", "").str.lower() == city.lower()]

        if category:
            cand = cand[cand.get("categories", "").str.contains(category, case=False, na=False)]

        cand = cand.head(top_k)

        items: List[SearchItem] = []
        for _, r in cand.iterrows():
            txt = str(r.get("text", "") or "")
            if len(txt) > 300:
                txt = txt[:280] + "…"
            items.append(
                SearchItem(
                    business=str(r.get("name", "") or ""),
                    city=str(r.get("city", "") or ""),
                    stars=float(r.get("review_stars", 0.0) or 0.0),
                    text=txt,
                    topic=None,
                )
            )
        return SearchResponse(items=items)
    except Exception as e:
        logger.exception("search_reviews failed: %s", e)
        raise HTTPException(status_code=500, detail=f"search_reviews failed: {e}")



