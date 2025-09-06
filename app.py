# app.py
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import faiss

from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.responses import RedirectResponse

# 路径设置
BASE = Path(__file__).parent
STORE = BASE / "index_store"

app = FastAPI(title="Complaint Search API")

# --- add these imports at top if not present ---
from fastapi import Response
from fastapi.responses import RedirectResponse

# 健康检查：Render 访问 /health 时返回 200
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

# 根路径重定向到 Swagger 文档
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# 给 Render/浏览器的 HEAD / 一个 200，避免日志里一堆 404
@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

# 浏览器的 favicon 请求，返回 204 即可
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# ---- 惰性加载资源：首次用到时才加载（并缓存）----
class Resources(BaseModel):
    meta: pd.DataFrame
    index: faiss.Index  # type: ignore
    model: SentenceTransformer

@lru_cache(maxsize=1)
def get_resources() -> Resources:
    # 读元数据（parquet）
    meta = pd.read_parquet(STORE / "meta.parquet")  # 需要 pyarrow

    # 读 faiss 索引
    index = faiss.read_index(str(STORE / "faiss.index"))

    # 读模型配置 + 加载模型
    cfg = json.load(open(STORE / "config.json"))
    model = SentenceTransformer(cfg["model"])  # 第一次会下载权重并缓存

    return Resources(meta=meta, index=index, model=model)

class SearchItem(BaseModel):
    business: str
    city: str
    stars: float
    text: str
    topic: int | None = None

class SearchResponse(BaseModel):
    items: list[SearchItem]

@app.get("/search_reviews", response_model=SearchResponse)
def search_reviews(query: str, city: str|None=None, category: str|None=None, top_k:int=5):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, top_k*10)
    cand = meta.iloc[I[0]].copy()
    if city:
        cand = cand[cand["city"].str.lower()==city.lower()]
    if category:
        cand = cand[cand["categories"].str.contains(category, case=False, na=False)]
    cand = cand.head(top_k)
    items = []
    for _, r in cand.iterrows():
        txt = r["text"][:280] + "…" if len(r["text"]) > 300 else r["text"]
        items.append({"business": r["name"], "city": r["city"], "stars": float(r["review_stars"]), "text": txt, "topic": None})
    return {"items": items}
