# app.py
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.responses import RedirectResponse

# --------------------
# 路径设置
# --------------------
BASE = Path(__file__).parent
STORE = BASE / "index_store"

# 注意：arbitrary_types_allowed 不再放在 FastAPI 里（那是 Pydantic 的配置）
app = FastAPI(title="Complaint Search API")

# --------------------
# 健康检查 & 根路径
# --------------------
@app.get("/health", include_in_schema=False)
def health():
    """给 Render 的健康检查"""
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    """根路径跳到 Swagger 文档"""
    return RedirectResponse(url="/docs")

@app.head("/", include_in_schema=False)
def root_head():
    """HEAD / 返回 200，避免 Render 日志里反复 404"""
    return Response(status_code=200)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """浏览器会自动请求 /favicon.ico，这里返回 204 即可"""
    return Response(status_code=204)

# --------------------
# 惰性加载资源（只在首次调用时加载并缓存）
# --------------------
class _Resources:
    """内部使用的资源容器（不是 Pydantic 模型，避免 schema 报错）"""
    def __init__(self, meta: pd.DataFrame, index: faiss.Index, model: SentenceTransformer):
        self.meta = meta
        self.index = index
        self.model = model

@lru_cache(maxsize=1)
def get_resources() -> _Resources:
    # 1) 读元数据（需要 pyarrow 或 fastparquet；推荐 pyarrow）
    meta = pd.read_parquet(STORE / "meta.parquet")

    # 2) 读 FAISS 索引
    index = faiss.read_index(str(STORE / "faiss.index"))

    # 3) 读模型配置并加载模型
    with open(STORE / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = SentenceTransformer(cfg["model"])  # 第一次会下载权重并缓存

    return _Resources(meta=meta, index=index, model=model)

# --------------------
# Pydantic 响应模型（只包含基础类型）
# --------------------
class SearchItem(BaseModel):
    business: str
    city: str
    stars: float
    text: str
    topic: Optional[int] = None  # 预留

class SearchResponse(BaseModel):
    items: List[SearchItem]

# --------------------
# 搜索接口
# --------------------
@app.get("/search_reviews", response_model=SearchResponse, summary="Search reviews")
def search_reviews(
    query: str = Query(..., description="What you want to find, e.g. 'waiting time'"),
    city: Optional[str] = Query(None, description="Filter by city (optional)"),
    category: Optional[str] = Query(None, description="Filter by category keyword (optional)"),
    top_k: int = Query(5, ge=1, le=50, description="How many results to return（1-50）"),
):
    R = get_resources()

    # 编码查询向量
    qv = R.model.encode([query], normalize_embeddings=True).astype(np.float32)
    # 用更大的候选数以便过滤后还能凑够 top_k
    search_k = max(top_k * 10, 50)
    D, I = R.index.search(qv, search_k)

    # 取候选并按可选条件过滤
    cand = R.meta.iloc[I[0]].copy()

    if city:
        cand = cand[cand["city"].str.lower() == city.lower()]

    if category:
        # categories 字段包含多个分类，用 contains 模糊匹配（忽略大小写）
        cand = cand[cand["categories"].str.contains(category, case=False, na=False)]

    # 只要前 top_k 条（如果过滤后不足，就返回实际数量）
    cand = cand.head(top_k)

    # 组装输出
    items: List[SearchItem] = []
    for _, r in cand.iterrows():
        txt = r["text"]
        if isinstance(txt, str) and len(txt) > 300:
            txt = txt[:280] + "…"
        items.append(
            SearchItem(
                business=str(r.get("name", "")),
                city=str(r.get("city", "")),
                stars=float(r.get("review_stars", 0.0)),
                text=txt if isinstance(txt, str) else "",
                topic=None,
            )
        )

    return SearchResponse(items=items)



#不再把 pandas.DataFrame / faiss.Index / SentenceTransformer 放进 Pydantic 模型（这会让 OpenAPI/Pydantic 生成 schema 时直接报错）。

#资源（索引、元数据、模型）通过 get_resources() 惰性加载 + 缓存，端点每次调用直接拿来用。

#响应只返回基本类型（字符串/数字/列表），Swagger UI 兼容好。

#/ 自动跳 /docs，/health 给 Render 用，HEAD / 和 /favicon.ico 避免日志刷 404。