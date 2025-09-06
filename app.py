# app.py
import json, numpy as np, pandas as pd, faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

STORE = "index_store"
meta  = pd.read_parquet(f"{STORE}/meta.parquet")
index = faiss.read_index(f"{STORE}/faiss.index")
cfg   = json.load(open(f"{STORE}/config.json"))
model = SentenceTransformer(cfg["model"])

app = FastAPI(title="Complaint Search API")

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
