# build_index.py
import os, json, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

PARQUET = "/home/jovyan/new folder/Analysis unstructure data Project/yelp_merged_reviews_business.parquet"   # 你的合并清洗文件
OUT_DIR  = "index_store"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 只保留低分抱怨评论（可按需调整阈值）
df = pd.read_parquet(PARQUET)
df = df[df["review_stars"] <= 2].copy()
keep_cols = ["business_id","name","city","categories","text","review_stars"]
df = df[keep_cols].dropna().reset_index(drop=True)

# 2) 生成向量（小而快；要更准可换 all-mpnet-base-v2）
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
emb = model.encode(df["text"].tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
emb = np.asarray(emb).astype("float32")

# 3) 建索引并保存
index = faiss.IndexFlatIP(emb.shape[1])            # 归一化向量 + 内积 = 余弦相似度
index.add(emb)
faiss.write_index(index, f"{OUT_DIR}/faiss.index")
df.to_parquet(f"{OUT_DIR}/meta.parquet", index=False)
with open(f"{OUT_DIR}/config.json","w") as f:
    json.dump({"model": model_name}, f)

print(f"Saved {len(df)} docs to {OUT_DIR}")