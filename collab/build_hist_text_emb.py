#!/usr/bin/env python3
"""History-text embeddings for the motive retriever (the 'text' feature), for datasets that lack them.

Same text and encoder as collab/tune_latent.py (embed_text / text_cache_name) + collab/latent_hyde.py (history_text):
"; "-joined history titles, Alibaba-NLP/gte-large-en-v1.5, normalised.
Writes data/<ds>/collab/hyp_cache/hist_gte-large-en-v1.5{,_valid,_test}.npy; skips files that exist.
"""
import argparse
from pathlib import Path

import numpy as np

from build_support import find_csv, load_split

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data-root", type=Path, default=Path("data"))
p.add_argument("--encoder", default="Alibaba-NLP/gte-large-en-v1.5")
# the snapshot that built the Toys/CDs caches (cos 1.0000 on CDs); the hub's current main is model_type "gte",
# which the rrec env's transformers cannot load
p.add_argument("--revision", default="104333d6af6f97649377c2afbde10a7704870c7b")
a = p.parse_args()
cache = a.data_root / a.dataset / "collab" / "hyp_cache"
cache.mkdir(parents=True, exist_ok=True)
stem = "hist_" + a.encoder.split("/")[-1]
model = None
for s in ("train", "valid", "test"):
    path = cache / (f"{stem}.npy" if s == "train" else f"{stem}_{s}.npy")
    if path.exists():
        print(f"{path} exists"); continue
    frame = load_split(find_csv(a.data_root, a.dataset, s))
    texts = ["; ".join(eval(t)) for t in frame.history_item_title]
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.encoder, device="cuda", trust_remote_code=True, revision=a.revision)
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(path, emb)
    print(f"{s}: {emb.shape} -> {path}", flush=True)
