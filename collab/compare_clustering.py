#!/usr/bin/env python3
"""Compare clustering methods for building the rationale prototype corpus.

Spaces (unit + embedding):
  rationale/mpnet   whole train rationales (deduplicated per user), mpnet
  phrase/mpnet      the 1-2 `motives` phrases of each rationale, mpnet
  phrase/qwen3      motive phrases, Qwen3-Embedding-8B with a motive instruction
Methods: spherical k-means (P clusters), HDBSCAN on PCA-50, kNN graph + Louvain
(resolution tuned to ~P communities).

Rows whose target title is broken HTML are left out, and each (user, text) pair
counts once, so a single user cannot form a cluster. Metrics:
  clusters / noise    number of clusters, share of units HDBSCAN leaves unassigned
  coherence_norm      within-cluster mean pairwise cosine, normalised by random pairs
  purity              median share of a cluster's member rows carrying its top facet
  ambiguous           share of units whose two nearest centroids differ by < 0.05 cosine
  near_dup            cluster pairs with centroid cosine > 0.9, per cluster
  stability_ari       ARI between runs on two different 90% subsamples (all units
                      assigned to the nearest centroid of each run)
  coverage tr/te/gap  row-level cosine to the nearest centroid (a phrase-level row
                      averages over its phrases); gap = train - test (leakage proxy)
Coverage and coherence are only comparable within one embedding space.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from build_prototypes import spherical_kmeans

SPLITS = ("train", "valid", "test")
QWEN_PROMPT = "Instruct: Represent the buyer's purchase motive expressed in this phrase\nQuery: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--P", type=int, default=1000)
    parser.add_argument("--spaces", nargs="+", default=["rationale/mpnet", "phrase/mpnet", "phrase/qwen3"])
    parser.add_argument("--methods", nargs="+", default=["kmeans", "hdbscan", "louvain"])
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def encode(texts: list, model: str, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if "Qwen3" in model else {}
    st = SentenceTransformer(model, device="cuda", **kwargs)
    emb = st.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True,
                    prompt=QWEN_PROMPT if "Qwen3" in model else None).astype(np.float32)
    del st
    torch.cuda.empty_cache()
    np.save(cache, emb)
    return emb


def load_space(args, space: str) -> dict:
    """Units to cluster (deduplicated train texts) and row-level embeddings for coverage."""
    unit, model_name = space.split("/")
    collab = args.data_root / args.dataset / "collab"
    cache = collab / "hyp_cache"
    info = [l.split("\t") for l in open(next((args.data_root / args.dataset / "info").glob("*.txt")))]
    broken = {int(x[1]) for x in info if x[0].lstrip('"').startswith("<")}
    frames = {s: pd.read_csv(next((args.data_root / args.dataset / s).glob("*.csv"))) for s in SPLITS}
    recs = {}
    for s in SPLITS:
        recs[s] = [None] * len(frames[s])
        for line in open(collab / f"rationales_facet_{s}_{args.rationale_tag}.jsonl"):
            r = json.loads(line)
            recs[s][r["row"]] = r

    def unit_texts(r):
        return [r["reason"]] if unit == "rationale" else [m.strip().lower() for m in r["motives"]]

    # Embeddings for every distinct text across splits.
    all_texts = sorted({t for s in SPLITS for r in recs[s] for t in unit_texts(r)})
    model = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "qwen3": "Qwen/Qwen3-Embedding-8B"}[model_name]
    emb = encode(all_texts, model, cache / f"cluster_{unit}_{model_name}.npy")
    text_id = {t: i for i, t in enumerate(all_texts)}

    # Train units: distinct texts after (user, text) dedup and broken-target removal.
    train = frames["train"]
    seen, unit_rows = set(), collections.defaultdict(list)
    for row, r in enumerate(recs["train"]):
        if train.item_id.iat[row] in broken:
            continue
        for t in unit_texts(r):
            if (train.user_id.iat[row], t) not in seen:
                seen.add((train.user_id.iat[row], t))
                unit_rows[t].append(row)
    units = sorted(unit_rows)
    raw = emb[[text_id[t] for t in units]]
    mean = raw.mean(0)
    x = normalise(raw - mean)
    rows_emb = {s: [normalise(emb[[text_id[t] for t in unit_texts(r)]] - mean) for r in recs[s]] for s in SPLITS}
    return {"units": units, "x": x, "unit_rows": [unit_rows[t] for t in units], "recs": recs,
            "rows_emb": rows_emb, "mean": mean}


def run_method(method: str, x: np.ndarray, P: int, knn: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        c = spherical_kmeans(torch.from_numpy(x).cuda(), P, 50, seed)
        return (torch.from_numpy(x).cuda() @ c.T).argmax(1).cpu().numpy()
    if method == "hdbscan":
        z = normalise(PCA(50, random_state=seed).fit_transform(x))
        return HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(z)
    # kNN graph + Louvain; bisect the resolution towards ~P communities.
    import faiss  # rrec env only; the Qwen3 encoding step runs in racer, which has no faiss
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sims, nbrs = index.search(x, knn + 1)
    g = nx.Graph()
    g.add_nodes_from(range(len(x)))
    for i in range(len(x)):
        for s, j in zip(sims[i, 1:], nbrs[i, 1:]):
            if s > 0:
                g.add_edge(i, int(j), weight=float(s))
    lo, hi, best = 0.5, 200.0, None
    for _ in range(8):
        res = (lo * hi) ** 0.5
        comms = nx.community.louvain_communities(g, weight="weight", resolution=res, seed=seed)
        if best is None or abs(len(comms) - P) < abs(len(best) - P):
            best = comms
        lo, hi = (res, hi) if len(comms) < P else (lo, res)
    labels = np.empty(len(x), dtype=int)
    for k, comm in enumerate(best):
        labels[list(comm)] = k
    return labels


def centroids(x: np.ndarray, labels: np.ndarray) -> tuple:
    ids = np.unique(labels[labels >= 0])
    c = np.stack([x[labels == k].mean(0) for k in ids])
    return normalise(c), ids


def evaluate(space: dict, labels: np.ndarray, method: str, args) -> dict:
    x, recs = space["x"], space["recs"]
    c, ids = centroids(x, labels)
    rng = np.random.default_rng(args.seed)
    pairs = rng.integers(0, len(x), (100000, 2))
    rand = float((x[pairs[:, 0]] * x[pairs[:, 1]]).sum(1).mean())
    coh, purity, sizes = [], [], []
    for k in ids:
        idx = np.where(labels == k)[0]
        sizes.append(len(idx))
        if len(idx) > 1:
            sub = idx if len(idx) <= 200 else rng.choice(idx, 200, replace=False)
            s = x[sub] @ x[sub].T
            coh.append((s.sum() - len(sub)) / (len(sub) * (len(sub) - 1)))
        facets = collections.Counter(f for i in idx for row in space["unit_rows"][i] for f in recs["train"][row]["facets"])
        n_rows = sum(len(space["unit_rows"][i]) for i in idx)
        purity.append(facets.most_common(1)[0][1] / n_rows)
    sims = torch.from_numpy(x).cuda() @ torch.from_numpy(c).cuda().T
    top2 = sims.topk(2, dim=1).values.cpu().numpy()
    cc = c @ c.T
    np.fill_diagonal(cc, 0)

    def coverage(split):
        return float(np.mean([(e @ c.T).max(1).mean() for e in space["rows_emb"][split]]))

    # Stability: two different 90% subsamples, every unit assigned to the nearest centroid of each run.
    assigned = []
    for s in (1, 2):
        sub = np.random.default_rng(s).choice(len(x), int(0.9 * len(x)), replace=False)
        lab = run_method(method, x[sub], args.P, args.knn, s)
        cs, _ = centroids(x[sub], lab)
        assigned.append((torch.from_numpy(x).cuda() @ torch.from_numpy(cs).cuda().T).argmax(1).cpu().numpy())
    cov = {s: coverage(s) for s in ("train", "test")}
    return {
        "clusters": len(ids), "noise": round(float((labels < 0).mean()), 3),
        "size_median": int(np.median(sizes)), "under_5": round(float(np.mean(np.array(sizes) < 5)), 3),
        "coherence_norm": round((float(np.mean(coh)) - rand) / (1 - rand), 3),
        "purity": round(float(np.median(purity)), 3),
        "ambiguous": round(float(np.mean(top2[:, 0] - top2[:, 1] < 0.05)), 3),
        "near_dup": round(float((cc > 0.9).sum() / 2 / len(ids)), 3),
        "stability_ari": round(adjusted_rand_score(*assigned), 3),
        "cov_train": round(cov["train"], 3), "cov_test": round(cov["test"], 3),
        "cov_gap": round(cov["train"] - cov["test"], 4),
    }


def main() -> None:
    args = parse_args()
    out = args.data_root / args.dataset / "collab" / "prototypes" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for space_name in args.spaces:
        space = load_space(args, space_name)
        print(f"[{space_name}] {len(space['units'])} units", flush=True)
        for method in args.methods:
            t0 = time.time()
            labels = run_method(method, space["x"], args.P, args.knn, args.seed)
            row = {"space": space_name, "method": method, **evaluate(space, labels, method, args),
                   "minutes": round((time.time() - t0) / 60, 1)}
            np.save(out / f"labels_{space_name.replace('/', '_')}_{method}.npy", labels)
            (out / f"units_{space_name.replace('/', '_')}.json").write_text(json.dumps(space["units"]))
            table.append(row)
            print(json.dumps(row), flush=True)
    df = pd.DataFrame(table)
    df.to_csv(out / "comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
