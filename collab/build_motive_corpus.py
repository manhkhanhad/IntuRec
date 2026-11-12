#!/usr/bin/env python3
"""Build the motive-prototype corpus from clustered motive phrases.

Input: the phrase/qwen3/louvain clustering from compare_clustering.py (units =
distinct train motive phrases after per-user dedup and broken-title removal) and
the Qwen3-Embedding phrase vectors (hyp_cache/cluster_phrase_qwen3.npy).

  1. merge near-duplicate clusters (centroid cosine > --merge-threshold), then fold
     clusters backed by fewer than --min-rows train rows into their nearest cluster;
  2. an LLM names every cluster from its most frequent phrases: a canonical motive
     phrase, a first-person sentence and a label, and classifies it as a purchase
     motive or a post-purchase complaint; complaint clusters are removed;
  3. every phrase of every train/valid/test row is assigned to its nearest cluster;
     a phrase landing in a removed cluster is dropped, otherwise the row gets that
     prototype (so each row has 0-2 motive prototypes);
  4. quality / leakage stats, using Qwen3 embeddings of the canonical texts.

Needs transformers >= 4.51 for Qwen3 (the `racer` env). Writes
``<data_root>/<dataset>/collab/motive_corpus/``: prototypes.json, centroids.npy,
row_motives_<split>.json (list of prototype ids per row), stats.json.
"""

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

SPLITS = ("train", "valid", "test")
QWEN_PROMPT = "Instruct: Represent the buyer's purchase motive expressed in this phrase\nQuery: "

NAME_PROMPT = """These short phrases were written to describe why {buyers} bought {item}. They were grouped together because they express a similar motive. Most frequent first:

{phrases}

1. type: "purchase_motive" if they describe a reason to BUY (a desire, need, taste, occasion, value, format...), or "post_purchase_complaint" if they mainly describe dissatisfaction or disappointment after buying.
2. motive: one canonical phrase (3-8 words) naming the shared motive.
3. reason: one first-person sentence ("I ...", at most 25 words) a buyer with this motive would say.
4. label: 2-4 words.
Never mention any {names}.
Return JSON: {{"type": "...", "motive": "...", "reason": "...", "label": "..."}}"""


DOMAIN_WORDS = {
    "music": {"buyers": "music buyers", "item": "an album", "names": "album, song, band or artist name"},
    "toys": {"buyers": "shoppers", "item": "a toy or game", "names": "product, brand, franchise or character name"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--domain", choices=list(DOMAIN_WORDS), default="music")
    parser.add_argument("--labels", default="labels_phrase_qwen3_louvain.npy")
    parser.add_argument("--merge-threshold", type=float, default=0.9)
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--name-phrases", type=int, default=25)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--force-keep", nargs="*", default=["replac"],
                        help="Keep clusters whose top phrases contain any of these substrings, whatever the LLM says "
                             "(replacing a damaged/worn copy is a purchase motive, not a complaint).")
    return parser.parse_args()


def normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def relabel_centroids(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple:
    ids = np.unique(labels)
    remap = {k: i for i, k in enumerate(ids)}
    labels = np.array([remap[k] for k in labels])
    c = np.zeros((len(ids), x.shape[1]), dtype=np.float32)
    np.add.at(c, labels, x * weights[:, None])
    return labels, normalise(c)


def name_cluster(session, args, phrases: list) -> dict:
    schema = {"type": "json_schema", "json_schema": {"name": "motive", "schema": {
        "type": "object", "properties": {
            "type": {"type": "string", "enum": ["purchase_motive", "post_purchase_complaint"]},
            "motive": {"type": "string"}, "reason": {"type": "string"}, "label": {"type": "string"}},
        "required": ["type", "motive", "reason", "label"], "additionalProperties": False}}}
    prompt = NAME_PROMPT.format(phrases="\n".join(f"- {p}" for p in phrases), **DOMAIN_WORDS[args.domain])
    for attempt in range(5):
        try:
            resp = session.post(f"{args.server}/v1/chat/completions", timeout=120, json={
                "model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": 150, "response_format": schema,
                "chat_template_kwargs": {"enable_thinking": False}})
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError) as err:
            if attempt == 4:
                raise RuntimeError(f"naming failed: {err}")


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    compare = collab / "prototypes" / "compare"
    out = collab / "motive_corpus"
    out.mkdir(parents=True, exist_ok=True)

    recs = {}
    for s in SPLITS:
        n = len(pd.read_csv(next((args.data_root / args.dataset / s).glob("*.csv"))))
        recs[s] = [None] * n
        for line in open(collab / f"rationales_facet_{s}_{args.rationale_tag}.jsonl"):
            r = json.loads(line)
            recs[s][r["row"]] = r
    phrases = lambda r: [m.strip().lower() for m in r["motives"]]
    all_texts = sorted({p for s in SPLITS for r in recs[s] for p in phrases(r)})
    emb = np.load(collab / "hyp_cache" / "cluster_phrase_qwen3.npy")
    assert len(all_texts) == len(emb), "phrase list does not match the cached Qwen3 embeddings"
    text_id = {t: i for i, t in enumerate(all_texts)}

    units = json.loads((compare / "units_phrase_qwen3.json").read_text())
    labels = np.load(compare / args.labels)
    raw = emb[[text_id[u] for u in units]]
    mean = raw.mean(0)
    x = normalise(raw - mean)
    occ = collections.Counter(p for r in recs["train"] for p in phrases(r))
    weights = np.array([occ[u] for u in units], dtype=np.float32)
    n0 = len(np.unique(labels))

    # 1a. Merge near-duplicate clusters (union-find over centroid pairs above the threshold).
    labels, c = relabel_centroids(x, labels, weights)
    parent = list(range(len(c)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    sim = c @ c.T
    for a, b in zip(*np.where(np.triu(sim, 1) > args.merge_threshold)):
        parent[find(a)] = find(b)
    labels, c = relabel_centroids(x, np.array([find(k) for k in labels]), weights)
    n_merged = len(c)

    # 1b. Fold clusters backed by too few train rows into their nearest other cluster.
    while True:
        rows_per = np.bincount(labels, weights=weights, minlength=len(c))
        small = np.where(rows_per < args.min_rows)[0]
        if len(small) == 0:
            break
        s = c[small] @ c.T
        s[np.arange(len(small)), small] = -2
        target = s.argmax(1)
        moved = dict(zip(small, target))
        labels, c = relabel_centroids(x, np.array([moved.get(k, k) for k in labels]), weights)

    # 2. Name and classify every cluster.
    members = [np.where(labels == k)[0] for k in range(len(c))]
    top_phrases = [[units[i] for i in sorted(m, key=lambda i: -weights[i])[:args.name_phrases]] for m in members]
    session = requests.Session()
    with ThreadPoolExecutor(args.workers) as pool:
        names = list(pool.map(lambda p: name_cluster(session, args, p), top_phrases))
    keep = np.array([n["type"] == "purchase_motive" or
                     any(f in ph for f in args.force_keep for ph in top_phrases[k][:5])
                     for k, n in enumerate(names)])

    # 3. Assign every phrase of every row; drop phrases whose nearest cluster was removed.
    new_id = {k: i for i, k in enumerate(np.where(keep)[0])}
    phrase_vec = normalise(emb - mean)
    phrase_cluster = (torch.from_numpy(phrase_vec) @ torch.from_numpy(c).T).argmax(1).numpy()
    row_motives = {}
    for s in SPLITS:
        row_motives[s] = [sorted({new_id[phrase_cluster[text_id[p]]] for p in phrases(r)
                                  if keep[phrase_cluster[text_id[p]]]}) for r in recs[s]]
        (out / f"row_motives_{s}.json").write_text(json.dumps(row_motives[s]))

    kept = np.where(keep)[0]
    prototypes = []
    for i, k in enumerate(kept):
        prototypes.append({"id": i, "label": names[k]["label"], "motive": names[k]["motive"],
                           "reason": names[k]["reason"], "train_rows": int(rows_per[k]),
                           "n_phrases": len(members[k]), "top_phrases": top_phrases[k][:10]})
    (out / "prototypes.json").write_text(json.dumps(prototypes, indent=2))
    (out / "removed_complaints.json").write_text(json.dumps(
        [{"label": names[k]["label"], "motive": names[k]["motive"], "top_phrases": top_phrases[k][:10]}
         for k in np.where(~keep)[0]], indent=2))
    np.save(out / "centroids.npy", c[kept])

    # 4. Quality and leakage: own phrase vs its prototype's canonical motive text (Qwen3 space).
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda", model_kwargs={"dtype": torch.bfloat16})
    motive_emb = normalise(st.encode([p["motive"] for p in prototypes], batch_size=64, normalize_embeddings=True,
                                     prompt=QWEN_PROMPT).astype(np.float32) - mean)
    np.save(out / "motive_text_emb.npy", motive_emb)
    faith = (motive_emb * c[kept]).sum(1)

    def own_text_cos(s):
        vals = [float(phrase_vec[text_id[p]] @ motive_emb[new_id[phrase_cluster[text_id[p]]]])
                for r in recs[s] for p in phrases(r) if keep[phrase_cluster[text_id[p]]]]
        return round(float(np.mean(vals)), 4)
    rows_kept = np.bincount([m for ms in row_motives["train"] for m in ms], minlength=len(kept))
    stats = {
        "clusters_louvain": int(n0), "after_merge": int(n_merged), "after_fold": int(len(c)),
        "removed_complaint_clusters": int((~keep).sum()), "prototypes": int(len(kept)),
        "train_phrase_occurrences_removed": round(float(1 - rows_per[keep].sum() / rows_per.sum()), 4),
        "rows_with_0_motives": {s: round(float(np.mean([len(m) == 0 for m in row_motives[s]])), 4) for s in SPLITS},
        "rows_per_prototype_min_median_max": [int(rows_kept.min()), int(np.median(rows_kept)), int(rows_kept.max())],
        "largest_share": round(float(rows_kept.max() / rows_kept.sum()), 4),
        "faithfulness_mean_min": [round(float(faith.mean()), 4), round(float(faith.min()), 4)],
        "own_motive_text_cos": {s: own_text_cos(s) for s in SPLITS},
    }
    stats["leak_gap_train_minus_test"] = round(stats["own_motive_text_cos"]["train"] - stats["own_motive_text_cos"]["test"], 4)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
