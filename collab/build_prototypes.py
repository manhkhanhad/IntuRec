#!/usr/bin/env python3
"""Group train rationales into a fixed corpus of prototype rationales.

The same corpus is then used as the support pool for train, valid and test, so
there is no row-specific retrieval pool (no leakage via a row's own rationale,
no train/test mismatch in what can be retrieved).

For each P in --num-prototypes:
  1. spherical k-means on the mean-centred, normalised train rationale embeddings;
  2. each cluster gets a canonical first-person rationale written by an LLM from
     its members closest to the centroid (never a member's own text, so a train
     row cannot receive its own rationale as input), plus a short label;
  3. quality stats: coverage (cosine of rationales to their centroid), balance
     (size distribution, entropy) and faithfulness (canonical text vs centroid);
  4. top-3 prototype assignments for every train/valid/test row from its OWN
     rationale: labels for training / evaluating the retriever, never inputs.

Writes ``<data_root>/<dataset>/collab/prototypes/P<P>/`` with prototypes.json,
centroids.npy, assign_<split>.npy (ids) / assign_sim_<split>.npy, stats.json.
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

from eval_support import centre, load_rationales, reason_embeddings
from hypothesis_test import embed_cached

SPLITS = ("train", "valid", "test")

SUMMARY_PROMPT = """Below are purchase reasons written by different music buyers. They were grouped together because they share a motive.

{reasons}

Write ONE reason, in the first person as a buyer ("I ..."), that captures the motive these buyers share. Keep the concrete specifics they have in common (era, format, occasion, kind of album, feeling); drop details only one buyer mentions. Never mention any album, song or artist name. At most 30 words.
Also give a short label (2-5 words) naming the motive.
Return JSON: {{"label": "...", "reason": "..."}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--num-prototypes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--summary-members", type=int, default=20)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--server", default="http://lexicon:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def spherical_kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """k-means++ init, then cosine k-means on unit vectors. Returns unit centroids [k, d]."""
    gen = torch.Generator(device=x.device).manual_seed(seed)
    centroids = [x[torch.randint(len(x), (1,), generator=gen, device=x.device)].squeeze(0)]
    closest = torch.full((len(x),), 2.0, device=x.device)
    for _ in range(1, k):
        closest = torch.minimum(closest, 1 - x @ centroids[-1])
        probs = closest.clamp_min(0) ** 2
        centroids.append(x[torch.multinomial(probs / probs.sum(), 1, generator=gen)].squeeze(0))
    c = torch.stack(centroids)
    for _ in range(iters):
        assign = (x @ c.T).argmax(1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        empty = new.norm(dim=1) == 0
        new[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=gen, device=x.device)]
        c = torch.nn.functional.normalize(new, dim=1)
    return c


def summarise(session, args, reasons: list) -> dict:
    schema = {"type": "json_schema", "json_schema": {"name": "proto", "schema": {
        "type": "object", "properties": {"label": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["label", "reason"], "additionalProperties": False}}}
    prompt = SUMMARY_PROMPT.format(reasons="\n".join(f"- {r}" for r in reasons))
    for attempt in range(5):
        try:
            resp = session.post(f"{args.server}/v1/chat/completions", timeout=120, json={
                "model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": 120, "response_format": schema,
                "chat_template_kwargs": {"enable_thinking": False}})
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError) as err:
            if attempt == 4:
                raise RuntimeError(f"summary failed: {err}")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collab = args.data_root / args.dataset / "collab"
    recs, emb = {}, {}
    for split in SPLITS:
        n = len(pd.read_csv(next((args.data_root / args.dataset / split).glob("*.csv"))))
        recs[split] = load_rationales(collab / f"rationales_facet_{split}_{args.rationale_tag}.jsonl", n)
        emb[split] = reason_embeddings(collab, split, recs[split], args.rationale_tag, args.embedder)
    mean = emb["train"].mean(0)
    x = {s: torch.from_numpy(centre(e, mean)).float().to(device) for s, e in emb.items()}
    session = requests.Session()

    for p in args.num_prototypes:
        out = collab / "prototypes" / f"P{p}"
        out.mkdir(parents=True, exist_ok=True)
        c = spherical_kmeans(x["train"], p, args.kmeans_iters, args.seed)
        sims = {s: x[s] @ c.T for s in SPLITS}
        top_sim, top_id = {}, {}
        for s in SPLITS:
            v, i = sims[s].topk(3, dim=1)
            top_sim[s], top_id[s] = v.cpu().numpy(), i.cpu().numpy()
            np.save(out / f"assign_{s}.npy", top_id[s])
            np.save(out / f"assign_sim_{s}.npy", top_sim[s])
        np.save(out / "centroids.npy", c.cpu().numpy())

        assign = top_id["train"][:, 0]
        sizes = np.bincount(assign, minlength=p)
        members = []
        for k in range(p):
            idx = np.where(assign == k)[0]
            order = idx[np.argsort(-sims["train"][idx, k].cpu().numpy())]
            members.append(order)
        with ThreadPoolExecutor(args.workers) as pool:
            summaries = list(pool.map(
                lambda k: summarise(session, args, [recs["train"][i]["reason"] for i in members[k][:args.summary_members]])
                if len(members[k]) else {"label": "empty", "reason": ""}, range(p)))

        texts = [s["reason"] for s in summaries]
        text_emb = embed_cached(texts, out / "prototype_text_emb.npy", args.embedder)
        text_emb = centre(text_emb, mean)
        faithful = (text_emb * c.cpu().numpy()).sum(1)
        prototypes = []
        for k in range(p):
            facets = collections.Counter(f for i in members[k] for f in recs["train"][i]["facets"])
            prototypes.append({
                "id": k, "size": int(sizes[k]), "label": summaries[k]["label"], "reason": summaries[k]["reason"],
                "medoid_reason": recs["train"][members[k][0]]["reason"] if len(members[k]) else "",
                "top_facets": [f for f, _ in facets.most_common(3)],
                "faithfulness": round(float(faithful[k]), 4)})
        (out / "prototypes.json").write_text(json.dumps(prototypes, indent=2))

        share = sizes / sizes.sum()
        stats = {
            "P": p,
            "coverage_cos": {s: round(float(top_sim[s][:, 0].mean()), 4) for s in SPLITS},
            "size_min_median_max": [int(sizes.min()), int(np.median(sizes)), int(sizes.max())],
            "largest_share": round(float(share.max()), 4),
            "norm_entropy": round(float(-(share[share > 0] * np.log(share[share > 0])).sum() / np.log(p)), 4),
            "faithfulness_mean": round(float(faithful.mean()), 4),
            "faithfulness_min": round(float(faithful.min()), 4),
            "clusters_under_5": int((sizes < 5).sum()),
            # Leakage check: a train row's own rationale is one of the members its prototype text was
            # written from; if this cosine is clearly higher for train than for valid/test, prototypes
            # are too small and leak the row's own reason.
            "own_prototype_text_cos": {
                s: round(float((x[s].cpu().numpy() * text_emb[top_id[s][:, 0]]).sum(1).mean()), 4) for s in SPLITS},
        }
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
