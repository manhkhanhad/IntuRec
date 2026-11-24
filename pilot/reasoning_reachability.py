#!/usr/bin/env python3
"""E8a': how well does each kind of motive text point to the target item? (Toys, CPU, sample of test rows)

Rank of the target among all items (history items masked) when the text alone is embedded with mxbai-embed-large
and compared with the GEARec mxbai item embeddings: the same "reachability" check (V3) as
GEARec_R3_toys/v2_target_reasoning/target_reasoning.py. Texts compared on the same rows:
  v5_reason       our Qwen3-8B first-person reason (target + review seen)
  v5_phrases      our 1-2 motive phrases, joined
  prototypes      the "motive: reason" texts of the row's mined prototypes (what MotiveRec labels)
  gearec_cand0    GEARec v2 reasoning, first sampled candidate (target seen, no selection)
  gearec_best     GEARec v2 best-of-8 candidate (selected by this same rank)

usage: python pilot/reasoning_reachability.py [n_rows] [corpus_dir, default motive_corpus]
"""
import ast
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

GEA = Path("/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys")
DS = Path("data/Toys_and_Games")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    te = pd.read_csv(glob.glob(str(DS / "test" / "*.csv"))[0])
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(te), n, replace=False))

    v5 = {}
    for line in open(DS / "collab" / "rationales_facet_test_v5.jsonl"):
        r = json.loads(line)
        v5[r["row"]] = r
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else DS / "collab" / "motive_corpus"
    protos = json.loads((corpus / "prototypes.json").read_text())
    rm = json.loads((corpus / "row_motives_test.json").read_text())
    gea = {}
    for line in open(GEA / "v2_target_reasoning" / "full_test_bo8_r100" / "validated.jsonl"):
        r = json.loads(line)
        gea[int(r["sample_id"])] = r

    def gea_text(i, best):
        r = gea.get(i)
        if not r:
            return ""
        if best:
            return (r.get("best") or {}).get("reasoning") or ""
        c0 = [c for c in r["checks"] if c.get("cand") == 0]
        return (c0[0].get("reasoning") if c0 else "") or ""

    texts = {
        "v5_reason": [v5[i]["reason"] for i in rows],
        "v5_phrases": ["; ".join(v5[i]["motives"]) for i in rows],
        "prototypes": [" ".join(f"{protos[m]['motive']}: {protos[m]['reason']}" for m in rm[i]) for i in rows],
        "gearec_cand0": [gea_text(i, False) for i in rows],
        "gearec_best": [gea_text(i, True) for i in rows],
    }
    ie = torch.load(GEA / "outputs" / "item_emb_mxbai.pt", map_location="cpu").float()
    ie = ie / ie.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    emb = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    res = {}
    for name, tx in texts.items():
        keep = [j for j, t in enumerate(tx) if t.strip()]
        q = emb.encode([tx[j] for j in keep], batch_size=64, convert_to_tensor=True).float()
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ranks = []
        for j, qq in zip(keep, q):
            i = rows[j]
            sc = qq @ ie.T
            tgt = int(te.item_id[i])
            excl = [h for h in ast.literal_eval(te.history_item_id[i]) if h != tgt]
            if excl:
                sc[excl] = float("-inf")
            ranks.append(int((sc > sc[tgt]).sum()) + 1)
        r = np.array(ranks)
        res[name] = {"n": len(r), "median_rank": float(np.median(r)), "top10": round(float((r <= 10).mean()), 4),
                     "top20": round(float((r <= 20).mean()), 4), "top100": round(float((r <= 100).mean()), 4)}
        print(name, res[name], flush=True)
    out = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/MotiveRec_pilot")
    (out / f"reasoning_reachability_{corpus.parent.name}_{corpus.name}.json").write_text(json.dumps(res, indent=1))
    print("example row", rows[0])
    for k, tx in texts.items():
        print(f"--- {k}: {tx[0][:400]}")


if __name__ == "__main__":
    main()
