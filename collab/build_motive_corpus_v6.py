#!/usr/bin/env python3
"""Motive mining v6 (E8b): item-linked motives from target-guided, history-phrased reasoning.

v5 mined 1-2 short "trigger" phrases per purchase (gift, value, collection...), which lost the link to the item
(Report/experiments/06_motive_mining). v6 starts from GEARec v2 reasoning (one paragraph per row, written as a
prediction from the history, best of 8 candidates by target reachability) and keeps the link explicitly: every
motive is a pair (need shown by the history -> kind of item it leads to).

Stages (separate processes, because the envs differ):
  extract  (racer, vLLM server)  reasoning -> 1-2 {need, item_kind} motives per row     -> extract_<split>.jsonl
  embed    (racer, GPU)          Qwen3-Embedding-8B of every distinct motive statement   -> statements.json, emb.npy
  cluster  (rrec, faiss)         kNN graph + Louvain on train statements, ~P clusters,
                                 fold clusters with < --min-rows train rows              -> P<P>/labels.npy
  name     (racer, vLLM server)  name each cluster from its most central statements;
                                 assign every statement to its nearest centroid           -> P<P>/ corpus files
Output corpus (same format as motive_corpus/, so pilot/mining_diagnostics.py works on it):
  data/<ds>/collab/motive_corpus_v6/P<P>/{prototypes.json, centroids.npy, row_motives_<split>.json, stats.json}
Valid/test labels come from target-guided reasoning too; like v5 they are retriever targets only, never inputs.
"""
import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")
GEA = Path("/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/v2_target_reasoning")
GEA_DIRS = {"train": "full_train_bo8_fullhist_r100", "valid": "full_valid_bo8_r100", "test": "full_test_bo8_r100"}
EMB_PROMPT = "Instruct: Represent the shopper's need and the kind of item it leads them to buy\nQuery: "

EXTRACT_PROMPT = """An analyst read a shopper's toy and game purchase history and wrote why the shopper buys their next item:

"{reasoning}"

Extract the 1-2 purchase motives this reasoning gives for the next item. Each motive has two parts:
- need: the underlying need or interest the history shows (who the item is for, age, interest, play style, skill), 3-10 words;
- item_kind: the kind of item this need leads to, specific enough to narrow down a toy catalogue (type of toy or game, theme, features, age range), 3-10 words.
Rules:
- Use only what the reasoning says.
- Never name a product, brand, franchise, character or product line; describe themes in general words (e.g. "superhero action figures", "space-themed building sets").
- Do not write vague motives ("a gift", "having fun", "good quality", "value for money") unless they are tied to a specific kind of item.
Return JSON: {{"motives": [{{"need": "...", "item_kind": "..."}}]}}"""

NAME_PROMPT = """These purchase motives of toy and game shoppers were grouped together. Each is "need -> kind of item it leads to". Most central first:

{statements}

Write the shared motive of the group, keeping BOTH the need and the kind of item:
1. need: 3-10 words;
2. item_kind: 3-10 words, as specific as the group allows;
3. reason: one first-person sentence ("I ...", at most 30 words) a shopper with this motive would say, mentioning the kind of item;
4. label: 2-5 words.
Never name a product, brand, franchise or character.
Return JSON: {{"need": "...", "item_kind": "...", "reason": "...", "label": "..."}}"""

# CDs_and_Vinyl (2026-11-17): reasoning from collab/gearec_v2/target_reasoning.py --domain cds (MotiveRec copy of the
# GEARec v2 pipeline), same Best-of-8 / full-history settings; music wording in the two prompts.
GEA_CDS = Path("data/CDs_and_Vinyl/collab/gearec_v2")
GEA_DIRS_CDS = {s: f"full_{s}_bo8" for s in SPLITS}

EXTRACT_PROMPT_CDS = """An analyst read a shopper's music (CDs and vinyl) purchase history and wrote why the shopper buys their next item:

"{reasoning}"

Extract the 1-2 purchase motives this reasoning gives for the next item. Each motive has two parts:
- need: the underlying need or taste the history shows (genres, era, artist style, mood or occasion, collecting habit, format), 3-10 words;
- item_kind: the kind of release this need leads to, specific enough to narrow down a music catalogue (genre or subgenre, era, style, format such as live album, compilation, box set, vinyl reissue), 3-10 words.
Rules:
- Use only what the reasoning says.
- Never name an artist, band, album, song or record label; describe them in general words (e.g. "1970s British progressive rock albums", "classic soul compilations").
- Do not write vague motives ("a gift", "enjoys music", "good quality", "value for money") unless they are tied to a specific kind of release.
Return JSON: {{"motives": [{{"need": "...", "item_kind": "..."}}]}}"""

NAME_PROMPT_CDS = """These purchase motives of music (CDs and vinyl) shoppers were grouped together. Each is "need -> kind of release it leads to". Most central first:

{statements}

Write the shared motive of the group, keeping BOTH the need and the kind of release:
1. need: 3-10 words;
2. item_kind: 3-10 words, as specific as the group allows;
3. reason: one first-person sentence ("I ...", at most 30 words) a shopper with this motive would say, mentioning the kind of release;
4. label: 2-5 words.
Never name an artist, band, album, song or record label.
Return JSON: {{"need": "...", "item_kind": "...", "reason": "...", "label": "..."}}"""


# Steam / Movielens (2026-11-20): reasoning from target_reasoning.py --domain steam / movielens, same settings as CDs.
EXTRACT_PROMPT_STEAM = """An analyst read a player's Steam game history and wrote why the player gets their next game:

"{reasoning}"

Extract the 1-2 motives this reasoning gives for the next game. Each motive has two parts:
- need: the underlying preference the history shows (genres, gameplay style, single-player or multiplayer, difficulty, setting, art style), 3-10 words;
- item_kind: the kind of game this need leads to, specific enough to narrow down a game catalogue (genre or subgenre, gameplay mechanics, setting, mode such as co-op or competitive), 3-10 words.
Rules:
- Use only what the reasoning says.
- Never name a game, franchise, developer or publisher; describe them in general words (e.g. "co-op zombie survival shooters", "pixel-art roguelike dungeon crawlers").
- Do not write vague motives ("having fun", "good games", "value for money") unless they are tied to a specific kind of game.
Return JSON: {{"motives": [{{"need": "...", "item_kind": "..."}}]}}"""

NAME_PROMPT_STEAM = """These motives of Steam players were grouped together. Each is "need -> kind of game it leads to". Most central first:

{statements}

Write the shared motive of the group, keeping BOTH the need and the kind of game:
1. need: 3-10 words;
2. item_kind: 3-10 words, as specific as the group allows;
3. reason: one first-person sentence ("I ...", at most 30 words) a player with this motive would say, mentioning the kind of game;
4. label: 2-5 words.
Never name a game, franchise, developer or publisher.
Return JSON: {{"need": "...", "item_kind": "...", "reason": "...", "label": "..."}}"""

EXTRACT_PROMPT_MOVIELENS = """An analyst read a viewer's movie history and wrote why the viewer watches their next movie:

"{reasoning}"

Extract the 1-2 motives this reasoning gives for the next movie. Each motive has two parts:
- need: the underlying taste the history shows (genres, themes, moods, eras, storytelling style), 3-10 words;
- item_kind: the kind of movie this need leads to, specific enough to narrow down a movie catalogue (genre or subgenre, theme, era, tone), 3-10 words.
Rules:
- Use only what the reasoning says.
- Never name a movie, franchise, director, actor or character; describe them in general words (e.g. "1980s action-adventure blockbusters", "dark psychological thrillers").
- Do not write vague motives ("entertainment", "good movies", "popular films") unless they are tied to a specific kind of movie.
Return JSON: {{"motives": [{{"need": "...", "item_kind": "..."}}]}}"""

NAME_PROMPT_MOVIELENS = """These motives of movie viewers were grouped together. Each is "need -> kind of movie it leads to". Most central first:

{statements}

Write the shared motive of the group, keeping BOTH the need and the kind of movie:
1. need: 3-10 words;
2. item_kind: 3-10 words, as specific as the group allows;
3. reason: one first-person sentence ("I ...", at most 30 words) a viewer with this motive would say, mentioning the kind of movie;
4. label: 2-5 words.
Never name a movie, franchise, director, actor or character.
Return JSON: {{"need": "...", "item_kind": "...", "reason": "...", "label": "..."}}"""

NEW_DOMAINS = {"Steam": (EXTRACT_PROMPT_STEAM, NAME_PROMPT_STEAM),
               "Movielens": (EXTRACT_PROMPT_MOVIELENS, NAME_PROMPT_MOVIELENS)}


def domain(args):
    """(reasoning root, split dirs, extract prompt, name prompt) for the dataset."""
    if args.dataset == "CDs_and_Vinyl":
        return GEA_CDS, GEA_DIRS_CDS, EXTRACT_PROMPT_CDS, NAME_PROMPT_CDS
    if args.dataset in NEW_DOMAINS:
        return (Path(f"data/{args.dataset}/collab/gearec_v2"), GEA_DIRS_CDS) + NEW_DOMAINS[args.dataset]
    return GEA, GEA_DIRS, EXTRACT_PROMPT, NAME_PROMPT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["extract", "embed", "cluster", "name"])
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--P", type=int, nargs="+", default=[1000, 2000])
    p.add_argument("--knn", type=int, default=15)
    p.add_argument("--min-rows", type=int, default=5)
    p.add_argument("--name-statements", type=int, default=20)
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="extract only the first N rows per split (smoke test)")
    return p.parse_args()


def normalise(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def statement(m):
    return f"{m['need'].strip().rstrip('.')} -> {m['item_kind'].strip().rstrip('.')}".lower()


def llm(session, args, prompt, schema, max_tokens):
    import requests
    payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
    for attempt in range(5):
        try:
            r = session.post(f"{args.server}/v1/chat/completions", json=payload, timeout=180)
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError) as err:
            if attempt == 4:
                return {"error": str(err)}
            time.sleep(2 ** attempt)


MOTIVE = {"type": "object", "properties": {"need": {"type": "string"}, "item_kind": {"type": "string"}},
          "required": ["need", "item_kind"], "additionalProperties": False}
EXTRACT_SCHEMA = {"type": "object", "properties": {"motives": {"type": "array", "items": MOTIVE, "minItems": 1,
                                                               "maxItems": 2}},
                  "required": ["motives"], "additionalProperties": False}
NAME_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in ("need", "item_kind", "reason", "label")},
               "required": ["need", "item_kind", "reason", "label"], "additionalProperties": False}


def stage_extract(args, out):
    import requests
    session = requests.Session()
    for s in SPLITS:
        path = out / f"extract_{s}.jsonl"
        done = {}
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                if "error" not in r:
                    done[r["row"]] = r
        todo = []
        gea, gea_dirs, _, _ = domain(args)
        for line in open(gea / gea_dirs[s] / "validated.jsonl"):
            g = json.loads(line)
            row = int(g["sample_id"])
            best = g.get("best") or {}
            if row in done or not best.get("reasoning"):
                continue
            todo.append((row, best["reasoning"], best.get("rank")))
        if args.limit:
            todo = todo[: args.limit]
        print(f"{s}: {len(done)} done, {len(todo)} to extract", flush=True)

        def work(t):
            row, rea, rank = t
            res = llm(session, args, domain(args)[2].format(reasoning=rea), EXTRACT_SCHEMA, 160)
            return {"row": row, "gearec_rank": rank, **res}
        with open(path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
            for i, r in enumerate(pool.map(work, todo)):
                f.write(json.dumps(r) + "\n")
                if i % 5000 == 0:
                    print(f"  {s} {i}/{len(todo)}", flush=True)


def load_extract(out):
    recs = {}
    for s in SPLITS:
        recs[s] = {}
        for line in open(out / f"extract_{s}.jsonl"):
            r = json.loads(line)
            if "error" not in r and r.get("motives"):
                recs[s][r["row"]] = [statement(m) for m in r["motives"]]
    return recs


def stage_embed(args, out):
    import torch
    from sentence_transformers import SentenceTransformer
    recs = load_extract(out)
    texts = sorted({t for s in SPLITS for ts in recs[s].values() for t in ts})
    st = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    emb = st.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True,
                    prompt=EMB_PROMPT).astype(np.float32)
    (out / "statements.json").write_text(json.dumps(texts))
    np.save(out / "emb.npy", emb)
    print(len(texts), "distinct statements", {s: len(recs[s]) for s in SPLITS})


def train_units(args, out):
    import pandas as pd
    recs = load_extract(out)
    texts = json.loads((out / "statements.json").read_text())
    tid = {t: i for i, t in enumerate(texts)}
    emb = np.load(out / "emb.npy")
    train = pd.read_csv(next((args.data_root / args.dataset / "train").glob("*.csv")))
    occ = collections.Counter()
    seen = set()
    for row, ts in recs["train"].items():
        for t in ts:
            if (train.user_id.iat[row], t) not in seen:  # one user cannot make a cluster alone
                seen.add((train.user_id.iat[row], t))
                occ[t] += 1
    units = sorted(occ)
    raw = emb[[tid[u] for u in units]]
    mean = raw.mean(0)
    return recs, texts, tid, emb, units, normalise(raw - mean), mean, np.array([occ[u] for u in units], np.float32)


def stage_cluster(args, out):
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_clustering import run_method
    _, _, _, _, units, x, _, w = train_units(args, out)
    print(len(units), "train units", flush=True)
    for P in args.P:
        lab = run_method("louvain", x, P, args.knn, 0)
        # fold clusters backed by fewer than --min-rows train rows into their nearest cluster
        while True:
            _, lab = np.unique(lab, return_inverse=True)
            c = np.zeros((lab.max() + 1, x.shape[1]), np.float32)
            np.add.at(c, lab, x)
            c = normalise(c)
            rows = np.bincount(lab, weights=w)
            small = np.where(rows < args.min_rows)[0]
            if len(small) == 0:
                break
            s = c[small] @ c.T
            s[np.arange(len(small)), small] = -2
            moved = dict(zip(small, s.argmax(1)))
            lab = np.array([moved.get(k, k) for k in lab])
        d = out / f"P{P}"
        d.mkdir(exist_ok=True)
        np.save(d / "labels.npy", lab)
        print(f"P={P}: {lab.max() + 1} clusters after fold; rows/cluster median {np.median(rows):.0f} max {rows.max():.0f}",
              flush=True)


def stage_name(args, out):
    import requests
    recs, texts, tid, emb, units, x, mean, w = train_units(args, out)
    session = requests.Session()
    allv = normalise(emb - mean)
    for P in args.P:
        d = out / f"P{P}"
        lab = np.load(d / "labels.npy")
        K = lab.max() + 1
        c = np.zeros((K, x.shape[1]), np.float32)
        np.add.at(c, lab, x)
        c = normalise(c)
        members = [np.where(lab == k)[0] for k in range(K)]
        central = [[units[i] for i in m[np.argsort(-(x[m] @ c[k]))][: args.name_statements]]
                   for k, m in enumerate(members)]
        with ThreadPoolExecutor(args.workers) as pool:
            names = list(pool.map(lambda st: llm(session, args, domain(args)[3].format(
                statements="\n".join(f"- {t}" for t in st)), NAME_SCHEMA, 200), central))
        assign = (allv @ c.T).argmax(1)  # every statement of every split -> nearest centroid
        row_motives = {}
        for s in SPLITS:
            import pandas as pd
            n = len(pd.read_csv(next((args.data_root / args.dataset / s).glob("*.csv"))))
            row_motives[s] = [sorted({int(assign[tid[t]]) for t in recs[s].get(r, [])}) for r in range(n)]
            (d / f"row_motives_{s}.json").write_text(json.dumps(row_motives[s]))
        cnt = np.bincount([m for ms in row_motives["train"] for m in ms], minlength=K)
        protos = []
        for k in range(K):
            nm = names[k] if "error" not in names[k] else {"need": central[k][0], "item_kind": "", "reason": "",
                                                           "label": ""}
            protos.append({"id": k, "label": nm["label"], "motive": f"{nm['need']} -> {nm['item_kind']}",
                           "need": nm["need"], "item_kind": nm["item_kind"], "reason": nm["reason"],
                           "train_rows": int(cnt[k]), "n_statements": int(len(members[k])),
                           "top_phrases": central[k][:10]})
        (d / "prototypes.json").write_text(json.dumps(protos, indent=1))
        np.save(d / "centroids.npy", c)
        stats = {"K": int(K), "naming_errors": int(sum("error" in n for n in names)),
                 "rows_with_0_motives": {s: round(float(np.mean([not m for m in row_motives[s]])), 4) for s in SPLITS},
                 "rows_per_prototype_min_median_max": [int(cnt.min()), int(np.median(cnt)), int(cnt.max())],
                 "largest_share": round(float(cnt.max() / cnt.sum()), 4)}
        (d / "stats.json").write_text(json.dumps(stats, indent=1))
        print(f"P={P}", json.dumps(stats), flush=True)


def main():
    args = parse_args()
    out = args.data_root / args.dataset / "collab" / "motive_corpus_v6"
    out.mkdir(parents=True, exist_ok=True)
    {"extract": stage_extract, "embed": stage_embed, "cluster": stage_cluster, "name": stage_name}[args.stage](args, out)


if __name__ == "__main__":
    main()
