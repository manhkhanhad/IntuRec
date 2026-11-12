#!/usr/bin/env python3
"""Collaborative structure and interventions of CollabRec's motive-refined thought (R3-style checks).

The thought is the InjectNet output (history query attending over the injected motive texts) written
into the <|Thought|> slot. For one trained checkpoint, on the test split:

1. Geometry (no decoding), for the refined thought and, as a reference, the pre-injection history query:
   collapse (random-pair cosine, effective rank of the centred matrix) and collaborative structure
   (centred cosine of test rows to train rows that share the target / the user / a motive prototype,
   versus random train rows; and the share of each test row's 10 nearest train thoughts that share them).
2. Interventions (constrained 10-beam decoding of the full test split, as run_collabrec.evaluate):
   raw             own predicted motives (the model as trained)
   mask            thought replaced by zeros
   mean            thought replaced by the mean test thought
   shuffle         thought of a donor row (different user and target)
   swap_motives    the donor row's motives injected, thought recomputed from the own history
   random_motives  k random motive prototypes
   oracle_motives  the row's 1-2 true motives + the top non-true predicted ones (k total, shuffled)
   Wrong-motive harm (mainly with --oracle_input, i.e. an O2 model whose raw input is its own motives):
   null            "No specific motive." in every slot (the no-motive reference)
   pred_top        the retriever's top-k motives (realistic: mostly wrong)
   wrong_top       the retriever's top-k NON-true motives (plausible but all wrong)
   half_wrong      one true motive + the top non-true prediction (shuffled; k = 2)
   Per condition: HR / NDCG @1/3/5/10, paired-bootstrap 95% CI of the change versus raw, top-10 overlap
   with raw, and hits on the donor's target (shuffle / swap_motives; raw gives the chance level).

Each decode is saved as <out_dir>/meta_<condition>.json and reused on a rerun (preemption-safe).
Writes <out_dir>/summary.json and prints the tables.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.CollabRecDataset import CollabRecDataset  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402
from model.LogitsProcessor import IntuRecLogitsProcessor  # noqa: E402
from utils.util import get_hash, get_prefix_data_inturec_f2  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONDITIONS = ["raw", "mask", "mean", "shuffle", "swap_motives", "random_motives", "oracle_motives",
              "null", "pred_top", "wrong_top", "half_wrong"]
CUTOFFS = (1, 3, 5, 10)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="Checkpoint folder of a text-mode CollabRec run.")
    parser.add_argument("--dataset", default="Toys_and_Games")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--support_dir", required=True, help="Support folder used in training (motive_corpus/stage2).")
    parser.add_argument("--support_k", type=int, default=10)
    parser.add_argument("--rationale_tag", default="v5")
    parser.add_argument("--max_rationale_len", type=int, default=64)
    parser.add_argument("--conditions", default=",".join(CONDITIONS[:7]))
    parser.add_argument("--oracle_input", action="store_true",
                        help="O2 checkpoint: raw supports = the row's own motive prototypes (k = 2), as in training.")
    parser.add_argument("--skip_geometry", action="store_true", help="Interventions only (no train-thought pass).")
    parser.add_argument("--out_dir", default="", help="Default: <run_dir>/diagnostics")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--thought_batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1, help="Random test rows (smoke tests); -1 = all.")
    parser.add_argument("--train_limit", type=int, default=-1, help="Random train rows for the neighbour pool.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_dataset(args, tokenizer, split, sample):
    return CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                            support_k=args.support_k, sample=sample, dataset_type=split, reason_step=1,
                            seed=args.seed, support_mode="text", rationale_tag=args.rationale_tag,
                            max_rationale_len=args.max_rationale_len,
                            oracle="own_motive" if args.oracle_input else "")


def row_labels(args, split, row_ids):
    """User, target item and motive-prototype ids of the dataset rows (CSV row order)."""
    frame = pd.read_csv(glob.glob(f"{args.data_path}/{args.dataset}/{split}/{args.dataset}*.csv")[0])
    with open(f"{args.data_path}/{args.dataset}/collab/motive_corpus/row_motives_{split}.json") as f:
        motives = json.load(f)
    return {"user": frame.user_id.to_numpy()[row_ids], "target": frame.item_id.to_numpy()[row_ids],
            "motives": [set(motives[r]) for r in row_ids]}


@torch.no_grad()
def collect_thoughts(model, ds, batch_size, device):
    thoughts, queries = [], []
    for start in tqdm(range(0, len(ds), batch_size), desc=f"thoughts {ds.dataset_type}"):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        ids = torch.tensor(np.stack([b["input_ids"] for b in items])).to(device)
        supports = torch.tensor(np.stack([b["support_tokens"] for b in items])).to(device)
        _, _, _, thought, _ = model.generate_embs(ids, support_tokens=supports)
        thoughts.append(thought.float().cpu())
        queries.append(model.last_his_query.float().cpu())
    return torch.cat(thoughts).numpy(), torch.cat(queries).numpy()


def normalise(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def effective_rank(x):
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def geometry(test_x, train_x, test_lab, train_lab, rng, device):
    """Collapse and collaborative-structure statistics of one latent (test rows against train rows)."""
    pairs = rng.integers(0, len(test_x), (100000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    raw_n = normalise(test_x)
    out = {"random_pair_cos_raw": float((raw_n[pairs[:, 0]] * raw_n[pairs[:, 1]]).sum(1).mean()),
           "effective_rank": effective_rank(test_x), "dim": int(test_x.shape[1]),
           "mean_norm": float(np.linalg.norm(test_x, axis=1).mean())}
    mean = train_x.mean(0)
    te, tr = normalise(test_x - mean), normalise(train_x - mean)

    by = {"target": collections.defaultdict(list), "user": collections.defaultdict(list),
          "motive": collections.defaultdict(list)}
    for j in range(len(tr)):
        by["target"][train_lab["target"][j]].append(j)
        by["user"][train_lab["user"][j]].append(j)
        for m in train_lab["motives"][j]:
            by["motive"][m].append(j)
    rand = rng.integers(0, len(tr), (len(te), 20))
    out["centred_cos_random"] = float(np.einsum("nd,nkd->nk", te, tr[rand]).mean())
    for rel in ("target", "user", "motive"):
        sims = []
        for i in range(len(te)):
            keys = test_lab["motives"][i] if rel == "motive" else [test_lab[rel][i]]
            cand = sorted({j for k in keys for j in by[rel].get(k, [])})
            if cand:
                pick = rng.choice(cand, min(20, len(cand)), replace=False)
                sims.append(float((tr[pick] @ te[i]).mean()))
        out[f"centred_cos_same_{rel}"] = float(np.mean(sims)) if sims else float("nan")
        out[f"rows_with_same_{rel}"] = len(sims)

    # 10 nearest train thoughts per test row: share that share the target / user / a motive.
    tr_t = torch.from_numpy(tr).to(device)
    nn_idx = []
    for start in range(0, len(te), 1024):
        sims = torch.from_numpy(te[start:start + 1024]).to(device) @ tr_t.T
        nn_idx.append(sims.topk(10, dim=1).indices.cpu().numpy())
    nn_idx = np.concatenate(nn_idx)

    def share(idx):
        tgt = np.mean([[train_lab["target"][j] == test_lab["target"][i] for j in row] for i, row in enumerate(idx)])
        usr = np.mean([[train_lab["user"][j] == test_lab["user"][i] for j in row] for i, row in enumerate(idx)])
        mot = np.mean([[bool(train_lab["motives"][j] & test_lab["motives"][i]) for j in row]
                       for i, row in enumerate(idx)])
        return float(tgt), float(usr), float(mot)

    for name, idx in (("knn10", nn_idx), ("random10", rand[:, :10])):
        out[f"{name}_same_target"], out[f"{name}_same_user"], out[f"{name}_shared_motive"] = share(idx)
    return out


def donor_permutation(lab, rng):
    """For every row a donor row with a different user and a different target."""
    n = len(lab["user"])
    donor = rng.permutation(n)
    for i in range(n):
        tries = 0
        while (donor[i] == i or lab["user"][donor[i]] == lab["user"][i]
               or lab["target"][donor[i]] == lab["target"][i]) and tries < 100:
            donor[i] = rng.integers(0, n)
            tries += 1
    return donor


@torch.no_grad()
def decode(model, tokenizer, ds, args, info_file, supports_of, override_of, desc):
    """Constrained top-k beam decoding as run_collabrec.evaluate, with per-row supports / thought override."""
    hash_dict = get_prefix_data_inturec_f2(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

    gen = GenerationConfig(num_beams=args.topk, length_penalty=1.0, num_return_sequences=args.topk,
                           pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
                           max_new_tokens=128)
    start_id = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
    end_id = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]
    preds = []
    for start in tqdm(range(0, len(ds), args.eval_batch_size), desc=desc):
        idx = np.arange(start, min(start + args.eval_batch_size, len(ds)))
        items = [ds[i] for i in idx]
        ids = torch.tensor(np.stack([b["input_ids"] for b in items])).to(model.device)
        mask = torch.tensor(np.stack([b["attention_mask"] for b in items])).to(model.device)
        extra = {"support_tokens": torch.tensor(supports_of(idx, items)).to(model.device)}
        override = override_of(idx)
        if override is not None:
            extra["thought_override"] = torch.tensor(override).to(model.device)
        processors = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=1.0),
            IntuRecLogitsProcessor(tokenizer=tokenizer, model=model, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                   num_beams=args.topk, guide_token="<|StartPredict|>"),
        ])
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen, return_dict_in_generate=True,
                             output_scores=True, logits_processor=processors, **extra)
        seqs = out.sequences
        pos_s = seqs.eq(start_id).long().argmax(dim=1)
        pos_e = seqs.eq(end_id).long().argmax(dim=1)
        arange_ids = torch.arange(seqs.shape[-1], device=seqs.device).unsqueeze(0)
        between = (arange_ids > pos_s.unsqueeze(1)) & (arange_ids < pos_e.unsqueeze(1))
        segments = torch.split(seqs.masked_select(between), between.sum(dim=1).tolist())
        texts = tokenizer.batch_decode(segments, skip_special_tokens=True)
        for i in range(len(idx)):
            preds.append(texts[i * args.topk:(i + 1) * args.topk])
    return preds


def clean(title):
    return title.strip(" \n").strip("\"").strip(" ")


def hit_index(pred, target):
    """Same rule as utils.util.calRecallAndNdcgFromFile (last matching position)."""
    hit = math.inf
    for i, p in enumerate(pred):
        if clean(p) == target:
            hit = i
    return hit


def per_row(preds, targets):
    hits = np.array([hit_index(p, t) for p, t in zip(preds, targets)], dtype=float)
    hr = {k: (hits < k).astype(float) for k in CUTOFFS}
    ndcg = {k: np.where(hits < k, 1.0 / np.log2(np.minimum(hits, 1e9) + 2), 0.0) for k in CUTOFFS}
    return hr, ndcg


def bootstrap_ci(delta, n_boot, rng):
    means = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or os.path.join(args.run_dir, "diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.run_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.run_dir, trust_remote_code=True)
    assert getattr(config, "support_mode", "emb") == "text", "text-mode CollabRec checkpoints only"
    model = CollabRecF2.from_pretrained(args.run_dir, config=config, torch_dtype=torch.bfloat16).to(device).eval()
    model.reason_step, model.beta, model.gamma = 1, 0.0, 0.0

    if args.oracle_input:
        args.support_k = 2
    test = load_dataset(args, tokenizer, "test", args.limit)
    test_lab = row_labels(args, "test", test.row_ids)
    targets = [clean(d["tgt_item"]) for d in test.data]
    print(f"test rows {len(test)}, supports k={args.support_k} ({'own motives' if args.oracle_input else args.support_dir})")

    # 1. Geometry of the refined thought and of the pre-injection history query.
    cache = out_dir / "thoughts_test.npz"
    test_thought, test_query = collect_thoughts(model, test, args.thought_batch_size, device)
    np.savez_compressed(cache, thought=test_thought.astype(np.float16), query=test_query.astype(np.float16),
                        row_ids=np.array(test.row_ids))
    geo = {}
    if not args.skip_geometry:
        train = load_dataset(args, tokenizer, "train", args.train_limit)
        train_lab = row_labels(args, "train", train.row_ids)
        train_thought, train_query = collect_thoughts(model, train, args.thought_batch_size, device)
        geo = {"refined_thought": geometry(test_thought, train_thought, test_lab, train_lab, rng, device),
               "history_query": geometry(test_query, train_query, test_lab, train_lab, rng, device)}
        del train_thought, train_query

    # 2. Interventions.
    table = np.asarray(test.support_table)
    rows = np.asarray(test.support_rows)[np.array(test.row_ids)]  # support_rows is indexed by CSV row; -> position
    n_proto, k = len(table), rows.shape[1]
    donor = donor_permutation(test_lab, rng)
    n_real = n_proto - 1 if args.oracle_input else n_proto  # never draw the "No specific motive." entry
    random_rows = np.stack([rng.choice(n_real, k, replace=False) for _ in range(len(test))])
    scores = np.load(f"{args.data_path}/{args.dataset}/collab/motive_corpus/retriever/cls/scores_test.npy")
    oracle_rows = rows.copy()
    for i, r in enumerate(test.row_ids):
        truth = sorted(test_lab["motives"][i])[:k]
        if truth:
            pred = [int(p) for p in np.argsort(-scores[r].astype(np.float32)) if int(p) not in truth][:k - len(truth)]
            ids = np.array(truth + pred)
            np.random.default_rng(args.seed + r).shuffle(ids)
            oracle_rows[i] = ids
    mean_thought = test_thought.mean(0, keepdims=True)
    # Wrong-motive conditions (motive ids index the prototype table; in --oracle_input mode the table's
    # last entry is "No specific motive.").
    null_rows = np.full_like(rows, len(table) - 1) if args.oracle_input else None
    pred_rows, wrong_rows, half_rows = rows.copy(), rows.copy(), rows.copy()
    for i, r in enumerate(test.row_ids):
        truth = sorted(test_lab["motives"][i])
        ranked = [int(p) for p in np.argsort(-scores[r].astype(np.float32))]
        pred_rows[i] = ranked[:k]
        wrong = [p for p in ranked if p not in truth]
        wrong_rows[i] = wrong[:k]
        ids = np.array(([truth[0]] if truth else []) + wrong[:k - (1 if truth else 0)])
        np.random.default_rng(args.seed + r).shuffle(ids)
        half_rows[i] = ids

    def own(idx, items):
        return np.stack([b["support_tokens"] for b in items])

    setups = {
        "raw": (own, lambda idx: None),
        "mask": (own, lambda idx: np.zeros((len(idx), test_thought.shape[1]), dtype=np.float32)),
        "mean": (own, lambda idx: np.repeat(mean_thought, len(idx), 0).astype(np.float32)),
        "shuffle": (own, lambda idx: test_thought[donor[idx]].astype(np.float32)),
        "swap_motives": (lambda idx, items: table[rows[donor[idx]]], lambda idx: None),
        "random_motives": (lambda idx, items: table[random_rows[idx]], lambda idx: None),
        "oracle_motives": (lambda idx, items: table[oracle_rows[idx]], lambda idx: None),
        "null": (lambda idx, items: table[null_rows[idx]], lambda idx: None),
        "pred_top": (lambda idx, items: table[pred_rows[idx]], lambda idx: None),
        "wrong_top": (lambda idx, items: table[wrong_rows[idx]], lambda idx: None),
        "half_wrong": (lambda idx, items: table[half_rows[idx]], lambda idx: None),
    }
    assert args.oracle_input or "null" not in args.conditions, "null needs --oracle_input (no-motive text entry)"
    info_file = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", "*.txt"))[0]
    conditions = [c for c in args.conditions.split(",") if c]
    assert conditions[0] == "raw" and set(conditions) <= set(CONDITIONS), "conditions must start with raw"
    preds = {}
    for cond in conditions:
        path = out_dir / f"meta_{cond}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if len(saved) == len(test):
                preds[cond] = [r["predict"] for r in saved]
                print(f"{cond}: reusing {path}")
                continue
        supports_of, override_of = setups[cond]
        preds[cond] = decode(model, tokenizer, test, args, info_file, supports_of, override_of, cond)
        path.write_text(json.dumps([{"row": int(r), "tgt_item": test.data[i]["tgt_item"], "predict": preds[cond][i],
                                     "donor_row": int(test.row_ids[donor[i]])}
                                    for i, r in enumerate(test.row_ids)]))

    base_hr, base_ndcg = per_row(preds["raw"], targets)
    null_hr, null_ndcg = per_row(preds["null"], targets) if "null" in preds else (None, None)
    donor_targets = [targets[d] for d in donor]
    results = {}
    for cond in conditions:
        hr, ndcg = per_row(preds[cond], targets)
        res = {f"HR@{c}": float(hr[c].mean()) for c in CUTOFFS}
        res.update({f"NDCG@{c}": float(ndcg[c].mean()) for c in CUTOFFS})
        if cond != "raw":
            for name, cur, base in (("HR@10", hr[10], base_hr[10]), ("NDCG@10", ndcg[10], base_ndcg[10])):
                delta = cur - base
                res[f"d{name}"] = float(delta.mean())
                res[f"d{name}_ci95"] = bootstrap_ci(delta, args.bootstrap, rng)
            if null_hr is not None and cond != "null":  # harm / gain versus no motive
                for name, cur, base in (("HR@10", hr[10], null_hr[10]), ("NDCG@10", ndcg[10], null_ndcg[10])):
                    delta = cur - base
                    res[f"d{name}_vs_null"] = float(delta.mean())
                    res[f"d{name}_vs_null_ci95"] = bootstrap_ci(delta, args.bootstrap, rng)
            res["top10_overlap_with_raw"] = float(np.mean(
                [len({clean(x) for x in a} & {clean(x) for x in b}) / args.topk
                 for a, b in zip(preds[cond], preds["raw"])]))
            res["top1_same_as_raw"] = float(np.mean(
                [clean(a[0]) == clean(b[0]) for a, b in zip(preds[cond], preds["raw"])]))
        donor_hr, _ = per_row(preds[cond], donor_targets)
        res["donor_target_HR@10"] = float(donor_hr[10].mean())
        results[cond] = res

    summary = {"run_dir": args.run_dir, "support_dir": args.support_dir, "test_rows": len(test),
               "train_rows": None if args.skip_geometry else len(train), "geometry": geo,
               "oracle_input": args.oracle_input, "interventions": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if geo:
        print_geometry(geo)
    print("\n== interventions (full test, values; delta vs raw / vs null with paired-bootstrap 95% CI) ==")
    for cond, res in results.items():
        line = f"{cond:15s} HR@10 {res['HR@10']:.4f}  NDCG@10 {res['NDCG@10']:.4f}"
        if cond != "raw":
            lo, hi = res["dHR@10_ci95"]
            line += f"  dHR@10 vs raw {res['dHR@10']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            if "dHR@10_vs_null" in res:
                lo, hi = res["dHR@10_vs_null_ci95"]
                line += f"  vs null {res['dHR@10_vs_null']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            line += f"  overlap {res['top10_overlap_with_raw']:.3f}"
        line += f"  donor-target HR@10 {res['donor_target_HR@10']:.4f}"
        print(line)
    print(f"\nwrote {out_dir / 'summary.json'}")


def print_geometry(geo):
    print("\n== geometry (test rows vs train rows; cos after centring on the train mean) ==")
    keys = ["random_pair_cos_raw", "effective_rank", "centred_cos_random", "centred_cos_same_target",
            "centred_cos_same_user", "centred_cos_same_motive", "knn10_same_target", "random10_same_target",
            "knn10_same_user", "random10_same_user", "knn10_shared_motive", "random10_shared_motive"]
    print(f"{'stat':28s} {'refined thought':>16s} {'history query':>14s}")
    for key in keys:
        print(f"{key:28s} {geo['refined_thought'][key]:16.4f} {geo['history_query'][key]:14.4f}")


if __name__ == "__main__":
    main()
