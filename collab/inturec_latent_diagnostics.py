#!/usr/bin/env python3
"""Does IntuRec's prediction depend on its latent thought? (R3-style interventions on an IntuRec stage-2 checkpoint)

IntuRec stage 2 writes one vector into the <|Thought|> slot: InjectNet(history query, the row's intuition list =
stage-1 top-N titles). This script decodes the full test split (constrained top-k beams, as run_inturec_f2.eval)
under:
   raw                  the model as trained (own intuitions)
   mask                 thought replaced by zeros
   mean                 thought replaced by the mean test thought
   shuffle              thought of a donor row (different user and target)
   swap_intuitions      the donor row's intuition list injected, thought recomputed from the own history
   random_intuitions    N random catalog titles as the intuition list
   oracle_intuitions    the true target title put into the list (replacing the last intuition) when it is absent
   remove_target        the true target taken out of the list (replaced by a random title) when it is present
Per condition: HR / NDCG @1/3/5/10 with paired-bootstrap 95% CIs of the change vs raw, top-10 overlap with raw,
hits on the donor's target, the share of the top-10 that comes from the injected list (copy rate), and HR split
by whether the true target is in the own intuition list. Also: collapse statistics of the test thoughts.

IntuRec's files are not modified: a subclass adds the thought override (the same pattern as CollabRecF2's
thought_override). Each decode is saved as <out_dir>/meta_<condition>.json and reused on a rerun.
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.IntuRecF2Dataset import IntuRecF2Dataset  # noqa: E402
from model.IntuRecF2 import IntuRecF2  # noqa: E402
from model.LogitsProcessor import IntuRecLogitsProcessor  # noqa: E402
from utils.util import get_hash, get_prefix_data_inturec_f2  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONDITIONS = ["raw", "mask", "mean", "shuffle", "swap_intuitions", "random_intuitions", "oracle_intuitions",
              "remove_target"]
CUTOFFS = (1, 3, 5, 10)


class DiagIntuRecF2(IntuRecF2):
    """IntuRecF2 + an optional per-row override of the thought vector, and a record of the thought."""

    def generate_embs(self, input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids):
        inputs_embeds, loss = super().generate_embs(input_ids, intuition_tokens, random_tokens, tgt_tokens, tgt_ids)
        thought_id = self.model.embed_tokens.num_embeddings - 3
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        first = torch.where(input_ids == thought_id, idxs, input_ids.new_full(idxs.shape, input_ids.shape[1])).min(1).values
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        self.last_thought = inputs_embeds[rows, first].detach()
        if getattr(self, "_override", None) is not None:
            inputs_embeds[rows, first] = self._override.to(inputs_embeds.dtype)
        return inputs_embeds, loss

    # Explicit arguments: generate() validates extra kwargs (intuition_tokens, thought_override) against this signature.
    def forward(self, input_ids=None, attention_mask=None, intuition_tokens=None, thought_override=None, **kwargs):
        self._override = thought_override
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, intuition_tokens=intuition_tokens,
                               **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", required=True, help="IntuRec stage-2 run folder (model.safetensors).")
    parser.add_argument("--dataset", default="Toys_and_Games")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--intuition_num", type=int, default=15)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--out_dir", default="", help="Default: <ckpt_dir>/diagnostics")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1, help="Random test rows (smoke tests); -1 = all.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


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


def collapse_stats(x, rng):
    x = x.astype(np.float64)
    n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    pairs = rng.integers(0, len(x), (200000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    mean = x.mean(0)
    ev = np.linalg.svd(x - mean, compute_uv=False) ** 2
    f = ev / ev.sum()
    f = f[f > 0]
    cum = np.cumsum(f)
    return {"effective_rank": float(np.exp(-(f * np.log(f)).sum())), "participation_ratio": float(1 / (f ** 2).sum()),
            "pc1_variance": float(f[0]), "pcs_for_90pct": int(np.searchsorted(cum, 0.9)) + 1,
            "random_pair_cos": float((n[pairs[:, 0]] * n[pairs[:, 1]]).sum(1).mean()),
            "cos_to_mean": float((n @ (mean / np.linalg.norm(mean))).mean())}


@torch.no_grad()
def decode(model, tokenizer, ds, lists, overrides, args, info_file, desc):
    """Constrained beam decoding as run_inturec_f2.eval, with per-row intuition lists / thought overrides."""
    hash_dict = get_prefix_data_inturec_f2(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

    gen = GenerationConfig(num_beams=args.topk, length_penalty=1.0, num_return_sequences=args.topk,
                           pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
                           max_new_tokens=128)
    start_id = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
    end_id = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]
    pad = tokenizer.pad_token_id
    preds = []
    for start in tqdm(range(0, len(ds), args.eval_batch_size), desc=desc):
        idx = np.arange(start, min(start + args.eval_batch_size, len(ds)))
        items = [ds[i] for i in idx]
        ids = torch.tensor(np.stack([b["input_ids"] for b in items])).to(model.device)
        mask = torch.tensor(np.stack([b["attention_mask"] for b in items])).to(model.device)
        toks = [[ds.tokenizer.encode(t, bos=False, eos=False) for t in lists[i]] for i in idx]  # IntuRec's wrapper
        width = max(len(t) for row in toks for t in row)
        intu = torch.tensor([[[pad] * (width - len(t)) + t for t in row] for row in toks]).to(model.device)
        extra = {"intuition_tokens": intu}
        if overrides is not None:
            extra["thought_override"] = torch.tensor(overrides[idx]).to(model.device)
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


@torch.no_grad()
def collect_thoughts(model, ds, batch_size, device):
    out = []
    pad = model.config.pad_token_id
    for start in tqdm(range(0, len(ds), batch_size), desc="thoughts"):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        ids = torch.tensor(np.stack([b["input_ids"] for b in items])).to(device)
        width = max(b["intuition_tokens"].shape[1] for b in items)
        intu = torch.tensor(np.stack([np.pad(b["intuition_tokens"], ((0, 0), (width - b["intuition_tokens"].shape[1], 0)),
                                             constant_values=pad) for b in items])).to(device)
        model._override = None
        model.generate_embs(ids, intu, None, None, None)
        out.append(model.last_thought.float().cpu())
    return torch.cat(out).numpy()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or os.path.join(args.ckpt_dir, "diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    model = DiagIntuRecF2.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16).to(device).eval()
    model.reason_step = 1
    ds = IntuRecF2Dataset(args.data_path, args.dataset, tokenizer, sample=args.limit, dataset_type="test",
                          intuition_num=args.intuition_num, seed=args.seed, reason_step=1)
    n = len(ds)
    raw_rows = ds.raw_data
    targets = [clean(d["tgt_item"]) for d in ds.data]
    users = [r["user_id"] for r in raw_rows]
    own = [list(d["intuition_items"]) for d in ds.data]
    catalog = ds.item_names
    print(f"test rows {n}, intuition list size {args.intuition_num}, catalog {len(catalog)}")

    thoughts = collect_thoughts(model, ds, 16, device)
    np.save(out_dir / "thoughts_test.npy", thoughts.astype(np.float16))
    geometry = collapse_stats(thoughts, rng)

    # Donor: a different user and a different target.
    donor = rng.permutation(n)
    for i in range(n):
        tries = 0
        while (donor[i] == i or users[donor[i]] == users[i] or targets[donor[i]] == targets[i]) and tries < 100:
            donor[i] = rng.integers(0, n)
            tries += 1
    in_list = np.array([targets[i] in {clean(t) for t in own[i]} for i in range(n)])
    random_lists = [[catalog[j] for j in rng.choice(len(catalog), args.intuition_num, replace=False)] for _ in range(n)]
    oracle_lists, removed_lists = [], []
    for i in range(n):
        tgt = "\"" + targets[i] + "\""
        lst = list(own[i])
        oracle_lists.append(lst if in_list[i] else lst[:-1] + [tgt])
        removed_lists.append([t if clean(t) != targets[i] else catalog[rng.integers(0, len(catalog))] for t in lst])
    mean_thought = thoughts.mean(0, keepdims=True)

    setups = {
        "raw": (own, None),
        "mask": (own, np.zeros_like(thoughts, dtype=np.float32)),
        "mean": (own, np.repeat(mean_thought, n, 0).astype(np.float32)),
        "shuffle": (own, thoughts[donor].astype(np.float32)),
        "swap_intuitions": ([own[d] for d in donor], None),
        "random_intuitions": (random_lists, None),
        "oracle_intuitions": (oracle_lists, None),
        "remove_target": (removed_lists, None),
    }
    info_file = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", "*.txt"))[0]
    conditions = [c for c in args.conditions.split(",") if c]
    assert conditions[0] == "raw" and set(conditions) <= set(CONDITIONS), "conditions must start with raw"
    preds = {}
    for cond in conditions:
        path = out_dir / f"meta_{cond}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if len(saved) == n:
                preds[cond] = [r["predict"] for r in saved]
                print(f"{cond}: reusing {path}")
                continue
        lists, overrides = setups[cond]
        preds[cond] = decode(model, tokenizer, ds, lists, overrides, args, info_file, cond)
        path.write_text(json.dumps([{"user_id": users[i], "tgt_item": ds.data[i]["tgt_item"], "predict": preds[cond][i],
                                     "intuitions": lists[i], "donor": int(donor[i])} for i in range(n)]))

    base_hr, base_ndcg = per_row(preds["raw"], targets)
    donor_targets = [targets[d] for d in donor]
    results = {}
    for cond in conditions:
        lists = setups[cond][0]
        hr, ndcg = per_row(preds[cond], targets)
        res = {f"HR@{c}": float(hr[c].mean()) for c in CUTOFFS}
        res.update({f"NDCG@{c}": float(ndcg[c].mean()) for c in CUTOFFS})
        res["HR@10_target_in_own_list"] = float(hr[10][in_list].mean()) if in_list.any() else float("nan")
        res["HR@10_target_not_in_own_list"] = float(hr[10][~in_list].mean())
        res["copy_rate"] = float(np.mean([len({clean(x) for x in p} & {clean(x) for x in l}) / len(p)
                                          for p, l in zip(preds[cond], lists)]))
        donor_hr, _ = per_row(preds[cond], donor_targets)
        res["donor_target_HR@10"] = float(donor_hr[10].mean())
        if cond != "raw":
            for name, cur, base in (("HR@10", hr[10], base_hr[10]), ("NDCG@10", ndcg[10], base_ndcg[10])):
                delta = cur - base
                res[f"d{name}"] = float(delta.mean())
                res[f"d{name}_ci95"] = bootstrap_ci(delta, args.bootstrap, rng)
            res["top10_overlap_with_raw"] = float(np.mean(
                [len({clean(x) for x in a} & {clean(x) for x in b}) / args.topk for a, b in zip(preds[cond], preds["raw"])]))
            res["top1_same_as_raw"] = float(np.mean([clean(a[0]) == clean(b[0]) for a, b in zip(preds[cond], preds["raw"])]))
        results[cond] = res

    summary = {"ckpt_dir": args.ckpt_dir, "test_rows": n, "target_in_own_list_rate": float(in_list.mean()),
               "geometry": geometry, "interventions": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntarget in own intuition list: {in_list.mean():.4f} of rows")
    print("geometry: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in geometry.items()))
    print("\ncondition           HR@10   NDCG@10  dHR@10 [95% CI]              overlap top1  copy   donor  HR@10(in/out list)")
    for cond, r in results.items():
        line = f"{cond:18s} {r['HR@10']:.4f}  {r['NDCG@10']:.4f}"
        if cond != "raw":
            lo, hi = r["dHR@10_ci95"]
            line += f"  {r['dHR@10']:+.4f} [{lo:+.4f}, {hi:+.4f}]  {r['top10_overlap_with_raw']:.3f}  {r['top1_same_as_raw']:.3f}"
        else:
            line += " " * 45
        line += (f"  {r['copy_rate']:.3f}  {r['donor_target_HR@10']:.4f}  "
                 f"{r['HR@10_target_in_own_list']:.4f}/{r['HR@10_target_not_in_own_list']:.4f}")
        print(line)
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
