#!/usr/bin/env python3
"""Per-thought latent geometry of multi-thought MotiveRec (model/MotiveRecMS.py). Read-only on the checkpoint.

For N rows of a split, rebuilds the thoughts exactly as MotiveRecMS.generate_embs does and records, per thought t:
  thought   the vector written into <|Thought|> slot t (the model's input at that slot)
  state     the last-layer hidden state at slot t in the final pass (all S thoughts in place)
plus the history query h_u (last-layer state at slot 1 before any thought) and the last-layer state at
<|StartPredict|> (the position that predicts the first item token) after 0, 1, ..., S thoughts are in place.
Reports (definitions of collab/inturec_latent_diagnostics.py / multistep_geometry.py):
  per_step / per_step_state / history_query   collapse statistics across users: effective rank, participation ratio,
                PC1 variance share, PCs for 90%, random-pair cosine, cosine to the mean, mean L2 norm
  cross_step    S x S: mean cosine cos(thought_i, thought_j) of the same user vs of two different users
                (raw, and "centered" = after removing each step's mean vector over users)
  motive_alignment[t]   cosine of the centered thought t with the centered value vector W_v_2(pool(motive)) of the
                row's true motive(s), of the motives fed to the model, and of random motives; and the mean rank
                / top-1 / top-10 rate of a true motive among all prototypes by that cosine (chance top-10 = 10 / P)
  start_state   how the <|StartPredict|> state moves as thoughts are added: cosine between consecutive prefixes,
                norm of the change relative to the state norm, and same-user vs other-user cosine
  refine_attention      (refine mode) entropy / max weight of the selection, weight on true motives when present
Writes <out_dir>/ms_geometry_<split>.json and ms_latents_<split>.npz (float16).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ARGV = sys.argv[1:]


def take(flag, default=None, cast=str):
    if flag in ARGV:
        i = ARGV.index(flag)
        v = cast(ARGV[i + 1])
        del ARGV[i:i + 2]
        return v
    return default


RUN_DIR = take("--run_dir")  # .../<dataset>/<experiment>/<run_name>
MODE = take("--thought_support")
S = take("--reason_step", cast=int)
K = take("--support_k", 3, int)
LIMIT = take("--limit", 2000, int)
SPLIT = take("--split", "valid")
BATCH = take("--batch_size", 8, int)
DEVICE = take("--device", "cuda:0")
OUT = take("--out_dir", "")
assert RUN_DIR and MODE and S, "need --run_dir --thought_support --reason_step"
run = Path(RUN_DIR)
sys.argv = ["x", "--thought_support", MODE, "--device", DEVICE, "--dataset", run.parts[-3],
            "--base_model", "Qwen/Qwen2.5-1.5B",
            "--support_dir", f"data/{run.parts[-3]}/collab/motive_corpus_v6/P1000/stage2",
            "--motive_corpus", "motive_corpus_v6/P1000", "--train_or_test", "test", "--support_mode", "text",
            "--support_k", str(K), "--beta", "0", "--reason_step", str(S), "--train_micro_batch_size", "2",
            "--eval_batch_size", "2", "--ckpt_root", str(run.parents[2]), "--experiment", run.parts[-2],
            "--run_name", run.parts[-1]]
import run_motiverec_ms  # noqa: E402,F401  (patches run_collabrec's model class)
import run_collabrec as rc  # noqa: E402


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
            "cos_to_mean": float((n @ (mean / np.linalg.norm(mean))).mean()),
            "mean_norm": float(np.linalg.norm(x, axis=1).mean())}


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cross_step(th, perm):
    z = unit(th.astype(np.float64))
    n = len(z)
    return (np.einsum("nid,njd->ij", z, z) / n).round(4).tolist(), (np.einsum("nid,njd->ij", z, z[perm]) / n).round(4).tolist()


@torch.no_grad()
def main():
    rng = np.random.default_rng(0)
    args = rc.parser_args()
    dev = rc.get_device(args)
    model, tok = rc.initial_model(args, dev)
    model.eval()
    loaders = dict(zip(("train", "valid", "test"), rc.initial_dataloaders(args, tok)))
    ds = loaders[SPLIT].dataset
    idx = np.sort(rng.permutation(len(ds))[:LIMIT]) if 0 < LIMIT < len(ds) else np.arange(len(ds))
    pad = model.config.pad_token_id
    tid = model.model.embed_tokens.num_embeddings - 3
    sid = tok.convert_tokens_to_ids("<|StartPredict|>")
    P = len(ds.true_motives and ds.support_table) if ds.true_motives is not None else 0
    n_proto = len(json.load(open(f"data/{run.parts[-3]}/collab/motive_corpus_v6/P1000/prototypes.json")))
    table = torch.tensor(ds.support_table[:n_proto]).to(dev)
    values = torch.cat([model.injectnet.W_v_2(model.pool_rationales(table[i:i + 128])).float().cpu()
                        for i in range(0, n_proto, 128)]).numpy()  # [P, dim] motive value vectors

    th_all, st_all, hq_all, sp_all, slots_all, true_all, attn_all = [], [], [], [], [], [], []
    for b0 in range(0, len(idx), BATCH):
        items = [ds[int(i)] for i in idx[b0:b0 + BATCH]]
        width = max(len(it["input_ids"]) for it in items)
        ids = torch.tensor(np.stack([np.pad(it["input_ids"], (width - len(it["input_ids"]), 0), constant_values=pad)
                                     for it in items])).to(dev)
        sup = torch.tensor(np.stack([it["support_tokens"] for it in items])).to(dev)
        am = (ids != pad).int()
        rows = torch.arange(ids.shape[0], device=dev)
        first = (ids == tid).int().argmax(1)
        sp = (ids == sid).int().argmax(1)
        assert bool(((ids == tid).sum(1) == S).all()) and bool((sp == first + S).all())
        emb = model.model.embed_tokens(ids)
        pooled = None
        if MODE == "refine" and S > 1:
            bsz, k, length = sup.shape
            pooled = model.pool_rationales(sup.reshape(-1, length)).reshape(bsz, k, -1)
        thoughts, starts, attn = [], [], None
        for t in range(S):
            now = first + t
            mid = model.model(inputs_embeds=emb, attention_mask=am).last_hidden_state
            starts.append(mid[rows, sp].float().cpu())
            if t == 0:
                hq_all.append(mid[rows, first].float().cpu())
            if t > 0 and MODE in ("first", "first_norm"):
                th = mid[rows, now - 1]
                if MODE == "first_norm":
                    th = th * (ref / th.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(th.dtype)
            elif t > 0 and MODE == "refine":
                th, attn = model.refine(mid[rows, now - 1], pooled)
            else:
                th = model.inject(mid[rows, now].unsqueeze(1), model.step_supports(sup, t, S))
            if t == 0:
                ref = th.float().norm(dim=-1, keepdim=True)
            thoughts.append(th.float().cpu())
            emb = emb.index_put((rows, now), th.to(emb.dtype))
        ref_emb, _, _ = model.generate_embs(ids, support_tokens=sup)
        assert float((ref_emb.float() - emb.float()).abs().max()) < 1e-2, "probe differs from the model's own thoughts"
        fin = model.model(inputs_embeds=emb, attention_mask=am).last_hidden_state
        starts.append(fin[rows, sp].float().cpu())
        th_all.append(torch.stack(thoughts, 1))
        st_all.append(torch.stack([fin[rows, first + t].float().cpu() for t in range(S)], 1))
        sp_all.append(torch.stack(starts, 1))
        for j, i in enumerate(idx[b0:b0 + BATCH]):
            row = ds.row_ids[int(i)]
            slots_all.append(np.asarray(ds.motive_slots(row)))
            true_all.append(list(dict.fromkeys(ds.true_motives[row])))
        if attn is not None:
            attn_all.append(attn.float().cpu())
        if (b0 // BATCH) % 25 == 0:
            print(f"rows {b0 + len(items)}/{len(idx)}", flush=True)

    th, st, hq, sp = (torch.cat(x).numpy() for x in (th_all, st_all, hq_all, sp_all))
    n = len(th)
    perm = rng.permutation(n)
    perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)
    w_raw, c_raw = cross_step(th, perm)
    w_cen, c_cen = cross_step(th - th.mean(0, keepdims=True), perm)
    vc = unit(values - values.mean(0, keepdims=True))
    align = []
    for t in range(S):
        z = unit(th[:, t] - th[:, t].mean(0, keepdims=True))
        sims = z @ vc.T  # [n, P]
        tr, fed, rnd, ranks = [], [], [], []
        for i in range(n):
            if true_all[i]:
                tr.append(sims[i, true_all[i]].max())
                ranks.append(int((sims[i] > sims[i, true_all[i]].max()).sum()) + 1)
            fed.append(sims[i, slots_all[i]].mean())
            rnd.append(sims[i, rng.integers(0, len(vc), 10)].mean())
        ranks = np.array(ranks)
        align.append({"cos_true_motive": float(np.mean(tr)), "cos_fed_motives": float(np.mean(fed)),
                      "cos_random_motive": float(np.mean(rnd)), "true_rank_mean": float(ranks.mean()),
                      "true_top1": float((ranks <= 1).mean()), "true_top10": float((ranks <= 10).mean()),
                      "chance_top10": 10 / len(vc)})
    zs = unit(sp.astype(np.float64))
    start = {"cos_consecutive_prefixes": [float((zs[:, k] * zs[:, k + 1]).sum(-1).mean()) for k in range(S)],
             "relative_change": [float((np.linalg.norm(sp[:, k + 1] - sp[:, k], axis=-1) / np.linalg.norm(sp[:, k], axis=-1)).mean())
                                 for k in range(S)],
             "per_prefix": [collapse_stats(sp[:, k], rng) for k in range(S + 1)]}
    summary = {"run_dir": RUN_DIR, "split": SPLIT, "n_rows": int(n), "thought_support": MODE, "reason_step": S, "support_k": K,
               "token_embedding_norm": float(model.model.embed_tokens.weight.float().norm(dim=-1).mean()),
               "history_query": collapse_stats(hq, rng),
               "per_step": [collapse_stats(th[:, t], rng) for t in range(S)],
               "per_step_state": [collapse_stats(st[:, t], rng) for t in range(S)],
               "cross_step_same_user_cos": w_raw, "cross_step_other_user_cos": c_raw,
               "cross_step_same_user_cos_centered": w_cen, "cross_step_other_user_cos_centered": c_cen,
               "motive_alignment": align, "start_state": start,
               "true_motive_in_fed_slots": float(np.mean([bool(set(t) & set(s.tolist())) for t, s in zip(true_all, slots_all)]))}
    if attn_all:
        a = torch.cat(attn_all).numpy()
        has = [i for i in range(n) if set(true_all[i]) & set(slots_all[i].tolist())]
        summary["refine_attention"] = {
            "entropy": float(-(a * np.log(np.clip(a, 1e-12, None))).sum(-1).mean()), "uniform_entropy": float(np.log(a.shape[1])),
            "max_weight": float(a.max(-1).mean()),
            "weight_on_true_when_present": float(np.mean([a[i, [j for j, m in enumerate(slots_all[i]) if m in set(true_all[i])]].sum()
                                                          for i in has])) if has else None,
            "expected_if_uniform": float(np.mean([sum(m in set(true_all[i]) for m in slots_all[i]) / a.shape[1] for i in has])) if has else None,
            "argmax_is_true_when_present": float(np.mean([slots_all[i][a[i].argmax()] in set(true_all[i]) for i in has])) if has else None}
    out = Path(OUT or os.path.join(RUN_DIR, "diagnostics"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ms_geometry_{SPLIT}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / f"ms_latents_{SPLIT}.npz", thoughts=th.astype(np.float16), states=st.astype(np.float16),
                        history_query=hq.astype(np.float16), start_states=sp.astype(np.float16), rows=idx)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
