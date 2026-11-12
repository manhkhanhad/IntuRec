#!/usr/bin/env python3
"""LLM setwise motive selector trained with GRPO (Rank-R1 / REARANK style), ranking-aware reward.

Policy: Qwen2.5-1.5B-Instruct + LoRA reads the user's recent purchases and the retriever's top-10 motive
candidates (options A-J, in retriever order) plus "K. None of these", and picks one option. The action
distribution is the next-token distribution restricted to the 11 option letters (no free-text
generation, so one forward per row).
Reward: the stored ranking-aware utility of the chosen motive (collab/rank_utility.py): the target's
length-normalised log-softmax among target + 20 SASRec hard negatives, under the frozen motive reader
with supports [m, m]; option K uses the reader's "No specific motive." score.
GRPO (R3 / DAPO-informed): G samples per row, group z-scored advantage, groups with zero reward
variance skipped (DAPO dynamic sampling), KL to the initial policy over the 11 options (adapter
disabled = reference), ONE pass over the train rows (as R3), lr warmup + cosine.
Test picks (argmax) are saved as motive ids (881 = no motive) for collab/motive_reader.py decode.
"""
import argparse
import ast
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

DATA = Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec/data/Toys_and_Games")
RANK = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_GRPO_select/rank_reward")
LETTERS = "ABCDEFGHIJK"
NULL = 881


def load_rewards(files, n_rows):
    """-> cand [n,10], R [n,11] (10 candidates + null), trank [n,11] target rank among the scored items."""
    cand = np.full((n_rows, 10), -1, np.int64)
    R = np.full((n_rows, 11), np.nan, np.float32)
    trank = np.full((n_rows, 11), np.nan, np.float32)
    for f in files:
        z = np.load(f)
        nll = z["ll"] / z["ntok"][:, None, :]
        lse = np.log(np.exp(nll - nll.max(-1, keepdims=True)).sum(-1)) + nll.max(-1)
        r = nll[:, :, 0] - lse                                    # [rows, 11]: col 0 null, 1.. candidates
        rk = (nll > nll[:, :, :1]).sum(-1)
        rows = z["row"]
        cand[rows] = z["motive"][:, 1:]
        R[rows] = np.concatenate([r[:, 1:], r[:, :1]], 1)          # options A-J = candidates, K = null
        trank[rows] = np.concatenate([rk[:, 1:], rk[:, :1]], 1)
    return cand, R, trank


def build_prompts(frame, cand, protos, max_hist):
    out = []
    for i in range(len(frame)):
        hist = ast.literal_eval(frame.history_item_title.iat[i])[-max_hist:]
        opts = "\n".join(f"{LETTERS[j]}. {protos[int(m)]['motive']}: {protos[int(m)]['reason']}"
                         for j, m in enumerate(cand[i]))
        out.append("A customer bought these toys and games, oldest to newest:\n"
                   + "\n".join(f"- {t}" for t in hist)
                   + "\n\nWhich purchase motive best explains the customer's NEXT purchase?\n"
                   + opts + f"\nK. None of these\n\nAnswer with one letter.\nAnswer:")
    return out


def option_logp(model, tok, prompts, letter_ids, device, max_len):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    logits = model(**enc).logits[:, -1, :]                       # left padding: last position = "Answer:"
    return torch.log_softmax(logits[:, letter_ids].float(), -1)  # [B, 11]


def evaluate(pick, R, trank, rows):
    r = R[rows, pick]
    tr = trank[rows, pick]
    return {"mean_reward": float(np.mean(r)), "target_top1": float(np.mean(tr == 0)),
            "target_top5": float(np.mean(tr < 5)), "target_mean_rank": float(np.mean(tr)),
            "picked_A": float(np.mean(pick == 0)), "picked_none": float(np.mean(pick == 10))}


@torch.no_grad()
def predict(model, tok, prompts, letter_ids, device, max_len, bs=32):
    model.eval()
    out = []
    for i in range(0, len(prompts), bs):
        out.append(option_logp(model, tok, prompts[i:i + bs], letter_ids, device, max_len).argmax(-1).cpu())
    model.train()
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--batch", type=int, default=16, help="rows per update")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.05, help="KL to the initial policy")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--full", action="store_true", help="full fine-tuning (frozen reference copy) instead of LoRA")
    p.add_argument("--max_hist", type=int, default=10)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_test", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    protos = json.load(open(DATA / "collab/motive_corpus/prototypes.json"))
    frames = {s: pd.read_csv(glob.glob(str(DATA / s / "*.csv"))[0]) for s in ("train", "test")}
    cand, R, trank = {}, {}, {}
    cand["train"], R["train"], trank["train"] = load_rewards([RANK / "rank_utility_train0.npz", RANK / "rank_utility_train1.npz"],
                                                             len(frames["train"]))
    cand["test"], R["test"], trank["test"] = load_rewards([RANK / "rank_utility_test.npz"], len(frames["test"]))
    users = frames["train"].user_id.to_numpy()
    uniq = np.unique(users)
    held = np.isin(users, rng.choice(uniq, size=len(uniq) // 10, replace=False))
    ok = ~np.isnan(R["train"]).any(1)
    tr_rows = rng.permutation(np.where(ok & ~held)[0])
    va_rows = np.where(ok & held)[0][:1000]
    te_rows = np.arange(len(frames["test"]))
    if a.limit_train:
        tr_rows = tr_rows[:a.limit_train]
    if a.limit_test:
        te_rows = te_rows[:a.limit_test]
    prompts = {s: build_prompts(frames[s], cand[s], protos, a.max_hist) for s in ("train", "test")}
    print(f"rows: train {len(tr_rows)}, held-out {len(va_rows)}, test {len(te_rows)}\n--- example prompt ---\n"
          f"{prompts['test'][0]}\n---", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]
    assert len(set(letter_ids)) == 11, letter_ids
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(device)
    ref_model = None
    if a.full:  # all weights trainable; the reference policy is a frozen copy of the initial model
        ref_model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(device).eval()
        for q in ref_model.parameters():
            q.requires_grad_(False)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print(f"full fine-tuning: {sum(q.numel() for q in model.parameters()):,} trainable parameters", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    n_steps = math.ceil(len(tr_rows) / a.batch)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, n_steps)

    results = {"args": vars(a)}
    base_pick = np.zeros(len(te_rows), np.int64)
    results["retriever_top1"] = evaluate(base_pick, R["test"], trank["test"], te_rows)
    results["null"] = evaluate(np.full(len(te_rows), 10), R["test"], trank["test"], te_rows)
    results["oracle_best"] = evaluate(np.nanargmax(R["test"][te_rows], 1), R["test"], trank["test"], te_rows)
    zs = predict(model, tok, [prompts["test"][i] for i in te_rows], letter_ids, device, a.max_len)
    results["zero_shot"] = evaluate(zs, R["test"], trank["test"], te_rows)
    for k in ("retriever_top1", "null", "zero_shot", "oracle_best"):
        print(k, results[k], flush=True)

    Rt = torch.from_numpy(R["train"])
    model.train()
    log = {"reward": [], "kl": [], "entropy": [], "skipped": [], "grad": []}
    for step in range(n_steps):
        rows = tr_rows[step * a.batch:(step + 1) * a.batch]
        logpi = option_logp(model, tok, [prompts["train"][i] for i in rows], letter_ids, device, a.max_len)
        with torch.no_grad():
            if ref_model is not None:
                ref = option_logp(ref_model, tok, [prompts["train"][i] for i in rows], letter_ids, device, a.max_len)
            else:
                with model.disable_adapter():
                    ref = option_logp(model, tok, [prompts["train"][i] for i in rows], letter_ids, device, a.max_len)
        acts = torch.multinomial(logpi.exp().detach(), a.group, replacement=True)               # [B, G]
        r = torch.gather(Rt[rows].to(device), 1, acts)
        std = r.std(1, keepdim=True)
        keep = (std.squeeze(1) > 1e-6).float()                                                    # DAPO dynamic sampling
        adv = (r - r.mean(1, keepdim=True)) / (std + 1e-6)
        pg = -((adv * torch.gather(logpi, 1, acts)).mean(1) * keep).sum() / keep.sum().clamp(min=1)
        kl = (logpi.exp() * (logpi - ref)).sum(-1).mean()
        loss = pg + a.beta * kl
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0).item()
        opt.step()
        sched.step()
        ent = -(logpi.exp() * logpi).sum(-1).mean().item()
        for k, v in (("reward", r.mean().item()), ("kl", kl.item()), ("entropy", ent),
                     ("skipped", 1 - keep.mean().item()), ("grad", gn)):
            log[k].append(v)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{n_steps}: " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items())
                  + f", lr {sched.get_last_lr()[0]:.2e}", flush=True)
            log = {k: [] for k in log}
        if (step + 1) % a.eval_every == 0:
            va = evaluate(predict(model, tok, [prompts["train"][i] for i in va_rows], letter_ids, device, a.max_len),
                          R["train"], trank["train"], va_rows)
            print(f"step {step + 1}: held-out {va}", flush=True)

    pk = predict(model, tok, [prompts["test"][i] for i in te_rows], letter_ids, device, a.max_len)
    results["llm_grpo"] = evaluate(pk, R["test"], trank["test"], te_rows)
    print("llm_grpo", results["llm_grpo"], flush=True)
    picks = np.where(pk == 10, NULL, cand["test"][te_rows, np.clip(pk, 0, 9)])
    np.save(os.path.join(a.out_dir, "picks_test.npy"), picks)
    model.save_pretrained(os.path.join(a.out_dir, "model" if a.full else "lora"))
    json.dump(results, open(os.path.join(a.out_dir, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
