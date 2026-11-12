"""DLRRec stage 2: latent SFT (DLR stage 2) + test decoding.

Sequence: [instruction][history] ### Response: <|Reason|> m_a m_b <|EndReason|>
          <|StartPredict|> title <|EndPredict|>
Loss = title NTP + lambda_code * reasoning-code NTP (both logged separately; the title part
includes <|StartPredict|>/<|EndPredict|>). --reasoning none trains the same model without
codes (control). Protocol matches run_inturec_f2.py / run_collabrec.py: lr 3e-4, effective
batch 128, 3 epochs, both the IntuRec-style (last train batch) and the real validation line.
Only the best checkpoint (lowest real valid title NTP) is kept, overwritten in place, and the
test set is decoded with it (like HF Trainer's load_best_model_at_end; IntuRec decodes the
last epoch instead).

Test decoding (--decode_codes, codes runs only):
  generated  greedy code generation restricted to code tokens (no repeats, <= 2 codes),
             then constrained title beam search
  gold       the row's own codes as a forced prefix (oracle upper bound)
"""

import argparse
import datetime
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList, \
    TemperatureLogitsWarper

from DLRRec.dlrrec_data import (DATA_PATH, END_REASON, REASON, START_PREDICT, DLRRecDataset, add_reasoning_tokens,
                                code_ids, load_prototypes)
from model.LogitsProcessor import IntuRecLogitsProcessor
from run_inturec_f2 import initial_optimizer_scheduler, set_seed
from utils.util import calRecallAndNdcgFromFile, get_hash, get_prefix_data_inturec_f2

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="CDs_and_Vinyl")
    p.add_argument("--reasoning", choices=["codes", "none"], default="codes")
    p.add_argument("--base_model", default="",
                   help="Default: DLRRec/checkpoints/align for codes, Qwen/Qwen2.5-1.5B for none.")
    p.add_argument("--lambda_code", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--nums_epoch", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--train_micro_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--train_or_test", default="train")
    p.add_argument("--eval_ckpt", default="", help="Test mode: checkpoint sub-folder, e.g. epoch_2 (default: the best checkpoint).")
    p.add_argument("--decode_codes", nargs="+", default=["generated", "gold"])
    p.add_argument("--max_codes", type=int, default=2)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--output_root", default=HERE)
    return p.parse_args()


def run_name(args):
    return f"{args.dataset}/{args.reasoning}/lr_{args.lr}_lam_{args.lambda_code}"


def split_losses(logits, labels, code_mask):
    """Mean NTP over reasoning-code positions and over title positions."""
    lg, lab, cm = logits[:, :-1].float(), labels[:, 1:], code_mask[:, 1:].bool()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100,
                         reduction="none").view(lab.shape)
    valid = lab != -100
    code = (ce * (valid & cm)).sum() / (valid & cm).sum().clamp_min(1)
    title = (ce * (valid & ~cm)).sum() / (valid & ~cm).sum().clamp_min(1)
    return code, title


def load_model(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = True  # SFT checkpoints are saved with use_cache=False
    return model, tok


def train(model, tok, args, device):
    out_dir = f"{args.output_root}/checkpoints/sft/{run_name(args)}"
    os.makedirs(out_dir, exist_ok=True)
    ds = {s: DLRRecDataset(args.dataset, tok, s, args.reasoning, args.sample, args.seed) for s in ("train", "valid")}
    loaders = {s: DataLoader(d, batch_size=args.train_micro_batch_size if s == "train" else args.eval_batch_size,
                             shuffle=s == "train", num_workers=args.num_workers, pin_memory=True)
               for s, d in ds.items()}
    accum = args.train_batch_size // args.train_micro_batch_size
    n_steps = (len(loaders["train"]) + accum - 1) // accum
    opt, sched = initial_optimizer_scheduler(
        model, lr=args.lr, wd=0.0, optimizer="adamw", epochs=args.nums_epoch, warmup_step=20,
        batchnum_per_epoch=(len(ds["train"]) + args.train_batch_size - 1) // args.train_batch_size,
        scheduler_type="linear")
    best = float("inf")
    for epoch in range(args.nums_epoch):
        print(f"Epoch [{epoch + 1}] - Start Training", flush=True)
        model.train()
        tot = {"loss": [], "title": [], "code": []}
        step = {k: 0.0 for k in tot}
        opt.zero_grad()
        t0 = time.time()
        for ministep, mb in enumerate(loaders["train"]):
            mb = {k: v.to(device) for k, v in mb.items()}
            logits = model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits
            code, title = split_losses(logits, mb["labels"], mb["code_mask"])
            loss = title + args.lambda_code * code
            (loss / accum).backward()
            step["loss"] += loss.item(); step["title"] += title.item(); step["code"] += code.item()
            if (ministep + 1) % accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
                for k in tot:
                    tot[k].append(step[k] / accum)
                step = {k: 0.0 for k in tot}
                if (ministep + 1) % (10 * accum) == 0:
                    print(f"Step [{(ministep + 1) // accum}/{n_steps}] - Avg Total Loss: {np.mean(tot['loss']):.4f}, "
                          f"Avg NTP Loss: {np.mean(tot['title']):.4f}, Avg Code Loss: {np.mean(tot['code']):.4f}, "
                          f"Current lr: {sched.get_last_lr()[0]:.10f}, "
                          f"Time for 10 steps: {datetime.timedelta(seconds=int(time.time() - t0))}", flush=True)
                    t0 = time.time()
        print(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(tot['loss']):.4f} Avg NTP Loss: {np.mean(tot['title']):.4f} "
              f"Avg Code Loss: {np.mean(tot['code']):.4f}", flush=True)

        print(f"Epoch [{epoch + 1}] - Start Validation", flush=True)
        model.eval()
        val = {"loss": [], "title": [], "code": []}
        with torch.no_grad():
            for b in loaders["valid"]:
                b = {k: v.to(device) for k, v in b.items()}
                code, title = split_losses(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits,
                                           b["labels"], b["code_mask"])
                val["loss"].append((title + args.lambda_code * code).item())
                val["title"].append(title.item()); val["code"].append(code.item())
            # IntuRec's validation loop scores the last training minibatch; log it too for comparison.
            code, title = split_losses(model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits,
                                       mb["labels"], mb["code_mask"])
        print(f"Epoch [{epoch + 1}] - Validation (IntuRec-style, last train batch) Avg Total Loss: "
              f"{(title + args.lambda_code * code).item():.4f} Avg NTP Loss: {title.item():.4f} "
              f"Avg Code Loss: {code.item():.4f}", flush=True)
        print(f"Epoch [{epoch + 1}] - Validation Avg Total Loss: {np.mean(val['loss']):.4f} "
              f"Avg NTP Loss: {np.mean(val['title']):.4f} Avg Code Loss: {np.mean(val['code']):.4f}", flush=True)
        # Keep only the best checkpoint (disk quota); best_epoch.txt records which epoch it is.
        if np.mean(val["title"]) < best:
            best = np.mean(val["title"])
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            with open(f"{out_dir}/best_epoch.txt", "w") as f:
                f.write(f"epoch_{epoch + 1}\n")
            print(f"Epoch [{epoch + 1}] - new best, saved to {out_dir}", flush=True)
    return out_dir


@torch.no_grad()
def generate_codes(model, tok, prompts, args, device):
    """Greedy reasoning-code generation restricted to code tokens / <|EndReason|>, no repeats."""
    codes = code_ids(tok, len(load_prototypes()))
    code_set = torch.tensor(codes, device=device)
    end_reason = tok.convert_tokens_to_ids(END_REASON)
    seqs = [p + [tok.convert_tokens_to_ids(REASON)] for p in prompts]
    chosen = [[] for _ in prompts]
    done = [False] * len(prompts)
    for step in range(args.max_codes):
        L = max(len(s) for s in seqs)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(s)) + s for s in seqs], device=device)
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
        allowed = torch.full_like(logits, -float("inf"))
        allowed[:, code_set] = 0
        allowed[:, end_reason] = 0 if step > 0 else -float("inf")  # at least one code
        logits = logits + allowed
        for i, c in enumerate(chosen):
            if c:
                logits[i, c] = -float("inf")
        nxt = logits.argmax(-1).tolist()
        for i, t in enumerate(nxt):
            if not done[i]:
                if t == end_reason:
                    done[i] = True
                else:
                    seqs[i].append(t)
                    chosen[i].append(t)
    k_of = {t: k for k, t in enumerate(codes)}
    return [[k_of[t] for t in c] for c in chosen]


@torch.no_grad()
def evaluate(model, tok, args, device, decode):
    ds = DLRRecDataset(args.dataset, tok, "test", args.reasoning, args.sample, args.seed)
    info = glob.glob(os.path.join(DATA_PATH, args.dataset, "info", "*.txt"))[0]
    hash_dict = get_prefix_data_inturec_f2(info, tok)
    gen_config = GenerationConfig(num_beams=args.topk, num_return_sequences=args.topk, length_penalty=1.0,
                                  pad_token_id=model.config.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|EndPredict|>"),
                                  max_new_tokens=128)
    codes = code_ids(tok, len(load_prototypes()))
    start_id, end_id = tok.convert_tokens_to_ids(START_PREDICT), tok.convert_tokens_to_ids("<|EndPredict|>")
    reason_id, end_reason_id = tok.convert_tokens_to_ids(REASON), tok.convert_tokens_to_ids(END_REASON)
    rows = ds.data
    model.eval()
    for start in tqdm(range(0, len(ds), args.eval_batch_size)):
        idx = list(range(start, min(start + args.eval_batch_size, len(ds))))
        prompts = [ds.prompt_ids(rows[i]) for i in idx]
        if args.reasoning == "codes":
            pred_codes = generate_codes(model, tok, prompts, args, device) if decode == "generated" \
                else [rows[i]["codes"] for i in idx]
            seqs = [p + [reason_id] + [codes[k] for k in c] + [end_reason_id, start_id] for p, c in zip(prompts, pred_codes)]
        else:
            pred_codes = [[] for _ in idx]
            seqs = [p + [start_id] for p in prompts]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(s)) + s for s in seqs], device=device)
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
        lp = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=1.0),
            IntuRecLogitsProcessor(tokenizer=tok, model=model, num_beams=args.topk, guide_token=START_PREDICT,
                                   prefix_allowed_tokens_fn=lambda b, x: hash_dict.get(get_hash(x), []))])
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_config,
                             return_dict_in_generate=True, output_scores=True, logits_processor=lp)
        new = out.sequences[:, L:]
        texts = []
        for s in new.tolist():
            s = s[:s.index(end_id)] if end_id in s else s
            texts.append(tok.decode(s, skip_special_tokens=True))
        scores = out.sequences_scores.tolist()
        for j, i in enumerate(idx):
            rows[i]["predict"] = texts[j * args.topk:(j + 1) * args.topk]
            rows[i]["predict_score"] = scores[j * args.topk:(j + 1) * args.topk]
            rows[i]["pred_codes"] = pred_codes[j]
    res_dir = f"{args.output_root}/results/{run_name(args)}/{decode}" + (f"/{args.eval_ckpt}" if args.eval_ckpt else "")
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/meta.json", "w") as f:
        json.dump(rows, f, indent=1)
    if args.reasoning == "codes" and decode == "generated":
        has = [r for r in rows if r["codes"]]
        hit = np.mean([bool(set(r["codes"]) & set(r["pred_codes"])) for r in has])
        exact = np.mean([set(r["codes"]) == set(r["pred_codes"]) for r in has])
        print(f"[{decode}] generated codes vs gold: any-overlap {hit:.4f}, exact set {exact:.4f}", flush=True)
    print(f"[{decode}] test metrics:", flush=True)
    calRecallAndNdcgFromFile(f"{res_dir}/meta.json", info, ans_col_name="tgt_item")


def main():
    args = parse_args()
    set_seed(args)
    device = torch.device("cuda")
    if args.train_or_test == "train":
        base = args.base_model or (f"{args.output_root}/checkpoints/align" if args.reasoning == "codes"
                                   else "Qwen/Qwen2.5-1.5B")
        model, tok = load_model(base, device)
        if args.reasoning == "none":
            add_reasoning_tokens(tok, len(load_prototypes()))
            model.resize_token_embeddings(max(len(tok), model.get_input_embeddings().weight.shape[0]))
        model.config.pad_token_id = tok.eos_token_id
        model.config.use_cache = False
        print(f"base model: {base}", flush=True)
        out_dir = train(model, tok, args, device)
        del model
        torch.cuda.empty_cache()
        model, tok = load_model(out_dir, device)
        with open(f"{out_dir}/best_epoch.txt") as f:
            print(f"test checkpoint: {out_dir} (best = {f.read().strip()})", flush=True)
    else:
        ckpt_root = f"{args.output_root}/checkpoints/sft/{run_name(args)}"
        if not args.eval_ckpt and not os.path.exists(f"{ckpt_root}/config.json"):
            # Runs from the per-epoch-save version keep the best epoch only as a sub-folder name.
            with open(f"{ckpt_root}/best_epoch.txt") as f:
                args.eval_ckpt = f.read().strip()
        path = f"{ckpt_root}/{args.eval_ckpt}" if args.eval_ckpt else ckpt_root
        model, tok = load_model(path, device)
        print(f"test checkpoint: {path}", flush=True)
    for decode in (args.decode_codes if args.reasoning == "codes" else ["none"]):
        evaluate(model, tok, args, device, decode)


if __name__ == "__main__":
    main()
