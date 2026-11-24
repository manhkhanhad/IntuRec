"""Train / evaluate CollabRecF2: latent collaborative reasoning without IntuRec's intuition.

Loss = NTP (+ beta * L_align rationale alignment, + gamma * L_support; both 0 by default = MotiveRec). Supports and rationale targets come from
collab/build_stage2_support.py (--support_dir). Ablations: --support_k 0 (no supports,
thought = history residual only) and --beta 0 (no rationale alignment).
"""
import argparse
import datetime
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

from data.CollabRecDataset import CollabRecDataset
from model.CollabRecF2 import CollabRecF2
from model.LogitsProcessor import IntuRecLogitsProcessor
from run_inturec_f2 import get_device, initial_optimizer_scheduler, set_seed
from utils.util import calRecallAndNdcgFromFile, get_hash, get_prefix_data_inturec_f2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "CollabRecF2"


def parser_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="CDs_and_Vinyl")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--support_dir", type=str, default="./data/CDs_and_Vinyl/collab/stage2")
    parser.add_argument("--support_k", type=int, default=3, help="Supports per row; MotiveRec default 3 (2026-11-15). 0 = no supports.")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Weight of L_align (rationale-alignment InfoNCE). MotiveRec: 0 (NTP only); CollabRec used 0.1.")
    parser.add_argument("--aux_temperature", type=float, default=0.05)
    parser.add_argument("--support_mode", choices=["text", "emb"], default="text",
                        help="text: rationale tokens pooled by IntuRec's InjectNet; emb: projected sentence embeddings.")
    parser.add_argument("--motive_pool", choices=["inject", "llm_mean", "llm_mean_ln"], default="inject",
                        help="text mode: InjectNet last-token attention over token embeddings (inject), or masked "
                             "mean of the LLM's last-layer hidden states over each motive/rationale (llm_mean).")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm; 0 = no clipping.")
    parser.add_argument("--rationale_tag", type=str, default="v5")
    parser.add_argument("--motive_corpus", type=str, default="",
                        help="Motive corpus under data/<ds>/collab whose prototype ids index the support texts "
                             "(e.g. motive_corpus_v6/P1000); needed by --insert_p / --motive_head_weight / --eval_motives.")
    parser.add_argument("--insert_p", type=float, default=0.0,
                        help="Train only: prob. that a row whose top-k misses its true motives gets them inserted (0 = off).")
    parser.add_argument("--insert_p_start", type=float, default=-1.0,
                        help="Curriculum: train insert prob. goes linearly per epoch from this to --insert_p "
                             "(e.g. 1.0 -> 0 over 3 epochs: 1.0, 0.5, 0.0); <0 = off.")
    parser.add_argument("--save_last", action="store_true",
                        help="Save the last epoch instead of the best-valid one (curriculum: last = deployment ratio).")
    parser.add_argument("--test_best", action="store_true",
                        help="Train mode: after the usual last-epoch test (result/<decode>/), also reload the saved "
                             "best-valid-NTP checkpoint and decode the test set into result/<decode>/best/.")
    parser.add_argument("--insert_fill", choices=["pred", "similar"], default="pred",
                        help="Fillers next to inserted true motives: retriever's non-true picks, or nearest prototypes.")
    parser.add_argument("--motive_head_weight", type=float, default=0.0,
                        help="LM-side motive head CE weight (true motive from the final hidden state); 0 = off.")
    parser.add_argument("--eval_motives", choices=["top", "true", "null", "random"], default="top",
                        help="Decode diagnostics: motives fed at decode (true = oracle, use with --eval_split valid).")
    parser.add_argument("--eval_split", choices=["test", "valid"], default="test",
                        help="Split decoded by --train_or_test test (valid for oracle diagnostics).")
    parser.add_argument("--readout", choices=["inject", "gated"], default="inject",
                        help="How the thought reads text supports: IntuRec's InjectNet, or the relevance-gated "
                             "readout of CollabRec 356575 (selection BCE, weight --sel_weight).")
    parser.add_argument("--sel_weight", type=float, default=1.0, help="Weight of the gated readout's selection BCE.")
    parser.add_argument("--noisy_precision", type=float, default=-1.0,
                        help="Noisy-oracle motives (356575; needs --motive_corpus): train/valid slots keep each true "
                             "motive with this prob., else a top non-true prediction; test = retriever top-k. <0 = off.")
    parser.add_argument("--noisy_precision_start", type=float, default=-1.0,
                        help="Curriculum: train precision goes linearly per epoch from this to --noisy_precision "
                             "(1.0 -> 0.25 over 3 epochs: 1.0, 0.625, 0.25); valid stays at --noisy_precision. <0 = off.")
    parser.add_argument("--max_rationale_len", type=int, default=64)
    parser.add_argument("--aux_negatives", type=int, default=63, help="Random train rationales per row (text mode).")
    parser.add_argument("--run_tag", type=str, default="", help="Extra checkpoint/result sub-folder, e.g. 'motive'.")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Weight of the contrastive support loss (IntuRec intuition-loss analog); 0 = off.")
    parser.add_argument("--sup_anchor", choices=["title", "motive"], default="title",
                        help="Anchor of the support loss: target title, or the row's ground-truth motive prototypes.")
    parser.add_argument("--oracle", choices=["", "own", "own_motive", "own_motive_plus_pred"], default="",
                        help="Analysis only: feed each row's own rationale ('own'), own motive prototypes "
                             "('own_motive'), or own motives + (support_k - 2) predicted motives "
                             "('own_motive_plus_pred') as the support, for every split (uses test ground truth).")
    parser.add_argument("--eval_ckpt", type=str, default="",
                        help="Test mode: checkpoint sub-folder to decode (default: the saved best by valid NTP; "
                             "runs since 2026-10-23 save only the best, so per-epoch folders exist only in older runs). "
                             "Results go to a matching sub-folder of the result directory.")
    parser.add_argument("--reason_step", type=int, default=1)
    parser.add_argument("--nums_epoch", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--train_micro_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--ckpt_root", type=str, default="/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints",
                        help="Checkpoints go to <ckpt_root>/<dataset>/<experiment>/<run_name>/ (scratch by default).")
    parser.add_argument("--experiment", type=str, default="CollabRec", help="Experiment folder, e.g. CollabRec_oracle.")
    parser.add_argument("--run_name", type=str, default="",
                        help="Run folder; default is built from the configuration (+ --run_suffix).")
    parser.add_argument("--run_suffix", type=str, default="", help="Appended to the default run name, e.g. job id.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_or_test", type=str, default="train")  # train / test
    parser.add_argument("--decode_method", type=str, default="standard")  # standard / d3
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def auto_run_name(args):
    """Run folder name from the configuration, e.g. 'text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335655'."""
    name = args.support_mode
    if args.run_tag:
        name += f"-{args.run_tag}"
    if args.motive_pool != "inject":
        name += f"-{args.motive_pool}"
    if args.grad_clip > 0:
        name += f"-clip{args.grad_clip}"
    if args.rationale_tag != "v5":
        name += f"-rat{args.rationale_tag}"
    if args.insert_p > 0 or args.insert_p_start >= 0:
        name += (f"-ins{args.insert_p_start}to{args.insert_p}" if args.insert_p_start >= 0 else f"-ins{args.insert_p}")
        name += "sim" if args.insert_fill == "similar" else ""
    if args.motive_head_weight > 0:
        name += f"-mh{args.motive_head_weight}"
    if args.noisy_precision >= 0:
        name += "-noisy" + (f"{args.noisy_precision_start}to" if args.noisy_precision_start >= 0 else "") \
                + f"{args.noisy_precision}"
    if args.readout != "inject":
        name += f"-{args.readout}{args.sel_weight}"
    if args.oracle:
        name += f"-oracle_{args.oracle}"
    if args.gamma > 0:
        name += f"-gamma{args.gamma}_{args.sup_anchor}"
    name += f"_k{args.support_k}_beta{args.beta}_rs{args.reason_step}_lr{args.lr}"
    return f"{name}_{args.run_suffix}" if args.run_suffix else name


def run_dir(args):
    """Checkpoints: <ckpt_root>/<dataset>/<experiment>/<run_name>/ (best + epoch_N/)."""
    return f"{args.ckpt_root}/{args.dataset}/{args.experiment}/{args.run_name}"


def initial_model(args, device):
    if args.train_or_test == "test":
        args.base_model = run_dir(args)
        if args.eval_ckpt:
            args.base_model += f"/{args.eval_ckpt}"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if args.train_or_test == "train":
        tokens = tokenizer.additional_special_tokens + ["<|Thought|>", "<|StartPredict|>", "<|EndPredict|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        tokenizer.padding_side = "left"
        config.support_mode = args.support_mode
        config.motive_pool = args.motive_pool
        config.motive_head_weight = args.motive_head_weight
        config.readout = args.readout
        if args.motive_head_weight > 0:
            assert args.motive_corpus, "--motive_head_weight needs --motive_corpus"
            with open(os.path.join(args.data_path, args.dataset, "collab", args.motive_corpus, "prototypes.json")) as f:
                config.n_motives = len(json.load(f)) + 1  # + "no motive"
        if args.support_mode == "emb":
            rationale_emb = np.load(os.path.join(args.support_dir, "rationale_emb_train.npy"), mmap_mode="r")
            config.rationale_dim = rationale_emb.shape[1]
    model = CollabRecF2.from_pretrained(args.base_model, config=config, torch_dtype=torch.bfloat16,
                                        trust_remote_code=True).to(device)
    if args.train_or_test == "train":
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EndPredict|>")
        model.config.use_cache = False
        if args.beta > 0 and args.support_mode == "emb":
            model.rationale_bank = torch.from_numpy(
                np.load(os.path.join(args.support_dir, "rationale_emb_train.npy"))).float().to(device)
    model.reason_step = args.reason_step
    model.beta = args.beta
    model.gamma = args.gamma
    model.aux_temperature = args.aux_temperature
    model.sel_weight = args.sel_weight if getattr(model, "readout", "inject") == "gated" else 0.0
    return model, tokenizer


def initial_dataloaders(args, tokenizer):
    datasets = {split: CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                                        support_k=args.support_k, sample=args.sample, dataset_type=split,
                                        reason_step=args.reason_step, seed=args.seed,
                                        support_mode=args.support_mode, rationale_tag=args.rationale_tag,
                                        max_rationale_len=args.max_rationale_len, aux_negatives=args.aux_negatives,
                                        oracle=args.oracle,
                                        sup_anchor=args.sup_anchor if args.gamma > 0 else "",
                                        motive_corpus=args.motive_corpus, insert_p=args.insert_p,
                                        insert_fill=args.insert_fill,
                                        eval_motives=args.eval_motives if split == args.eval_split else "top",
                                        noisy_precision=args.noisy_precision,
                                        sel_labels=args.readout == "gated")
                for split in ("train", "valid", "test")}
    if args.eval_split == "valid":
        # decode the valid split (prompt only, no target tokens) in place of test
        datasets["test"] = CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                                            support_k=args.support_k, sample=args.sample, dataset_type="valid",
                                            reason_step=args.reason_step, seed=args.seed,
                                            support_mode=args.support_mode, rationale_tag=args.rationale_tag,
                                            max_rationale_len=args.max_rationale_len,
                                            aux_negatives=args.aux_negatives, oracle=args.oracle,
                                            motive_corpus=args.motive_corpus, eval_motives=args.eval_motives,
                                            decode_format=True)
    print(f"Train dataset sample: \n{datasets['train'].text_data[0]['all_input']}\n")
    loaders = {split: DataLoader(ds, batch_size=args.train_micro_batch_size if split == "train" else args.eval_batch_size,
                                 shuffle=split == "train", num_workers=args.num_workers, pin_memory=True)
               for split, ds in datasets.items()}
    return loaders["train"], loaders["valid"], loaders["test"]


def train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, args):
    output_dir = run_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    min_valid_loss = float("inf")
    accum = args.train_batch_size // args.train_micro_batch_size
    n_steps = (len(train_loader) + accum - 1) // accum
    for epoch in range(args.nums_epoch):
        print(f"Epoch [{epoch + 1}] - Start Training", flush=True)
        if args.insert_p_start >= 0:
            frac = epoch / max(args.nums_epoch - 1, 1)
            train_loader.dataset.insert_p = args.insert_p_start + frac * (args.insert_p - args.insert_p_start)
            print(f"Epoch [{epoch + 1}] - train insert_p {train_loader.dataset.insert_p:.3f}", flush=True)
        if args.noisy_precision >= 0 and args.noisy_precision_start >= 0:
            frac = epoch / max(args.nums_epoch - 1, 1)
            p = args.noisy_precision_start + frac * (args.noisy_precision - args.noisy_precision_start)
            train_loader.dataset.noisy_precision = p
            print(f"Epoch [{epoch + 1}] - curriculum: train noisy precision {p:.3f} "
                  f"(valid {args.noisy_precision}, test retriever top-k)", flush=True)
        model.train()
        totals = {"loss": [], "ntp": [], "rationale": [], "support": [], "motive": [], "selection": []}
        step = {k: 0.0 for k in totals}
        optimizer.zero_grad()
        time_start = time.time()
        for ministep, minibatch in enumerate(train_loader):
            minibatch = {k: v.to(device) for k, v in minibatch.items()}
            res = model.forward(**minibatch)
            step["loss"] += res.loss.item()
            step["ntp"] += res.ntp_loss.item()
            step["rationale"] += res.rationale_loss.item()
            step["support"] += res.support_loss.item()
            step["motive"] += res.motive_loss.item() if hasattr(res, "motive_loss") else 0.0
            step["selection"] += res.selection_loss.item() if hasattr(res, "selection_loss") else 0.0
            (res.loss / accum).backward()
            if (ministep + 1) % accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                for k in totals:
                    totals[k].append(step[k] / accum)
                step = {k: 0.0 for k in totals}
                if (ministep + 1) % (10 * accum) == 0:
                    duration = time.time() - time_start
                    time_start = time.time()
                    print(f"Step [{(ministep + 1) // accum}/{n_steps}] - Avg Total Loss: {np.mean(totals['loss']):.4f}, "
                          f"Avg NTP Loss: {np.mean(totals['ntp']):.4f}, "
                          f"Avg Rationale Loss: {np.mean(totals['rationale']):.4f}, "
                          f"Avg Support Loss: {np.mean(totals['support']):.4f}, "
                          f"Avg Motive Loss: {np.mean(totals['motive']):.4f}, "
                          f"Avg Selection Loss: {np.mean(totals['selection']):.4f}, "
                          f"Current lr: {scheduler.get_last_lr()[0]:.10f}, "
                          f"Time for 10 steps: {datetime.timedelta(seconds=int(duration))}", flush=True)
        print(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(totals['loss']):.4f} Avg NTP Loss: {np.mean(totals['ntp']):.4f} "
              f"Avg Rationale Loss: {np.mean(totals['rationale']):.4f} "
              f"Avg Support Loss: {np.mean(totals['support']):.4f}", flush=True)

        print(f"Epoch [{epoch + 1}] - Start Validation", flush=True)
        model.eval()
        valid = {"loss": [], "ntp": [], "rationale": [], "support": [], "motive": [], "selection": []}
        motive_hits, gate_true, gate_false = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                res = model.forward(**batch)
                valid["loss"].append(res.loss.item())
                valid["ntp"].append(res.ntp_loss.item())
                valid["rationale"].append(res.rationale_loss.item())
                valid["support"].append(res.support_loss.item())
                valid["motive"].append(res.motive_loss.item() if hasattr(res, "motive_loss") else 0.0)
                if getattr(model, "last_motive_hits", None) is not None:
                    motive_hits += model.last_motive_hits.tolist()
                valid["selection"].append(res.selection_loss.item() if hasattr(res, "selection_loss") else 0.0)
                if getattr(model, "last_gates", None) is not None and "support_is_true" in batch:
                    is_true = batch["support_is_true"] > 0.5
                    gate_true += model.last_gates[is_true].tolist()
                    gate_false += model.last_gates[~is_true].tolist()
            # run_inturec_f2.py's validation loop scores the last training minibatch (it reads
            # `minibatch`, not `batch`); log the same quantity so the two logs line up.
            res = model.forward(**minibatch)
        print(f"Epoch [{epoch + 1}] - Validation (IntuRec-style, last train batch) Avg Total Loss: {res.loss.item():.4f} "
              f"Avg NTP Loss: {res.ntp_loss.item():.4f} Avg Rationale Loss: {res.rationale_loss.item():.4f} "
              f"Avg Support Loss: {res.support_loss.item():.4f}", flush=True)
        print(f"Epoch [{epoch + 1}] - Validation Avg Total Loss: {np.mean(valid['loss']):.4f} "
              f"Avg NTP Loss: {np.mean(valid['ntp']):.4f} Avg Rationale Loss: {np.mean(valid['rationale']):.4f} "
              f"Avg Support Loss: {np.mean(valid['support']):.4f} Avg Motive Loss: {np.mean(valid['motive']):.4f}"
              + (f" Motive head top-1 hit: {np.mean(motive_hits):.4f}" if motive_hits else "")
              + (f" Avg Selection Loss: {np.mean(valid['selection']):.4f}" if args.readout == "gated" else ""),
              flush=True)
        if gate_true and gate_false:
            from sklearn.metrics import roc_auc_score
            t, f_ = np.array(gate_true), np.array(gate_false)
            auc = roc_auc_score(np.r_[np.ones(len(t)), np.zeros(len(f_))], np.r_[t, f_])
            print(f"Epoch [{epoch + 1}] - Valid gates: true {t.mean():.4f} (n {len(t)}) wrong {f_.mean():.4f} "
                  f"(n {len(f_)}) AUC {auc:.4f}", flush=True)
        # Keep only the best checkpoint (lowest average valid NTP), overwriting the previous best.
        if np.mean(valid["ntp"]) < min_valid_loss or args.save_last:
            min_valid_loss = min(min_valid_loss, np.mean(valid["ntp"]))
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Epoch [{epoch + 1}] - saved {'last' if args.save_last else 'best'} checkpoint "
                  f"(valid NTP {np.mean(valid['ntp']):.4f}) to {output_dir}", flush=True)


@torch.no_grad()
def evaluate(model, tokenizer, test_dataset, info_file, args):
    hash_dict = get_prefix_data_inturec_f2(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

    generation_config = GenerationConfig(
        num_beams=args.topk, length_penalty=args.length_penalty, num_return_sequences=args.topk,
        pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id, max_new_tokens=128)
    start_id = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
    end_id = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]

    rows = test_dataset.data
    for start in tqdm(range(0, len(test_dataset), args.eval_batch_size)):
        batch = [test_dataset[i] for i in range(start, min(start + args.eval_batch_size, len(test_dataset)))]
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (max_len - len(b["input_ids"])) + b["input_ids"].tolist()
                                  for b in batch]).to(model.device)
        attention_mask = torch.tensor([[0] * (max_len - len(b["attention_mask"])) + b["attention_mask"].tolist()
                                       for b in batch]).to(model.device)
        extra = {}
        for key in ("support_emb", "support_tokens"):
            if key in batch[0]:
                extra[key] = torch.tensor(np.stack([b[key] for b in batch])).to(model.device)
        logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=args.temperature),
            IntuRecLogitsProcessor(tokenizer=tokenizer, model=model, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                   num_beams=args.topk, guide_token="<|StartPredict|>"),
        ])
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config,
                             return_dict_in_generate=True, output_scores=True, logits_processor=logits_processor, **extra)

        seqs = out.sequences
        pos_s = seqs.eq(start_id).long().argmax(dim=1)
        pos_e = seqs.eq(end_id).long().argmax(dim=1)
        arange_ids = torch.arange(seqs.shape[-1], device=seqs.device).unsqueeze(0)
        between = (arange_ids > pos_s.unsqueeze(1)) & (arange_ids < pos_e.unsqueeze(1))
        segments = torch.split(seqs.masked_select(between), between.sum(dim=1).tolist())
        texts = tokenizer.batch_decode(segments, skip_special_tokens=True)
        scores = out.sequences_scores.tolist()
        for i in range(len(batch)):
            rows[start + i]["predict"] = texts[i * args.topk:(i + 1) * args.topk]
            rows[start + i]["predict_score"] = scores[i * args.topk:(i + 1) * args.topk]

    path = f"{args.data_result_meta_dir_path}/meta.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=4)
    return path


def run():
    args = parser_args()
    set_seed(args)
    device = get_device(args)
    args.run_name = args.run_name or auto_run_name(args)
    print(f"run: {args.dataset}/{args.experiment}/{args.run_name}  checkpoints: {run_dir(args)}", flush=True)
    # Everything for a run lives in its folder: epoch_N/ checkpoints, result/<decode>/ predictions, logs.
    args.data_result_meta_dir_path = f"{run_dir(args)}/result/{args.decode_method}"
    if args.eval_split != "test" or args.eval_motives != "top":
        args.data_result_meta_dir_path += f"/{args.eval_split}_{args.eval_motives}"
    if args.train_or_test == "test" and args.eval_ckpt:
        args.data_result_meta_dir_path += f"/{args.eval_ckpt}"
    os.makedirs(args.data_result_meta_dir_path, exist_ok=True)

    model, tokenizer = initial_model(args, device)
    train_loader, valid_loader, test_loader = initial_dataloaders(args, tokenizer)
    if args.train_or_test == "train":
        optimizer, scheduler = initial_optimizer_scheduler(
            model, lr=args.lr, wd=0.0, optimizer="adamw", epochs=args.nums_epoch, warmup_step=20,
            batchnum_per_epoch=(len(train_loader.dataset) + args.train_batch_size - 1) // args.train_batch_size,
            scheduler_type="linear")
        # As in run_inturec_f2.py, the test set is decoded with the model after the last epoch.
        train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, args)

    info_files = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", "*.txt"))
    assert len(info_files) == 1, "There should be only one txt file in the directory."
    if args.decode_method == "d3":
        args.length_penalty = 0.0
    path = evaluate(model, tokenizer, test_loader.dataset, info_files[0], args)
    calRecallAndNdcgFromFile(path, info_files[0], ans_col_name="tgt_item")

    if args.train_or_test == "train" and args.test_best and not args.save_last:
        # Second decode with the best-valid-NTP checkpoint saved in the run folder (valid uses retrieved motives).
        print("Test with the saved best-valid checkpoint -> result/<decode>/best", flush=True)
        del model
        torch.cuda.empty_cache()
        args.train_or_test, args.eval_ckpt = "test", ""
        model, tokenizer = initial_model(args, device)
        args.data_result_meta_dir_path += "/best"
        os.makedirs(args.data_result_meta_dir_path, exist_ok=True)
        path = evaluate(model, tokenizer, test_loader.dataset, info_files[0], args)
        calRecallAndNdcgFromFile(path, info_files[0], ans_col_name="tgt_item")


if __name__ == "__main__":
    run()
