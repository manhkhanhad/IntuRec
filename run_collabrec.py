"""Train / evaluate CollabRecF2: latent collaborative reasoning without IntuRec's intuition.

Loss = NTP + beta * rationale InfoNCE. Supports and rationale targets come from
collab/build_stage2_support.py (--support_dir). Ablations: --support_k 0 (no supports,
thought = history residual only) and --beta 0 (no rationale alignment).
"""
import argparse
import collections
import datetime
import glob
import json
import os
import shutil
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
    parser.add_argument("--support_k", type=int, default=10, help="0 = no supports.")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight of the rationale-alignment InfoNCE loss.")
    parser.add_argument("--aux_temperature", type=float, default=0.05)
    parser.add_argument("--support_mode", choices=["text", "emb"], default="text",
                        help="text: rationale tokens pooled by IntuRec's InjectNet; emb: projected sentence embeddings.")
    parser.add_argument("--rationale_tag", type=str, default="v5")
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
    parser.add_argument("--collab_weight", type=float, default=0.0,
                        help="Weight of each collaborative InfoNCE relation on the refined thought (R3 Attempt 10 "
                             "style queue); 0 = off.")
    parser.add_argument("--collab_relations", type=str, default="user,target,motive",
                        help="Positive relations, one projection head each: user, target, motive.")
    parser.add_argument("--collab_bank_size", type=int, default=10000)
    parser.add_argument("--collab_temperature", type=float, default=0.1)
    parser.add_argument("--collab_min_token_accuracy", type=float, default=0.5,
                        help="Queue gate: share of target tokens predicted correctly.")
    parser.add_argument("--collab_max_positives", type=int, default=16)
    parser.add_argument("--collab_num_negatives", type=int, default=256)
    parser.add_argument("--collab_projection_dim", type=int, default=256)
    parser.add_argument("--eval_ckpt", type=str, default="",
                        help="Test mode: checkpoint sub-folder to decode (default: the saved best by valid NTP; "
                             "runs since 2026-10-23 save only the best, so per-epoch folders exist only in older runs). "
                             "Results go to a matching sub-folder of the result directory.")
    parser.add_argument("--readout", choices=["inject", "gated", "soft"], default="inject",
                        help="How the thought reads text supports: IntuRec's InjectNet, a relevance-gated readout "
                             "(selection BCE, weight --sel_weight), or a retriever-probability-weighted mixture.")
    parser.add_argument("--sel_weight", type=float, default=1.0, help="Weight of the gated readout's selection BCE.")
    parser.add_argument("--motive_source", choices=["", "predicted", "noisy"], default="",
                        help="Motive supports: retriever top-k ('predicted') or noisy oracle ('noisy': each true "
                             "motive kept with prob. --noisy_precision, else replaced by a top non-true prediction).")
    parser.add_argument("--utility_files", type=str, default="",
                        help="Comma-separated utility_train_fold*.npz (collab/motive_utility.py): train gate "
                             "labels = u(m) > --utility_margin instead of 'is a true motive'.")
    parser.add_argument("--utility_margin", type=float, default=0.0)
    parser.add_argument("--train_fold", type=str, default="",
                        help="'i/n': train only on users outside fold i of n (cross-fitted utility judges).")
    parser.add_argument("--noisy_precision", type=float, default=1.0,
                        help="Noisy oracle: probability a true motive is kept (final value under a curriculum; "
                             "valid / test always use it).")
    parser.add_argument("--noisy_precision_start", type=float, default=-1.0,
                        help="Curriculum clean -> noisy: train precision goes linearly per epoch from this value "
                             "to --noisy_precision (e.g. 1.0 -> 0.25 over 3 epochs: 1.0, 0.625, 0.25); <0 = off.")
    parser.add_argument("--motive_scores", type=str, default="cls", help="motive_corpus/retriever/<name>/scores_*.npy")
    parser.add_argument("--extra_test_predicted", action="store_true",
                        help="After the main decode, decode the test set again with the retriever's top-k motives "
                             "(result/<decode>/predicted/): the deployable number for a noisy-oracle run.")
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
    parser.add_argument("--resume_every", type=int, default=0,
                        help="Save a resume state (~9 GB: model + optimizer) every N optimizer steps and at each "
                             "epoch end; 0 = off (default, saves storage; a preempted run restarts from scratch). "
                             "An existing resume/state.pt is still loaded.")
    parser.add_argument("--stop_after_steps", type=int, default=0, help=argparse.SUPPRESS)  # tests: simulate preemption
    return parser.parse_args()


def auto_run_name(args):
    """Run folder name from the configuration, e.g. 'text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335655'."""
    name = args.support_mode
    if args.run_tag:
        name += f"-{args.run_tag}"
    if args.oracle:
        name += f"-oracle_{args.oracle}"
    if args.gamma > 0:
        name += f"-gamma{args.gamma}_{args.sup_anchor}"
    if args.motive_source:
        name += f"-{args.motive_source}" + (f"{args.noisy_precision}" if args.motive_source == "noisy" else "")
        if args.motive_source == "noisy" and args.noisy_precision_start >= 0:
            name = name[:-len(f"{args.noisy_precision}")] + f"{args.noisy_precision_start}to{args.noisy_precision}"
    if args.readout != "inject":
        name += f"-{args.readout}" + (f"{args.sel_weight}" if args.readout == "gated" else "")
        if args.utility_files:
            name += f"-util{args.utility_margin:g}"
    if args.collab_weight > 0:
        name += f"-cq{args.collab_weight}_{args.collab_relations.replace(',', '+')}"
    if args.train_fold:
        name += "-fold" + args.train_fold.replace("/", "of")
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
        config.readout = args.readout
        config.collab_queue = None
        if args.collab_weight > 0:
            config.collab_queue = {
                "relations": args.collab_relations.split(","), "bank_size": args.collab_bank_size,
                "projection_dim": args.collab_projection_dim, "temperature": args.collab_temperature,
                "min_token_accuracy": args.collab_min_token_accuracy,
                "max_positives": args.collab_max_positives, "num_negatives": args.collab_num_negatives}
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
    model.sel_weight = args.sel_weight
    model.aux_temperature = args.aux_temperature
    if model.collab:
        model.collab_weights = {rel: args.collab_weight for rel in model.collab["relations"]}
    return model, tokenizer


def initial_dataloaders(args, tokenizer):
    datasets = {split: CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                                        support_k=args.support_k, sample=args.sample, dataset_type=split,
                                        reason_step=args.reason_step, seed=args.seed,
                                        support_mode=args.support_mode, rationale_tag=args.rationale_tag,
                                        max_rationale_len=args.max_rationale_len, aux_negatives=args.aux_negatives,
                                        oracle=args.oracle,
                                        sup_anchor=args.sup_anchor if args.gamma > 0 else "",
                                        collab_queue=args.collab_weight > 0,
                                        motive_source=args.motive_source, noisy_precision=args.noisy_precision,
                                        motive_scores=args.motive_scores, train_fold=args.train_fold,
                                        utility_files=args.utility_files, utility_margin=args.utility_margin)
                for split in ("train", "valid", "test")}
    print(f"Train dataset sample: \n{datasets['train'].text_data[0]['all_input']}\n")
    # The train order is drawn from a generator re-seeded per epoch (seed + epoch), so a resumed run
    # replays exactly the same micro-batches.
    loaders = {split: DataLoader(ds, batch_size=args.train_micro_batch_size if split == "train" else args.eval_batch_size,
                                 shuffle=split == "train", num_workers=args.num_workers, pin_memory=True,
                                 generator=torch.Generator().manual_seed(args.seed) if split == "train" else None)
               for split, ds in datasets.items()}
    return loaders["train"], loaders["valid"], loaders["test"]


BANK_BUFFERS = ("bank_r", "bank_user", "bank_target", "bank_sample", "bank_motive", "bank_pointer", "bank_count")


def resume_path(args):
    return os.path.join(run_dir(args), "resume", "state.pt")


def save_resume(model, optimizer, scheduler, args, **progress):
    """Everything needed to continue training after a preemption (written atomically)."""
    path = resume_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "bank": {k: getattr(model, k) for k in BANK_BUFFERS} if model.collab else None,
             "rng": {"torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     "numpy": np.random.get_state()},
             **progress}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_resume(model, optimizer, scheduler, args):
    path = resume_path(args)
    if not os.path.exists(path):  # loaded even when saving is off (runs saved before the default changed)
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state["bank"] is not None:
        for k, v in state["bank"].items():
            getattr(model, k).copy_(v)
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    np.random.set_state(state["rng"]["numpy"])
    print(f"resumed from {path}: epoch {state['epoch'] + 1}, {state['ministeps_done']} micro-batches done, "
          f"epoch complete: {state['epoch_complete']}", flush=True)
    return state


def train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, args):
    output_dir = run_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    min_valid_loss = float("inf")
    accum = args.train_batch_size // args.train_micro_batch_size
    n_steps = (len(train_loader) + accum - 1) // accum
    resumed = load_resume(model, optimizer, scheduler, args)
    start_epoch, skip, totals_resumed = 0, 0, None
    if resumed is not None:
        min_valid_loss = resumed["min_valid_loss"]
        start_epoch = resumed["epoch"] + int(resumed["epoch_complete"])
        if not resumed["epoch_complete"]:
            skip, totals_resumed = resumed["ministeps_done"], resumed["totals"]
    opt_steps = 0  # optimizer steps taken in this process (for --stop_after_steps)
    for epoch in range(start_epoch, args.nums_epoch):
        print(f"Epoch [{epoch + 1}] - Start Training", flush=True)
        if args.motive_source == "noisy" and args.noisy_precision_start >= 0:
            # Curriculum: set before the loader is iterated (workers are re-created each epoch).
            frac = epoch / max(args.nums_epoch - 1, 1)
            p = args.noisy_precision_start + frac * (args.noisy_precision - args.noisy_precision_start)
            train_loader.dataset.noisy_precision = p
            print(f"Epoch [{epoch + 1}] - curriculum: train noisy precision {p:.3f} "
                  f"(valid/test {args.noisy_precision})", flush=True)
        model.train()
        totals = totals_resumed or {"loss": [], "ntp": [], "rationale": [], "support": [], "selection": []}
        totals_resumed = None
        step = {k: 0.0 for k in totals}
        collab_log = collections.defaultdict(list)
        optimizer.zero_grad()
        time_start = time.time()
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(args.seed + epoch)
        for ministep, minibatch in enumerate(train_loader):
            if ministep < skip:  # already trained on before the preemption
                continue
            minibatch = {k: v.to(device) for k, v in minibatch.items()}
            res = model.forward(**minibatch)
            step["loss"] += res.loss.item()
            step["ntp"] += res.ntp_loss.item()
            step["rationale"] += res.rationale_loss.item()
            step["support"] += res.support_loss.item()
            step["selection"] += res.selection_loss.item()
            for k, v in res.collab_losses.items():
                collab_log[f"{k}_loss"].append(v.item())
            for k, v in res.collab_rates.items():
                collab_log[k if k in ("admit", "queue") else f"{k}_pos"].append(v)
            (res.loss / accum).backward()
            if (ministep + 1) % accum == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                for k in totals:
                    totals[k].append(step[k] / accum)
                step = {k: 0.0 for k in totals}
                opt_steps += 1
                if args.resume_every and ((ministep + 1) // accum) % args.resume_every == 0:
                    save_resume(model, optimizer, scheduler, args, epoch=epoch, ministeps_done=ministep + 1,
                                epoch_complete=False, totals=totals, min_valid_loss=min_valid_loss)
                if args.stop_after_steps and opt_steps >= args.stop_after_steps:
                    raise SystemExit(f"stop_after_steps={args.stop_after_steps} reached (simulated preemption)")
                if (ministep + 1) % (10 * accum) == 0:
                    duration = time.time() - time_start
                    time_start = time.time()
                    print(f"Step [{(ministep + 1) // accum}/{n_steps}] - Avg Total Loss: {np.mean(totals['loss']):.4f}, "
                          f"Avg NTP Loss: {np.mean(totals['ntp']):.4f}, "
                          f"Avg Rationale Loss: {np.mean(totals['rationale']):.4f}, "
                          f"Avg Support Loss: {np.mean(totals['support']):.4f}, "
                          f"Avg Selection Loss: {np.mean(totals['selection']):.4f}, "
                          f"Current lr: {scheduler.get_last_lr()[0]:.10f}, "
                          f"Time for 10 steps: {datetime.timedelta(seconds=int(duration))}", flush=True)
                    if collab_log:
                        # last 10 steps: per-relation InfoNCE, share of anchors with a positive, gate rate, queue size
                        print("  collab " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in collab_log.items()), flush=True)
                        collab_log.clear()
        print(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(totals['loss']):.4f} Avg NTP Loss: {np.mean(totals['ntp']):.4f} "
              f"Avg Rationale Loss: {np.mean(totals['rationale']):.4f} "
              f"Avg Support Loss: {np.mean(totals['support']):.4f} "
              f"Avg Selection Loss: {np.mean(totals['selection']):.4f}", flush=True)

        print(f"Epoch [{epoch + 1}] - Start Validation", flush=True)
        model.eval()
        valid = {"loss": [], "ntp": [], "rationale": [], "support": [], "selection": []}
        gate_true, gate_false = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                res = model.forward(**batch)
                valid["loss"].append(res.loss.item())
                valid["ntp"].append(res.ntp_loss.item())
                valid["rationale"].append(res.rationale_loss.item())
                valid["support"].append(res.support_loss.item())
                valid["selection"].append(res.selection_loss.item())
                if getattr(model, "last_gates", None) is not None and "support_is_true" in batch:
                    is_true = batch["support_is_true"].bool()
                    gate_true += model.last_gates[is_true].tolist()
                    gate_false += model.last_gates[~is_true].tolist()
            # run_inturec_f2.py's validation loop scores the last training minibatch (it reads
            # `minibatch`, not `batch`); log the same quantity so the two logs line up. (After a resume that
            # skipped the whole epoch, the last batch was never moved to the device.)
            res = model.forward(**{k: v.to(device) for k, v in minibatch.items()})
        print(f"Epoch [{epoch + 1}] - Validation (IntuRec-style, last train batch) Avg Total Loss: {res.loss.item():.4f} "
              f"Avg NTP Loss: {res.ntp_loss.item():.4f} Avg Rationale Loss: {res.rationale_loss.item():.4f} "
              f"Avg Support Loss: {res.support_loss.item():.4f}", flush=True)
        print(f"Epoch [{epoch + 1}] - Validation Avg Total Loss: {np.mean(valid['loss']):.4f} "
              f"Avg NTP Loss: {np.mean(valid['ntp']):.4f} Avg Rationale Loss: {np.mean(valid['rationale']):.4f} "
              f"Avg Support Loss: {np.mean(valid['support']):.4f} "
              f"Avg Selection Loss: {np.mean(valid['selection']):.4f}", flush=True)
        if gate_true or gate_false:
            # mean gate on true vs wrong motives, and AUC of the gate as a true-motive detector
            t, f_ = np.array(gate_true), np.array(gate_false)
            auc = (np.mean(t[:, None] > f_[None, :]) + 0.5 * np.mean(t[:, None] == f_[None, :])
                   if len(t) and len(f_) and len(t) * len(f_) < 4e8 else float("nan"))
            print(f"Epoch [{epoch + 1}] - Valid gates: true {t.mean() if len(t) else float('nan'):.4f} "
                  f"(n {len(t)}), wrong {f_.mean() if len(f_) else float('nan'):.4f} (n {len(f_)}), "
                  f"AUC {auc:.4f}", flush=True)
        # Keep only the best checkpoint (lowest average valid NTP), overwriting the previous best.
        if np.mean(valid["ntp"]) < min_valid_loss:
            min_valid_loss = np.mean(valid["ntp"])
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Epoch [{epoch + 1}] - saved best checkpoint (valid NTP {min_valid_loss:.4f}) to {output_dir}", flush=True)
        skip = 0
        if args.resume_every:
            save_resume(model, optimizer, scheduler, args, epoch=epoch, ministeps_done=0, epoch_complete=True,
                        totals=None, min_valid_loss=min_valid_loss)


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
    # Optional preemption safety (args.partial_every > 0): predictions so far go to partial.json every N
    # batches and a restarted decode continues from there. Off by default (training-time decodes).
    every = getattr(args, "partial_every", 0)
    partial = f"{args.data_result_meta_dir_path}/partial.json"
    first = 0
    if every and os.path.exists(partial):
        with open(partial) as f:
            done = json.load(f)
        if len(done) <= len(rows):
            for i, (pred, sc) in enumerate(done):
                rows[i]["predict"], rows[i]["predict_score"] = pred, sc
            first = len(done)
            print(f"resuming decode from {partial}: {first}/{len(rows)} rows", flush=True)
    for start in tqdm(range(first, len(test_dataset), args.eval_batch_size)):
        batch = [test_dataset[i] for i in range(start, min(start + args.eval_batch_size, len(test_dataset)))]
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (max_len - len(b["input_ids"])) + b["input_ids"].tolist()
                                  for b in batch]).to(model.device)
        attention_mask = torch.tensor([[0] * (max_len - len(b["attention_mask"])) + b["attention_mask"].tolist()
                                       for b in batch]).to(model.device)
        extra = {}
        for key in ("support_emb", "support_tokens", "support_logp"):
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
        if every and (start // args.eval_batch_size + 1) % every == 0:
            n = start + len(batch)
            with open(partial + ".tmp", "w") as f:
                json.dump([(r["predict"], r["predict_score"]) for r in rows[:n]], f)
            os.replace(partial + ".tmp", partial)

    path = f"{args.data_result_meta_dir_path}/meta.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=4)
    if every and os.path.exists(partial):
        os.remove(partial)
    return path


def run():
    args = parser_args()
    set_seed(args)
    device = get_device(args)
    args.run_name = args.run_name or auto_run_name(args)
    print(f"run: {args.dataset}/{args.experiment}/{args.run_name}  checkpoints: {run_dir(args)}", flush=True)
    # Everything for a run lives in its folder: epoch_N/ checkpoints, result/<decode>/ predictions, logs.
    args.data_result_meta_dir_path = f"{run_dir(args)}/result/{args.decode_method}"
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
    if args.extra_test_predicted:
        # Same model, test supports = the retriever's top-k motives (deployable setting).
        print("Decode with predicted motives (retriever top-k):", flush=True)
        test_pred = CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                                     support_k=args.support_k, dataset_type="test", reason_step=args.reason_step,
                                     seed=args.seed, support_mode=args.support_mode,
                                     rationale_tag=args.rationale_tag, max_rationale_len=args.max_rationale_len,
                                     motive_source="predicted", motive_scores=args.motive_scores)
        args.data_result_meta_dir_path += "/predicted"
        os.makedirs(args.data_result_meta_dir_path, exist_ok=True)
        path = evaluate(model, tokenizer, test_pred, info_files[0], args)
        calRecallAndNdcgFromFile(path, info_files[0], ans_col_name="tgt_item")
    if args.train_or_test == "train" and os.path.exists(resume_path(args)):
        shutil.rmtree(os.path.dirname(resume_path(args)))  # run finished: drop the ~10 GB resume state


if __name__ == "__main__":
    run()
