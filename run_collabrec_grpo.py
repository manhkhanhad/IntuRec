"""GRPO on top of a CollabRec SFT checkpoint: R3's stage-2 "noise GRPO", ported to CollabRecF2.

R3 (research/R3/src/grpo_attention_tuning/{train_noise_grpo,grpo_trainer}.py) recipe, kept as-is:
  * exploration: each prompt is repeated num_generations (8) times; Gaussian noise (std 1.5) is added to
    the latent written into <|Thought|>; sample 0 of every group stays noise-free;
  * reward: -PPL of the ground-truth title under teacher forcing (no decoding), from the rollout policy;
  * advantage: reward - mean(noise-free rewards of the groups gathered together), divided by the L2 norm
    of the gathered advantage vector. R3 gathers the two groups of its two GPUs (swap_adjacent_blocks only
    permutes before a mean, so the baseline is the mean noise-free reward over both groups);
    --groups_per_gather 2 reproduces that on one GPU;
  * loss: PPO-clipped per-token objective (eps 0.2 / 0.28), token-mean per group, averaged over the groups
    of an optimizer step (R3: 256 rollouts = 32 prompts x 8), + KL to the reference (beta 0.01);
    num_iterations 1 (R3's launch script) => ratio == 1 and clipping is inactive;
  * AdamW, 200 warmup steps, cosine schedule, grad-norm clip 1.0 (HF Trainer default), 1 epoch.

Also matched to R3's runtime (trl 0.16.1 + transformers 4.48.2, the rrec env R3 runs in):
  * log-probs, the -PPL reward and the KL are computed in the model dtype (bf16): Qwen2 4.48 returns bf16
    logits and trl's selective_log_softmax keeps bf16 (--fp32_logps disables this);
  * each group is left-padded to a multiple of 8 tokens (DataCollatorForSeq2Seq pad_to_multiple_of=8), so
    positions are offset by the padding as in R3;
  * the reference decoder scores the POLICY's rollout embeddings (thought + noise), as R3 does;
  * fp32 noise is added into the bf16 embeddings (one rounding);
  * the last incomplete batch of prompts is dropped (RepeatRandomSampler): 1684 steps on Toys;
  * trainable = the latent module only (R3 freezes model.model and lm_head and trains model.attention); here
    the thought generator = InjectNet + no_support_proj (+ readout heads), lr 5e-5. --lr_llm > 0 also trains
    the LLM (a run with the LLM at 5e-5 diverged: job 357958).

Differences from R3 (by design):
  * the thought is computed once per prompt and broadcast to its 8 samples (R3 recomputes it on 8
    identical rows: same values and same gradient, 8x the cost; tests/test_grpo.py).

Usage (see submit_collabrec_grpo.sbatch):
  python run_collabrec_grpo.py --dataset Toys_and_Games --init_run_dir <SFT run folder> \
      --support_dir data/Toys_and_Games/collab/motive_corpus/stage2 --run_tag motive
  --probe_noise 0.05,0.2,1.5   reward statistics per noise level on --probe_prompts train prompts, then exit.
"""
import argparse
import datetime
import glob
import inspect
import json
import math
import os
import shutil
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM, get_cosine_schedule_with_warmup

from data.CollabRecDataset import CollabRecDataset
from model.CollabRecF2 import CollabRecF2
from run_collabrec import evaluate
from run_inturec_f2 import set_seed
from utils.util import calRecallAndNdcgFromFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GEN_KEYS = ("support_emb", "support_tokens", "support_logp")  # generate_embs inputs taken from a row
_KEEP_ARG = "logits_to_keep" if "logits_to_keep" in inspect.signature(Qwen2ForCausalLM.forward).parameters \
    else "num_logits_to_keep"


def parser_args(argv=None):
    p = argparse.ArgumentParser(description="CollabRec GRPO (R3 stage 2)")
    p.add_argument("--seed", type=int, default=42)  # R3 GRPO seed
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--dataset", type=str, default="Toys_and_Games")
    p.add_argument("--sample", type=int, default=-1, help="Train prompts (-1 = all, R3's launch setting).")
    p.add_argument("--init_run_dir", type=str, default="", help="CollabRec SFT run folder (best checkpoint).")
    # data: must match the SFT run
    p.add_argument("--support_dir", type=str, default="")
    p.add_argument("--support_k", type=int, default=10)
    p.add_argument("--rationale_tag", type=str, default="v5")
    p.add_argument("--max_rationale_len", type=int, default=64)
    p.add_argument("--motive_source", choices=["", "predicted", "noisy"], default="")
    p.add_argument("--noisy_precision", type=float, default=1.0)
    p.add_argument("--motive_scores", type=str, default="cls")
    p.add_argument("--reason_step", type=int, default=1)
    # GRPO (R3 defaults from scripts/archive/attention_grpo.sh)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--groups_per_gather", type=int, default=2, help="R3: 2 GPUs x 1 group per micro-batch.")
    p.add_argument("--batch_size", type=int, default=256, help="Rollouts per optimizer step.")
    p.add_argument("--noise_std", type=float, default=1.5)
    p.add_argument("--noise_relative", action="store_true",
                   help="std = noise_std * ||thought|| / sqrt(dim) (per prompt) instead of absolute.")
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=0.28)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0, help="Logit temperature (rollout and decode).")
    p.add_argument("--lr", type=float, default=5e-5, help="Thought generator (InjectNet / readout heads).")
    p.add_argument("--lr_llm", type=float, default=0.0,
                   help="Qwen backbone + lm_head: 0 = frozen (R3: only the latent module trains); "
                        "-1 = same as --lr (one Trainer lr); >0 = own lr.")
    p.add_argument("--fp32_logps", action="store_true",
                   help="Log-probs / reward / KL in fp32 instead of the model dtype (R3: bf16).")
    p.add_argument("--pad_multiple", type=int, default=8, help="Left-pad each group to a multiple (R3: 8).")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--nums_epoch", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--valid_every", type=int, default=200, help="Valid NTP every N steps (0 = start/end only).")
    p.add_argument("--valid_rows", type=int, default=2000, help="Valid rows for the NTP monitor (-1 = all).")
    p.add_argument("--resume_every", type=int, default=100, help="Resume state every N steps (0 = off).")
    # output
    p.add_argument("--ckpt_root", type=str, default="/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints")
    p.add_argument("--experiment", type=str, default="CollabRec_GRPO")
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--run_suffix", type=str, default="")
    p.add_argument("--train_or_test", choices=["train", "test"], default="train")
    # decode (as run_collabrec.py)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--decode_method", type=str, default="standard")
    p.add_argument("--skip_test", action="store_true")
    # probe
    p.add_argument("--probe_noise", type=str, default="", help="Comma-separated noise stds: probe and exit.")
    p.add_argument("--probe_prompts", type=int, default=64)
    p.add_argument("--stop_after_steps", type=int, default=0, help=argparse.SUPPRESS)  # tests: simulate preemption
    return p.parse_args(argv)


def auto_run_name(args):
    name = "grpo" + (f"-{args.run_tag}" if args.run_tag else "")
    name += f"-init_{os.path.basename(os.path.normpath(args.init_run_dir))}" if args.init_run_dir else ""
    name += f"_noise{args.noise_std}{'rel' if args.noise_relative else ''}_g{args.num_generations}"
    name += f"_kl{args.kl_beta}_lr{args.lr}_llm{args.lr_llm}"
    if args.sample > 0:
        name += f"_n{args.sample}"
    return f"{name}_{args.run_suffix}" if args.run_suffix else name


def run_dir(args):
    return f"{args.ckpt_root}/{args.dataset}/{args.experiment}/{args.run_name}"


# --------------------------------------------------------------------------------------------- model
def load_model(path, device, trainable=True):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = CollabRecF2.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.config.use_cache = False
    model.beta, model.gamma, model.sel_weight = 0.0, 0.0, 0.0  # GRPO objective only
    if not trainable:
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
    return model, tokenizer


def param_groups(model, lr, lr_llm):
    """Generator = everything outside the Qwen backbone / lm_head (InjectNet, readout heads)."""
    gen, llm, seen = [], [], set()
    for name, prm in model.named_parameters():
        if id(prm) in seen:  # lm_head is tied to embed_tokens
            continue
        seen.add(id(prm))
        (llm if name.startswith(("model.", "lm_head.")) else gen).append(prm)
    for prm in llm:
        prm.requires_grad_(lr_llm > 0)
    groups = [{"params": gen, "lr": lr}]
    if lr_llm > 0:
        groups.append({"params": llm, "lr": lr_llm})
    return groups, sum(p.numel() for p in gen), sum(p.numel() for p in llm)


def thought_id(model):
    return model.model.embed_tokens.num_embeddings - 3  # <|Thought|> (as CollabRecF2.generate_embs)


def prompt_tensors(row, pad_id, device, pad_multiple=8):
    """One dataset row -> ids [1, L] left-padded to a multiple of pad_multiple (R3's collator), its attention
    mask, target length T and generate_embs kwargs."""
    ids = torch.as_tensor(row["input_ids"])
    start = int(torch.as_tensor(row["attention_mask"]).argmax())
    labels = torch.as_tensor(row["labels"])[start:]
    ids = ids[start:]
    n_target = int((labels != -100).sum())
    assert (labels[-n_target:] != -100).all(), "the target must be the suffix of the sequence"
    n_pad = (-len(ids)) % pad_multiple if pad_multiple > 1 else 0
    mask = torch.cat([torch.zeros(n_pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)])
    ids = torch.cat([torch.full((n_pad,), pad_id, dtype=ids.dtype), ids])
    extra = {k: torch.as_tensor(np.asarray(row[k]))[None].to(device) for k in GEN_KEYS if k in row}
    return ids[None].to(device), mask[None].to(device), n_target, extra


def build_embeds(model, ids, extra):
    """Prompt+target embeddings with the model's thought written into <|Thought|>; returns (embeds, thought, pos)."""
    embeds, _, _, thought, _ = model.generate_embs(ids, **extra)
    pos = int((ids[0] == thought_id(model)).nonzero()[0])
    return embeds, thought, pos


def perturb(embeds, pos, noise):
    """Broadcast [1, L, d] -> [G, L, d] and add noise [G, d] at the thought position (differentiable)."""
    out = embeds.expand(noise.size(0), -1, -1).clone()
    out[:, pos] = (out[:, pos].float() + noise.float()).to(out.dtype)  # R3: fp32 noise added into bf16 in place
    return out


def target_logps(model, embeds, mask, ids, n_target, temperature, fp32=False):
    """Per-token log p(target) [G, T] under teacher forcing (R3 my_get_per_token_logps): no position_ids, so
    left padding offsets positions as in R3; log-softmax in the logits' dtype (bf16) unless fp32."""
    out = Qwen2ForCausalLM.forward(model, inputs_embeds=embeds, attention_mask=mask.expand(embeds.size(0), -1),
                                   use_cache=False, **{_KEEP_ARG: n_target + 1})
    logits = out.logits[:, :-1]  # positions L-T-1 .. L-2 predict the T target tokens
    logits = (logits.float() if fp32 else logits) / temperature
    target = ids[:, -n_target:].expand(embeds.size(0), -1)
    return torch.stack([row.log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)  # trl selective_log_softmax
                        for row, t in zip(logits, target)])


def sample_noise(thought, args, generator):
    dim = thought.size(-1)
    noise = torch.randn(args.num_generations, dim, device=thought.device, generator=generator) * args.noise_std
    if args.noise_relative:
        noise = noise * (thought.float().norm() / math.sqrt(dim))
    noise[0] = 0  # sample 0 of the group is the unperturbed policy (R3)
    return noise


def r3_advantages(rewards, num_generations):
    """rewards [U*G] for U gathered groups -> advantages (R3 grpo_trainer.py:166-176)."""
    clean = rewards.view(-1, num_generations)[:, 0]
    adv = rewards - clean.mean()
    return adv / (adv.norm() + 1e-6)


# ------------------------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(policy, ref, rows, args, generator, pad_id):
    """Rewards / old and reference log-probs for a list of prompts (one group each)."""
    groups = []
    for row in rows:
        ids, mask, n_target, extra = prompt_tensors(row, pad_id, policy.device, args.pad_multiple)
        embeds, thought, pos = build_embeds(policy, ids, extra)
        noise = sample_noise(thought, args, generator)
        noisy = perturb(embeds, pos, noise)
        old = target_logps(policy, noisy, mask, ids, n_target, args.temperature, args.fp32_logps)
        ref_lp = None
        if ref is not None and args.kl_beta > 0:  # R3: reference decoder on the policy's rollout embeddings
            ref_lp = target_logps(ref, noisy, mask, ids, n_target, args.temperature, args.fp32_logps)
        reward = (-torch.exp(-old.sum(-1) / n_target)).float()  # -PPL of the title, in the logps dtype (R3)
        groups.append({"ids": ids, "mask": mask, "n_target": n_target, "extra": extra, "noise": noise, "old": old,
                       "ref": ref_lp, "reward": reward, "thought_norm": thought.float().norm().item()})
    for start in range(0, len(groups), args.groups_per_gather):
        unit = groups[start:start + args.groups_per_gather]
        adv = r3_advantages(torch.cat([g["reward"] for g in unit]), args.num_generations)
        for i, g in enumerate(unit):
            g["adv"] = adv[i * args.num_generations:(i + 1) * args.num_generations]
    return groups


def group_loss(policy, g, args):
    embeds, _, pos = build_embeds(policy, g["ids"], g["extra"])
    lp = target_logps(policy, perturb(embeds, pos, g["noise"]), g["mask"], g["ids"], g["n_target"], args.temperature,
                      args.fp32_logps)
    old = g["old"] if args.num_iterations > 1 else lp.detach()
    ratio = torch.exp(lp - old)
    adv = g["adv"].unsqueeze(1)
    per_token = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon_high) * adv)
    kl = torch.zeros((), device=lp.device)
    if g["ref"] is not None:
        diff = g["ref"] - lp
        per_token_kl = torch.exp(diff) - diff - 1
        per_token = per_token + args.kl_beta * per_token_kl
        kl = per_token_kl.mean().detach()
    clipped = ((ratio * adv) > (torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon_high) * adv)).float().mean()
    return per_token.mean(), kl, clipped.detach()


# ---------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def valid_ntp(model, dataset, args, rows):
    """Average teacher-forced NTP of the noise-free model on the first `rows` valid rows (as run_collabrec)."""
    was_training = model.training
    model.eval()
    n = len(dataset) if rows < 0 else min(rows, len(dataset))
    losses = []
    for start in range(0, n, args.eval_batch_size):
        batch = [dataset[i] for i in range(start, min(start + args.eval_batch_size, n))]
        keys = ["input_ids", "attention_mask", "labels"] + [k for k in GEN_KEYS if k in batch[0]]
        inputs = {k: torch.as_tensor(np.stack([b[k] for b in batch])).to(model.device) for k in keys}
        losses.append(model.forward(**inputs).ntp_loss.item())
    model.train(was_training)
    return float(np.mean(losses))


# -------------------------------------------------------------------------------------------- resume
def resume_path(args):
    return os.path.join(run_dir(args), "resume", "state.pt")


def save_resume(model, optimizer, scheduler, args, **progress):
    path = resume_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), **progress}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_resume(model, optimizer, scheduler, args):
    path = resume_path(args)
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"resumed from {path}: epoch {state['epoch'] + 1}, step {state['step']}", flush=True)
    return state


# --------------------------------------------------------------------------------------------- train
def make_dataset(args, tokenizer, split):
    return CollabRecDataset(args.data_path, args.dataset, tokenizer, support_dir=args.support_dir,
                            support_k=args.support_k, sample=args.sample if split == "train" else -1,
                            dataset_type=split, reason_step=args.reason_step, seed=args.seed,
                            support_mode=args.support_mode, rationale_tag=args.rationale_tag,
                            max_rationale_len=args.max_rationale_len, aux_negatives=1,
                            motive_source=args.motive_source, noisy_precision=args.noisy_precision,
                            motive_scores=args.motive_scores)


def train(policy, ref, tokenizer, train_set, valid_set, args):
    prompts_per_step = args.batch_size // args.num_generations
    assert prompts_per_step * args.num_generations == args.batch_size
    steps_per_epoch = len(train_set) // prompts_per_step  # RepeatRandomSampler drops the incomplete batch
    groups, n_gen, n_llm = param_groups(policy, args.lr, args.lr_llm)
    print(f"trainable: generator {n_gen:,} (lr {args.lr}), LLM {n_llm if args.lr_llm > 0 else 0:,} "
          f"(lr {args.lr_llm}); {prompts_per_step} prompts x {args.num_generations} per step, "
          f"{steps_per_epoch} steps/epoch", flush=True)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps,
                                                steps_per_epoch * args.nums_epoch * args.num_iterations)
    state = load_resume(policy, optimizer, scheduler, args)
    start_epoch, start_step = (state["epoch"], state["step"]) if state else (0, 0)
    if state is None:
        print(f"Step [0] - Valid NTP {valid_ntp(policy, valid_set, args, args.valid_rows):.4f} (SFT init)", flush=True)
    pad_id = tokenizer.pad_token_id
    generator = torch.Generator(device=policy.device)
    done = 0
    policy.train()
    for epoch in range(start_epoch, args.nums_epoch):
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        log, t0 = {}, time.time()
        for step in range(start_step if epoch == start_epoch else 0, steps_per_epoch):
            generator.manual_seed(args.seed * 1_000_003 + epoch * 100_003 + step)  # resume-reproducible noise
            rows = [train_set[i] for i in order[step * prompts_per_step:(step + 1) * prompts_per_step]]
            gs = rollout(policy, ref, rows, args, generator, pad_id)
            for _ in range(args.num_iterations):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = kl_sum = clip_sum = 0.0
                for g in gs:
                    loss, kl, clipped = group_loss(policy, g, args)
                    (loss / len(gs)).backward()
                    loss_sum, kl_sum, clip_sum = loss_sum + loss.item(), kl_sum + kl.item(), clip_sum + clipped.item()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for grp in groups for p in grp["params"]],
                                                           args.max_grad_norm).item()
                optimizer.step()
                scheduler.step()
            rewards = torch.stack([g["reward"] for g in gs])  # [P, G]
            clean, noisy = rewards[:, 0], rewards[:, 1:]
            for k, v in {"loss": loss_sum / len(gs), "kl": kl_sum / len(gs), "clip": clip_sum / len(gs),
                         "grad_norm": grad_norm, "reward": rewards.mean().item(), "clean_ppl": -clean.mean().item(),
                         "noisy_ppl": -noisy.mean().item(), "noisy_beats_clean": (noisy > clean[:, None]).float().mean().item(),
                         "reward_std_in_group": rewards.std(1).mean().item(),
                         "thought_norm": float(np.mean([g["thought_norm"] for g in gs]))}.items():
                log.setdefault(k, []).append(v)
            done += 1
            if (step + 1) % args.log_every == 0:
                print(f"Step [{step + 1}/{steps_per_epoch}] " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items())
                      + f", lr {scheduler.get_last_lr()[0]:.2e}/{scheduler.get_last_lr()[-1]:.2e}"
                      + f", time {datetime.timedelta(seconds=int(time.time() - t0))}", flush=True)
                log, t0 = {}, time.time()
            if args.valid_every and (step + 1) % args.valid_every == 0:
                print(f"Step [{step + 1}] - Valid NTP {valid_ntp(policy, valid_set, args, args.valid_rows):.4f}", flush=True)
            if args.resume_every and (step + 1) % args.resume_every == 0:
                save_resume(policy, optimizer, scheduler, args, epoch=epoch, step=step + 1)
            if args.stop_after_steps and done >= args.stop_after_steps:
                raise SystemExit(f"stop_after_steps={args.stop_after_steps} reached (simulated preemption)")
        start_step = 0
        print(f"Epoch [{epoch + 1}] - Valid NTP {valid_ntp(policy, valid_set, args, args.valid_rows):.4f}", flush=True)
        if args.resume_every:
            save_resume(policy, optimizer, scheduler, args, epoch=epoch + 1, step=0)
    # R3 keeps the final model.
    policy.save_pretrained(run_dir(args))
    tokenizer.save_pretrained(run_dir(args))
    print(f"saved final model to {run_dir(args)}", flush=True)


@torch.no_grad()
def probe(policy, train_set, args, pad_id):
    """Reward statistics per noise std (no training): is the noise informative or just destructive?"""
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(train_set), size=min(args.probe_prompts, len(train_set)), replace=False)
    rows = [train_set[int(i)] for i in idx]
    generator = torch.Generator(device=policy.device)
    norms = []
    for std in [float(s) for s in args.probe_noise.split(",")]:
        args.noise_std = std
        generator.manual_seed(args.seed)
        gs = rollout(policy, None, rows, args, generator, pad_id)
        rewards = torch.stack([g["reward"] for g in gs])
        clean, noisy = -rewards[:, 0], -rewards[:, 1:]
        norms = [g["thought_norm"] for g in gs]
        rel = (std * math.sqrt(policy.config.hidden_size) / np.mean(norms)) if not args.noise_relative else std
        print(f"probe noise_std {std} ({'relative' if args.noise_relative else 'absolute'}; "
              f"||noise||/||thought|| ~ {rel:.2f}): clean PPL {clean.mean():.4f}, noisy PPL {noisy.mean():.4f} "
              f"(median {noisy.median():.4f}), noisy beats clean {(noisy < clean[:, None]).float().mean():.3f}, "
              f"in-group reward std {rewards.std(1).mean():.4f}", flush=True)
    print(f"probe thought norm: mean {np.mean(norms):.4f}, sd {np.std(norms):.4f}; "
          f"hidden {policy.config.hidden_size}", flush=True)


def run(argv=None):
    args = parser_args(argv)
    set_seed(args)
    if args.lr_llm < 0:
        args.lr_llm = args.lr  # R3 / HF Trainer: one lr for all trainable parameters
    args.run_name = args.run_name or auto_run_name(args)
    print(f"run: {args.dataset}/{args.experiment}/{args.run_name}  folder: {run_dir(args)}", flush=True)
    os.makedirs(run_dir(args), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.train_or_test == "train" or args.probe_noise:
        assert args.init_run_dir, "--init_run_dir (CollabRec SFT run folder) is required"
        policy, tokenizer = load_model(args.init_run_dir, device)
    else:
        policy, tokenizer = load_model(run_dir(args), device, trainable=False)
    args.support_mode = getattr(policy.config, "support_mode", "emb")
    policy.reason_step = args.reason_step
    with open(os.path.join(run_dir(args), "grpo_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.probe_noise:
        probe(policy, make_dataset(args, tokenizer, "train"), args, tokenizer.pad_token_id)
        return
    if args.train_or_test == "train":
        ref = load_model(args.init_run_dir, device, trainable=False)[0] if args.kl_beta > 0 else None
        if ref is not None:
            ref.reason_step = args.reason_step
        train(policy, ref, tokenizer, make_dataset(args, tokenizer, "train"), make_dataset(args, tokenizer, "valid"),
              args)
        del ref
    if args.skip_test:
        return
    policy.eval()
    args.data_result_meta_dir_path = f"{run_dir(args)}/result/{args.decode_method}"
    os.makedirs(args.data_result_meta_dir_path, exist_ok=True)
    info_files = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", "*.txt"))
    assert len(info_files) == 1, "There should be only one txt file in the directory."
    if args.decode_method == "d3":
        args.length_penalty = 0.0
    path = evaluate(policy, tokenizer, make_dataset(args, tokenizer, "test"), info_files[0], args)
    calRecallAndNdcgFromFile(path, info_files[0], ans_col_name="tgt_item")
    if args.train_or_test == "train" and os.path.exists(resume_path(args)):
        shutil.rmtree(os.path.dirname(resume_path(args)))  # finished: drop the resume state


if __name__ == "__main__":
    run()
