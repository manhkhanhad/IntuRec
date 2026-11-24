#!/usr/bin/env python3
"""Pilot C (GPU part): R3-style thought interventions on the ORIGINAL LatentR3 checkpoints, one protocol for
both datasets and both stages (SFT = `latent`, RL = `nomul_attn_lr_5e-5`).

Runs R3's own, unmodified `src/prototype_grounding/experiment.py` (extract / generate) with its output root
redirected to <scratch>/IntuRec/checkpoints/<dataset>/MotiveRec_pilot/r3/<run>/ (nothing is written into the R3
repos). Steps, each skipped when its output exists (requeue-safe):
  1. extract train thoughts (needed for the train-mean and shuffle interventions) and test thoughts;
  2. constrained 10-beam decoding of the full test split with the thought slot set to
     raw / mask (zeros) / mean (train mean) / shuffle_user (a train thought of another user) /
     shuffle_target (a train thought with another target).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import R3_RUN, pilot_dir, r3_checkpoint, r3_generation_path  # noqa: E402

R3_REPO = Path("/common/home/users/m/mk.ngohuu.2024/research/R3")
MODES = ("raw", "mask", "mean", "shuffle_user", "shuffle_target")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=list(R3_RUN), required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--limit", type=int, default=-1, help="Smoke test: first N rows only.")
    args = p.parse_args()

    sys.path.insert(0, str(R3_REPO / "src" / "prototype_grounding"))
    import experiment as E  # R3's code, unmodified

    out_root = pilot_dir(args.dataset) / "r3" / R3_RUN[args.model]
    if args.limit > 0:
        out_root = out_root / f"smoke{args.limit}"
    E.ANALYSIS = out_root  # every reader/writer in experiment.py resolves ANALYSIS at call time
    E.configure_category(args.dataset)
    ckpt = r3_checkpoint(args.dataset, args.model)
    tag = "pilot"
    print(f"checkpoint {ckpt}\noutput {out_root}", flush=True)

    for split in ("train", "test"):
        if not (out_root / "evidence" / args.dataset / tag / split / "summary.json").exists():
            E.extract(split, args.batch_size, args.limit, ckpt, tag)
    for mode in args.modes.split(","):
        path = r3_generation_path(args.dataset, args.model, mode)
        if args.limit > 0:
            path = out_root / path.relative_to(pilot_dir(args.dataset) / "r3" / R3_RUN[args.model])
        if path.exists():
            print(f"skip {mode}: {path}", flush=True)
            continue
        E.generate_rankings("test", "all", 32, 42, 4, 0.1, 0.5, mode, args.batch_size, args.limit, ckpt, tag)


if __name__ == "__main__":
    main()
