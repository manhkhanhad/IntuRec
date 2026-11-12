#!/usr/bin/env python3
"""One-off (2026-10-23): move every run's logs, Slurm outputs and predictions into its run folder
<CKPT_ROOT>/CDs_and_Vinyl/<experiment>/<run>/ (train.log, slurm-<job>.out, result/<decode>/...).
Pipeline logs (log/collab/*) go to <CKPT_ROOT>/CDs_and_Vinyl/collab_pipeline/logs/."""

import shutil
from pathlib import Path

REPO = Path("/common/home/users/m/mk.ngohuu.2024/research/IntuRec")
ROOT = Path("/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/CDs_and_Vinyl")
L, S, R = "log/CDs_and_Vinyl", "run_logs", "result_data/CDs_and_Vinyl"

CR = "CollabRec"
OR = "CollabRec_oracle"
IR = "IntuRec_original"
runs = {
    # run folder: {destination name: source path (relative to repo)}
    f"{CR}/emb_k10_beta0.1_rs1_lr0.0003_job335192": {
        "train.log": f"{L}/collabrec-k_10-beta_0.1-rs_1.log", "slurm-335192.out": f"{S}/CollabRec.335192.out",
        "result/standard": f"{R}/{CR}/emb_k10_beta0.1_rs1_lr0.0003_job335192/standard"},
    f"{CR}/emb_k10_beta0.1_rs1_lr0.0003_run2-rtx3090-micro4-bestepoch2": {
        "train.log": f"{L}/collabrec-k_10-beta_0.1-rs_1.run2_bestepoch2.log",
        "result/standard": f"{R}/{CR}/emb_k10_beta0.1_rs1_lr0.0003_run2-rtx3090-micro4-bestepoch2/standard"},
    f"{CR}/text_k10_beta0.1_rs1_lr0.0003_job335212": {
        "train.log": f"{L}/collabrec-text-k_10-beta_0.1-rs_1.log", "slurm-335212.out": f"{S}/CollabRec.335212.out",
        "test_epoch_2_job335282.log": f"{L}/collabrec-text-k_10-beta_0.1-rs_1-test-epoch_2.log",
        "slurm-335282.out": f"{S}/CollabRec.335282.out",
        "result/standard": f"{R}/{CR}/text_k10_beta0.1_rs1_lr0.0003_job335212/standard"},
    f"{CR}/text-motive_k10_beta0.1_rs1_lr0.0003_job335237": {
        "train.log": f"{L}/collabrec-text-motive-k_10-beta_0.1-rs_1.log", "slurm-335237.out": f"{S}/CollabRec.335237.out",
        "result/standard": f"{R}/{CR}/text-motive_k10_beta0.1_rs1_lr0.0003_job335237/standard"},
    f"{CR}/text-item_k10_beta0.1_rs1_lr0.0003_job335328": {
        "train.log": f"{L}/collabrec-text-item-k_10-beta_0.1-rs_1.log", "slurm-335328.out": f"{S}/CollabRec.335328.out",
        "result/standard": f"{R}/{CR}/text-item_k10_beta0.1_rs1_lr0.0003_job335328/standard"},
    f"{CR}/text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335655-cancelled": {
        "train.log": f"{L}/collabrec-text-item-gamma_1.0_title-k_10-beta_0.1-rs_1.log", "slurm-335655.out": f"{S}/CollabRec.335655.out"},
    f"{CR}/text-item-gamma1.0_motive_k10_beta0.1_rs1_lr0.0003_job335656-cancelled": {
        "train.log": f"{L}/collabrec-text-item-gamma_1.0_motive-k_10-beta_0.1-rs_1.log", "slurm-335656.out": f"{S}/CollabRec.335656.out"},
    f"{CR}/text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335680-cancelled": {
        "train.log": f"{L}/CollabRec/text-item-gamma1.0_title_k10_beta0.1_rs1_lr0.0003_job335680.log", "slurm-335680.out": f"{S}/CollabRec.335680.out"},
    f"{CR}/text-item-gamma1.0_motive_k10_beta0.1_rs1_lr0.0003_job335681-cancelled": {
        "train.log": f"{L}/CollabRec/text-item-gamma1.0_motive_k10_beta0.1_rs1_lr0.0003_job335681.log", "slurm-335681.out": f"{S}/CollabRec.335681.out"},
    f"{OR}/text_k0_beta0.0_rs1_lr0.0003_job335415": {
        "train.log": f"{L}/collabrec-text-k_0-beta_0-rs_1.log", "slurm-335415.out": f"{S}/CollabRec.335415.out",
        "result/standard": f"{R}/{OR}/text_k0_beta0.0_rs1_lr0.0003_job335415/standard"},
    f"{OR}/text_k0_beta0.0_rs1_lr0.0003_job335336-failed-quota": {
        "train.log": f"{L}/collabrec-text-k_0-beta_0-rs_1.failed_quota.log", "slurm-335336.out": f"{S}/CollabRec.335336.out"},
    f"{OR}/text-oracle_own_k1_beta0.0_rs1_lr0.0003_job335337-preempted": {
        "train.log": f"{L}/collabrec-text-oracle_own-k_1-beta_0-rs_1.preempted.log", "slurm-335337.out": f"{S}/CollabRec.335337.out"},
    f"{OR}/text-oracle_own_k1_beta0.0_rs1_lr0.0003_job335636-cancelled-epoch1": {
        "train.log": f"{L}/collabrec-text-oracle_own-k_1-beta_0-rs_1.log", "slurm-335636.out": f"{S}/CollabRec.335636.out"},
    f"{OR}/text-oracle_own_motive_k2_beta0.0_rs1_lr0.0003_job335338-failed-quota": {
        "train.log": f"{L}/collabrec-text-oracle_own_motive-k_2-beta_0-rs_1.failed_quota.log", "slurm-335338.out": f"{S}/CollabRec.335338.out"},
    f"{OR}/text-oracle_own_motive_k2_beta0.0_rs1_lr0.0003_job335416-preempted": {
        "train.log": f"{L}/collabrec-text-oracle_own_motive-k_2-beta_0-rs_1.preempted.log", "slurm-335416.out": f"{S}/CollabRec.335416.out"},
    f"{OR}/text-oracle_own_motive_k2_beta0.0_rs1_lr0.0003_job335637-cancelled": {
        "train.log": f"{L}/collabrec-text-oracle_own_motive-k_2-beta_0-rs_1.log", "slurm-335637.out": f"{S}/CollabRec.335637.out"},
    f"{IR}/IntuRecF1": {
        "train.log": f"{L}/inturec-f1-lr_3e-4-train-standard.log",
        "test_decode_lucas.log": f"{L}/inturec-f1-lr_3e-4-test-standard.log",
        **{f"genf1midres/{n}": f"{L}/{n}" for n in ["genf1midres-test-beam_15.log", "genf1midres-test-beam_16.log",
                                                       "genf1midres-train-beam_16.log", "genf1midres-valid-beam_15.log",
                                                       "genf1midres-valid-beam_16.log"]},
        "result/standard": f"{R}/{IR}/IntuRecF1/standard",
        "result/standard_orig_oct12": f"{R}/{IR}/IntuRecF1/standard_orig_oct12"},
    f"{IR}/IntuRecF2_original_oct15": {
        "train.log": f"{L}/inturec-f2-lr_3e-4-intunum_15-alpha_1.0-train-standard-rs_1.log"},
    f"{IR}/IntuRecF2_original_oct15_balance0295": {
        "train.log": f"{L}/inturec-f2-lr_3e-4-intunum_15-alpha_1.0-train-standard-rs_1_balance_ratio0295.log"},
    f"{IR}/IntuRecF2_micro8_job335355": {
        "train.log": f"{L}/inturec-f2-micro8_job335355-lr_3e-4-intunum_15-alpha_1.0-train-standard-rs_1.log",
        "slurm-335355.out": f"{S}/IntuRec.335355.out",
        "result/standard": f"{R}/{IR}/IntuRecF2_micro8_job335355/standard"},
    f"{IR}/IntuRecF2_micro8_job335679-cancelled": {
        "train.log": f"{L}/IntuRec_original/IntuRecF2_micro8_job335679.log", "slurm-335679.out": f"{S}/IntuRec.335679.out"},
    "collab_pipeline/logs": {p.name: str(p.relative_to(REPO)) for p in sorted((REPO / "log/collab").glob("*.log"))},
}

moved, missing = 0, []
for run, files in runs.items():
    for dest, src in files.items():
        src_path, dst_path = REPO / src, ROOT / run / dest
        if not src_path.exists():
            missing.append(src)
            continue
        if dst_path.exists():
            raise SystemExit(f"destination exists, refusing to overwrite: {dst_path}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        moved += 1
print(f"moved {moved} items; missing sources: {missing}")
left = [str(p.relative_to(REPO)) for d in ("log", "run_logs", "result_data") for p in (REPO / d).rglob("*") if p.is_file()]
print("files still in log/, run_logs/, result_data/:", left)
