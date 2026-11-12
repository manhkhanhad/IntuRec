#!/usr/bin/env bash
# Latent collaborative reasoning (CollabRecF2), independent of IntuRec's intuition.
# Ablations: SUPPORT_K=0 (no supports), BETA=0 (no rationale alignment).
dataset="CDs_and_Vinyl"
support_k=${SUPPORT_K:-10}
beta=${BETA:-0.1}
reason_step=${REASON_STEP:-1}
micro=${MICRO:-8}
mkdir -p log/$dataset
python -u run_collabrec.py \
    --device cuda:0 \
    --dataset $dataset \
    --base_model Qwen/Qwen2.5-1.5B \
    --support_dir data/$dataset/collab/stage2 \
    --support_k $support_k \
    --beta $beta \
    --reason_step $reason_step \
    --lr 3e-4 \
    --train_batch_size 128 \
    --train_micro_batch_size $micro \
    --eval_batch_size $micro \
    --output_dir checkpoints > log/$dataset/collabrec-k_$support_k-beta_$beta-rs_$reason_step.log 2>&1
