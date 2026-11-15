#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=20           # Number of CPU to request for the job
#SBATCH --mem=64GB                 # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00         # How long to run the job for? Jobs exceed this time will be terminated
                                # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out          # Where should the log files go?
                                # You must provide an absolute path eg /common/home/module/username/
                                # If no paths are provided, the output file will be placed in your current working directory
# Remove if you do not want the workload scheduler to requeue your job after preemption --requeue
#SBATCH --requeue                   # restart from scratch if the job is preempted
#SBATCH --constraint="l40|a40|l40s"      # no 40 GB A100s (Toys needs >= 48 GB)
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you?
#SBATCH --job-name=LatentMS2    # Give the job a name
# LatentMS2 = LatentMS with the normalised RATT (model/RATTNorm.py; RATT_RESIDUAL 0/1) via run_latent_ms2.py: REASON_STEP
# thoughts, thought 1 = FIRST_THOUGHT (last = hidden state before slot 1), thoughts 2..S = RATT; loss =
# NTP + PREFIX_WEIGHT * prefix NTP losses + MONO_WEIGHT * relu(L_t - sg(L_{t-1}) + MONO_MARGIN).
# Same batch settings as the IntuRec baselines (micro 2, batch 128, lr 3e-4, 3 epochs).
#   sbatch --export=ALL,DATASET=Toys_and_Games,REASON_STEP=3 submit_latent_ms.sbatch
export DATASET=${DATASET:-CDs_and_Vinyl}
export MICRO=${MICRO:-2}
export REASON_STEP=${REASON_STEP:-2}
export FIRST_THOUGHT=last
export RATT_RESIDUAL=${RATT_RESIDUAL:-1}
export PREFIX_WEIGHT=${PREFIX_WEIGHT:-1.0}
export MONO_WEIGHT=${MONO_WEIGHT:-1.0}
export MONO_MARGIN=${MONO_MARGIN:-0.02}
export RUN_ID=${RUN_ID:-micro${MICRO}_job${SLURM_JOB_ID:-local}}
export CKPT_ROOT=${CKPT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints}
export EXPERIMENT=${EXPERIMENT:-Latent_multithink}
export RUN_NAME=${RUN_NAME:-LatentMS2_res${RATT_RESIDUAL}_rs${REASON_STEP}_pw${PREFIX_WEIGHT}_mw${MONO_WEIGHT}_m${MONO_MARGIN}_$RUN_ID}

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/multithink || exit 1

dataset="$DATASET"
RUN_DIR=$CKPT_ROOT/$dataset/$EXPERIMENT/$RUN_NAME
log=$RUN_DIR/train.log
mkdir -p $RUN_DIR
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "RATT_RESIDUAL=$RATT_RESIDUAL"
echo "MICRO=$MICRO REASON_STEP=$REASON_STEP FIRST_THOUGHT=$FIRST_THOUGHT PREFIX_WEIGHT=$PREFIX_WEIGHT MONO_WEIGHT=$MONO_WEIGHT MONO_MARGIN=$MONO_MARGIN RUN_ID=$RUN_ID"
echo "code: $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted"
echo "run folder: $RUN_DIR"
echo "training log: $log"

python -u run_latent_ms2.py \
    --ratt_residual $RATT_RESIDUAL \
    --first_thought $FIRST_THOUGHT \
    --prefix_weight $PREFIX_WEIGHT \
    --mono_weight $MONO_WEIGHT \
    --mono_margin $MONO_MARGIN \
    --device cuda:0 \
    --dataset $dataset \
    --base_model Qwen/Qwen2.5-1.5B \
    --ckpt_dir $RUN_DIR \
    --result_dir $RUN_DIR/result/standard \
    --train_or_test train \
    --decode_method standard \
    --lr 3e-4 \
    --intuition_num 15 \
    --reason_step $REASON_STEP \
    --train_batch_size 128 \
    --train_micro_batch_size $MICRO \
    --eval_batch_size $MICRO > $log 2>&1
status=$?
grep -E "Recall|NDCG" $log
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $RUN_DIR/slurm-$SLURM_JOB_ID.out 2>/dev/null
exit $status
