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
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=IntuRec     # Give the job a name

# IntuRec stage 2 with the same batch settings as submit_collabrec.sbatch (micro-batch 8,
# effective batch 128, eval batch 8) and checkpoints kept under checkpoints/.
export DATASET=${DATASET:-CDs_and_Vinyl}
export MICRO=${MICRO:-8}
export ALPHA=${ALPHA:-1.0}
export INTUITION_NUM=${INTUITION_NUM:-15}
export REASON_STEP=${REASON_STEP:-1}
# Layout: $CKPT_ROOT/<dataset>/$EXPERIMENT/$RUN_NAME/ (checkpoints, scratch) and
# result_data/<dataset>/$EXPERIMENT/$RUN_NAME/ (predictions); every run gets its own folder.
export RUN_ID=${RUN_ID:-micro${MICRO}_job${SLURM_JOB_ID:-local}}
export CKPT_ROOT=${CKPT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints}
export EXPERIMENT=${EXPERIMENT:-IntuRec_original}
export RUN_NAME=${RUN_NAME:-IntuRecF2_$RUN_ID}

# Nodes have no system CUDA: load the toolkit matching rrec's torch (cu124) so CUDA_HOME is set
# (DeepSpeed, imported by transformers' Trainer / save_pretrained, requires it).
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1

dataset="$DATASET"
# One folder per run: checkpoints (best + epoch_N/), result/ predictions, train.log, slurm-<job>.out.
RUN_DIR=$CKPT_ROOT/$dataset/$EXPERIMENT/$RUN_NAME
log=$RUN_DIR/train.log
mkdir -p $RUN_DIR
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "MICRO=$MICRO ALPHA=$ALPHA INTUITION_NUM=$INTUITION_NUM REASON_STEP=$REASON_STEP RUN_ID=$RUN_ID"
echo "run folder: $RUN_DIR"
echo "training log: $log"
python -u run_inturec_f2.py \
    --device cuda:0 \
    --dataset $dataset \
    --base_model Qwen/Qwen2.5-1.5B \
    --ckpt_dir $RUN_DIR \
    --result_dir $RUN_DIR/result/standard \
    --train_or_test train \
    --decode_method standard \
    --lr 3e-4 \
    --intuition_num $INTUITION_NUM \
    --reason_step $REASON_STEP \
    --alpha $ALPHA \
    --train_batch_size 128 \
    --train_micro_batch_size $MICRO \
    --eval_batch_size $MICRO > $log 2>&1
status=$?
grep -E "Recall|NDCG" $log
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $RUN_DIR/slurm-$SLURM_JOB_ID.out 2>/dev/null
exit $status
