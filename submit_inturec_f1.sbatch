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
#SBATCH --job-name=IntuRecF1     # Give the job a name

# IntuRec stage 1 + intuition lists + stage-2 data for a dataset, in one job:
#   1) run_inturec_f1.py (train; HF Trainer keeps the best model by valid loss)
#   2) genf1midres.py beams: train 16, valid 15, test 15 (-> data/$DATASET/<split>_beam_N.pt)
#   3) prepare_stage2_data.py -> data/$DATASET/{train,valid,test}_extra/extra.json
# Then submit stage 2 with: sbatch --dependency=afterok:<this job> --export=ALL,DATASET=$DATASET submit_inturec_f2.sbatch
export DATASET=${DATASET:-Toys_and_Games}
export MICRO=${MICRO:-8}
export CKPT_ROOT=${CKPT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints}
export EXPERIMENT=${EXPERIMENT:-IntuRec_original}
export RUN_NAME=${RUN_NAME:-IntuRecF1_micro${MICRO}_job${SLURM_JOB_ID:-local}}
# Nodes have no system CUDA: load the toolkit matching rrec's torch (cu124) so CUDA_HOME is set
# (DeepSpeed, imported by transformers' Trainer / save_pretrained, requires it).
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1
SRC=/common/home/users/m/mk.ngohuu.2024/research/R3/data
RUN_DIR=$CKPT_ROOT/$DATASET/$EXPERIMENT/$RUN_NAME
mkdir -p $RUN_DIR data/$DATASET
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "run folder: $RUN_DIR"
set -e
# Steps whose output already exists are skipped, so a requeue (preemption) resumes instead of retraining.
if [ ! -f $RUN_DIR/model.safetensors ]; then
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$((RANDOM % 5000 + 10000)) run_inturec_f1.py \
    --data_path $SRC --dataset $DATASET --lr 3e-4 --base_model Qwen/Qwen2.5-1.5B \
    --train_batch_size 128 --train_micro_batch_size $MICRO --eval_batch_size $MICRO \
    --train_or_test train --decode_method standard \
    --ckpt_dir $RUN_DIR --result_dir $RUN_DIR/result/standard > $RUN_DIR/train.log 2>&1
fi
grep -E "Recall|NDCG" $RUN_DIR/train.log || true
for split in train valid test; do
    beam=15; [ $split = train ] && beam=16
    [ -f data/$DATASET/${split}_beam_$beam.pt ] && { echo "skip $split beams (exists)"; continue; }
    python -u genf1midres.py --device cuda:0 --dataset $DATASET --dataset_type $split --beam_num $beam \
        --f1_ckpt_path $RUN_DIR --batch_size 8 > $RUN_DIR/genf1midres-$split-beam_$beam.log 2>&1
done
python -u prepare_stage2_data.py --dataset $DATASET --source-data-root $SRC --output-data-root data \
    --beam-num 15 > $RUN_DIR/prepare_stage2_data.log 2>&1
cat $RUN_DIR/prepare_stage2_data.log
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $RUN_DIR/slurm-$SLURM_JOB_ID.out 2>/dev/null || true
