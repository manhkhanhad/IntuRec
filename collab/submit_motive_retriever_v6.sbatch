#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --job-name=MotiveRetrV6

# v6 motive retriever (cls head, 5-fold cross-fitted train scores) + text supports for MotiveRec.
#   sbatch --export=ALL,DATASET=CDs_and_Vinyl collab/submit_motive_retriever_v6.sbatch
export DATASET=${DATASET:-CDs_and_Vinyl} CORPUS=${CORPUS:-motive_corpus_v6/P1000}
export SASREC_CKPT=${SASREC_CKPT:-}   # default: train_motive_retriever.py's (CDs) checkpoint; set for other datasets
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec || exit 1
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  $DATASET $CORPUS"
set -e
C=data/$DATASET/collab/$CORPUS
# a SASRec trained by collab/submit_sasrec.sbatch for this dataset wins over the default (CDs) checkpoint
[ -z "$SASREC_CKPT" ] && SASREC_CKPT=$(ls -t data/$DATASET/collab/sasrec/*/models/SASRec/*.pkl 2>/dev/null | head -1)
echo "SASREC_CKPT=${SASREC_CKPT:-default}"
python -u collab/build_hist_text_emb.py --dataset $DATASET   # history-text feature (no-op if cached)
[ -f $C/retriever/cls/scores_train.npy ] || python -u collab/train_motive_retriever.py \
    --dataset $DATASET --corpus $CORPUS --head cls --crossfit-folds 5 ${SASREC_CKPT:+--sasrec-ckpt $SASREC_CKPT}
python -u collab/build_motive_text_support.py --dataset $DATASET --corpus $CORPUS --head cls --k 10
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $C/retriever-slurm-$SLURM_JOB_ID.out || true
