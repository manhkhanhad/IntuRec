#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=20           # Number of CPU to request for the job
#SBATCH --mem=96GB                 # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=01-00:00:00         # How long to run the job for? Jobs exceed this time will be terminated
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out          # Where should the log files go?
#SBATCH --requeue                   # restart if preempted (finished steps are skipped)
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=CollabSupports     # Give the job a name

# Inputs for CollabRec and the predicted-motive run on one dataset (needs the v5 rationales and
# motive_corpus/ from submit_motive_corpus.sbatch, and a SASRec model):
#   1. tune_latent.py          history text caches (mpnet, gte-large) + predictor / alpha -> collab/tune_latent.json
#   2. build_stage2_support.py peer-rationale supports -> collab/stage2/
#   3. train_motive_retriever.py (cls head, 5-fold cross-fitted train scores) -> motive_corpus/retriever/cls/
#   4. top-k predicted motive prototypes -> motive_corpus/stage2/ (support_texts.json, support_rows_<split>.npy)
export DATASET=${DATASET:-Toys_and_Games}
export SASREC_CKPT=${SASREC_CKPT:-/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/results/sasrec/Toys_and_Games_5_2017-1-2018-11/models/SASRec/2026-10-31_12-00-01-480407.pkl}
export TOPK=${TOPK:-10}

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1
COLLAB=data/$DATASET/collab
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "DATASET=$DATASET SASREC_CKPT=$SASREC_CKPT"
set -e

[ -f $COLLAB/tune_latent.json ] || python -u collab/tune_latent.py --dataset $DATASET --sasrec-ckpt $SASREC_CKPT
[ -f $COLLAB/stage2/support_rows_test.npy ] || python -u collab/build_stage2_support.py --dataset $DATASET --sasrec-ckpt $SASREC_CKPT
[ -f $COLLAB/motive_corpus/retriever/cls/scores_train.npy ] || python -u collab/train_motive_retriever.py \
    --dataset $DATASET --sasrec-ckpt $SASREC_CKPT --head cls --crossfit-folds 5

python -u - <<EOF
import json
from pathlib import Path
import numpy as np
corpus = Path("$COLLAB/motive_corpus")
out = corpus / "stage2"
out.mkdir(exist_ok=True)
protos = json.loads((corpus / "prototypes.json").read_text())
(out / "support_texts.json").write_text(json.dumps([f"{p['motive']}: {p['reason']}" for p in protos]))
for split in ("train", "valid", "test"):
    scores = np.load(corpus / "retriever" / "cls" / f"scores_{split}.npy").astype(np.float32)
    rows = np.argsort(-scores, axis=1)[:, :$TOPK]
    truth = json.loads((corpus / f"row_motives_{split}.json").read_text())
    hit = np.mean([bool(set(t) & set(r.tolist())) for t, r in zip(truth, rows) if t])
    np.save(out / f"support_rows_{split}.npy", rows)
    print(f"{split}: top-$TOPK predicted motives, rows with a true motive in top-$TOPK {hit:.3f}")
EOF
cp -f run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $COLLAB/supports-slurm-$SLURM_JOB_ID.out
