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
#SBATCH --job-name=SASRec

# SASRec for the motive retriever's "sasrec" feature (same hyper-parameters as Verifiable-Rec run_sasrec_toys.sh).
# Runs Verifiable-Rec's sasrec.py read-only; the model is written inside this worktree:
#   data/<DATASET>/collab/sasrec/<csv basename>/models/SASRec/<timestamp>.pkl
#   sbatch --export=ALL,DATASET=Steam collab/submit_sasrec.sbatch
export DATASET=${DATASET:?set DATASET}
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec || exit 1
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  $DATASET"
set -e
python -u /common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/baselines/sasrec.py \
    --data-dir $(readlink -f data/$DATASET) --output-dir data/$DATASET/collab/sasrec \
    --epochs 500 --embedding-dim 128 --batch-size 512 --learning-rate 0.001 --negative-samples 2048 --max-len 10 \
    --num-blocks 2 --num-heads 1 --dropout 0.2 --device cuda --cutoffs 1,3,5,10,20 --top-k 20 \
    --model-selection best --val-eval-every 20
ls data/$DATASET/collab/sasrec/*/models/SASRec/*.pkl
