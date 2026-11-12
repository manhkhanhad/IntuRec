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
#SBATCH --time=01-12:00:00         # How long to run the job for? Jobs exceed this time will be terminated
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out          # Where should the log files go?
#SBATCH --requeue                   # restart if preempted (every step resumes / skips finished outputs)
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=MotiveCorpus     # Give the job a name

# Motive pipeline for one dataset (needs collab/reviews_<split>.json from extract_reviews.py):
#   1. Qwen3-8B (vLLM, this GPU) facet rationales for train/valid/test, tag v5;
#   2. Qwen3-Embedding-8B vectors of all motive phrases (racer env);
#   3. kNN-graph Louvain clustering of the train phrases (rrec env, faiss);
#   4. merge / fold / LLM naming / complaint removal -> collab/motive_corpus/ (racer env).
export DATASET=${DATASET:-Toys_and_Games}
export DOMAIN=${DOMAIN:-toys}
export TAG=${TAG:-v5}
export WORKERS=${WORKERS:-64}
RACER=/common/home/users/m/mk.ngohuu.2024/.conda/envs/racer/bin
RREC=/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1
COLLAB=data/$DATASET/collab
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "DATASET=$DATASET DOMAIN=$DOMAIN TAG=$TAG"

PORT=$((18000 + ${SLURM_JOB_ID:-0} % 1000))
SERVER=http://localhost:$PORT
# vLLM runs in the LLM env with the CUDA module, in a subshell so the racer / rrec steps keep their own setup.
(
    module purge
    eval "$(conda shell.bash hook)"
    module load CUDA/13.0.2
    conda activate LLM
    exec vllm serve Qwen/Qwen3-8B --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.45
) > $COLLAB/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for _ in $(seq 120); do
    curl -sf $SERVER/v1/models > /dev/null && break
    sleep 10
done
curl -sf $SERVER/v1/models > /dev/null || { echo "vLLM did not start"; tail -50 $COLLAB/vllm_$SLURM_JOB_ID.log; exit 1; }
echo "vLLM up at $SERVER"

set -e
# 1. Rationales (resumable; a second pass retries rows that errored).
for split in train valid test; do
    for pass in 1 2; do
        $RACER/python -u collab/gen_rationales.py --dataset $DATASET --domain $DOMAIN --split $split --mode facet \
            --tag $TAG --server $SERVER --workers $WORKERS
    done
done

# 2. Phrase embeddings (cached in hyp_cache/cluster_phrase_qwen3.npy).
mkdir -p $COLLAB/hyp_cache
$RACER/python -u -c "
import sys, argparse; from pathlib import Path
sys.path.insert(0, 'collab')
import compare_clustering as cc
args = argparse.Namespace(dataset='$DATASET', data_root=Path('data'), rationale_tag='$TAG')
print(len(cc.load_space(args, 'phrase/qwen3')['units']), 'train phrase units')
"

# 3. Louvain clustering (~1000 communities).
$RREC/python -u collab/compare_clustering.py --dataset $DATASET --rationale-tag $TAG \
    --spaces phrase/qwen3 --methods louvain

# 4. Motive corpus.
$RACER/python -u collab/build_motive_corpus.py --dataset $DATASET --domain $DOMAIN --rationale-tag $TAG \
    --server $SERVER
cp -f run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $COLLAB/motive_corpus/slurm-$SLURM_JOB_ID.out
