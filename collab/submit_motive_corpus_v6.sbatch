#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a40|l40s"   # >= 48 GB: vLLM Qwen3-8B (0.45) + Qwen3-Embedding-8B share the GPU; fails on A100-40GB
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=MotiveV6

# Motive mining v6 (E8b), Toys: GEARec v2 reasoning -> {need -> item kind} motives -> Qwen3 embeddings ->
# Louvain (P=1000, 2000) -> naming. Writes data/Toys_and_Games/collab/motive_corpus_v6/. Every stage resumes.
# LIMIT=200 gives a smoke test of the extraction (then delete motive_corpus_v6/ before the full run).
export DATASET=${DATASET:-Toys_and_Games}
LIMIT=${LIMIT:-0}
RACER=/common/home/users/m/mk.ngohuu.2024/.conda/envs/racer/bin
RREC=/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec || exit 1
OUT=data/$DATASET/collab/motive_corpus_v6
mkdir -p $OUT
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

PORT=$((18000 + ${SLURM_JOB_ID:-0} % 1000))
SERVER=http://localhost:$PORT
(
    module purge
    eval "$(conda shell.bash hook)"
    module load CUDA/13.0.2
    conda activate LLM
    exec vllm serve Qwen/Qwen3-8B --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.45
) > $OUT/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf $SERVER/v1/models > /dev/null && break; sleep 10; done
curl -sf $SERVER/v1/models > /dev/null || { echo "vLLM did not start"; tail -50 $OUT/vllm_$SLURM_JOB_ID.log; exit 1; }

set -e
PY="collab/build_motive_corpus_v6.py --dataset $DATASET --server $SERVER"
for pass in 1 2; do $RACER/python -u $PY --stage extract --limit $LIMIT; done
[ "$LIMIT" != "0" ] && { echo "smoke test done"; exit 0; }
[ -f $OUT/emb.npy ] || $RACER/python -u $PY --stage embed
$RREC/python -u $PY --stage cluster
$RACER/python -u $PY --stage name
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $OUT/slurm-$SLURM_JOB_ID.out 2>/dev/null || true
