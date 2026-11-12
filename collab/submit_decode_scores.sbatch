#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue                   # finished decodes are skipped on requeue (meta.json exists)
#SBATCH --constraint="l40|l40s|a40" # Toys needs 48 GB
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=CollabRecRetest

# Re-decode a saved CollabRec checkpoint on the test split with each retriever's top-k motives
# (collab/decode_with_scores.py). Outputs: $RUN_DIR/result/standard/retest_<scores>_top<k>/.
#   RUN_DIR=<run folder> sbatch --export=ALL collab/submit_decode_scores.sbatch
export SCORES=${SCORES:-cls mlp_cand}   # space-separated score sets under motive_corpus/retriever/
export SUPPORT_K=${SUPPORT_K:-10}
export DATASET=${DATASET:-Toys_and_Games}
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-refined-infonce}
[ -d "$RUN_DIR" ] || { echo "RUN_DIR not found: $RUN_DIR"; exit 1; }

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd "$CODE_DIR" || exit 1

log=$RUN_DIR/retest_job${SLURM_JOB_ID:-local}.log
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "RUN_DIR=$RUN_DIR SCORES=$SCORES SUPPORT_K=$SUPPORT_K DATASET=$DATASET  log: $log"
status=0
for s in $SCORES; do
    echo "=== scores $s, top-$SUPPORT_K" >> "$log"
    python -u collab/decode_with_scores.py --run_dir "$RUN_DIR" --scores "$s" --support_k $SUPPORT_K \
        --dataset "$DATASET" --eval_batch_size 8 >> "$log" 2>&1 || status=$?
done
grep -E "===|Recall|NDCG" "$log"
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out "$RUN_DIR/slurm-retest-$SLURM_JOB_ID.out" 2>/dev/null
exit $status
