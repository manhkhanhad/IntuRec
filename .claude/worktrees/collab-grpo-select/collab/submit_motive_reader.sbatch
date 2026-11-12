#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|l40s|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=MotiveReader

# Decode the Toys test split with a frozen motive reader (default: O2 + InfoNCE, run 351001) and one
# chosen motive per row, as [m, m] (collab/motive_reader.py).
#   PICKS=collab/select_picks/toys_test_grpo_b0_s0.npy NAME=grpo_b0_s0 sbatch --export=ALL collab/submit_motive_reader.sbatch
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-grpo-select}
export OUT_ROOT=${OUT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_GRPO_select}
[ -n "$PICKS" ] && [ -n "$NAME" ] || { echo "set PICKS and NAME"; exit 1; }
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
out=$OUT_ROOT/decode_$NAME
mkdir -p "$out"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  PICKS=$PICKS NAME=$NAME  out=$out"
log=$out/decode_job${SLURM_JOB_ID:-local}.log
case "$PICKS" in /*) picks=$PICKS ;; *) picks=$CODE_DIR/$PICKS ;; esac   # absolute or worktree-relative
python -u "$CODE_DIR/collab/motive_reader.py" decode --picks "$picks" --out_dir "$out" --eval_batch_size 8 \
    ${READER:+--reader "$READER"} > "$log" 2>&1
status=$?
grep -E "Recall|NDCG" "$log"
exit $status
