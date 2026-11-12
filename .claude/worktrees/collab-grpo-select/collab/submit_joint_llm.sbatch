#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue                   # joint: resumes from resume_state.pt; llm: restarts (~1-2 h)
#SBATCH --constraint="l40|l40s|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=MotiveSelect

# Literature-based motive selection on Toys (worktree collab-grpo-select):
#   MODE=joint  collab/joint_reader_selector.py  (Atlas / RA-DIT marginal-likelihood reader + selector)
#   MODE=llm    collab/llm_selector_grpo.py      (Rank-R1-style LLM setwise selector, GRPO, ranking reward)
#   MODE=joint NAME=joint_k4 sbatch --export=ALL collab/submit_joint_llm.sbatch
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-grpo-select}
export OUT_ROOT=${OUT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_GRPO_select}
export EXTRA=${EXTRA:-}
[ -n "$MODE" ] && [ -n "$NAME" ] || { echo "set MODE and NAME"; exit 1; }
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
out=$OUT_ROOT/${MODE}_$NAME
mkdir -p "$out"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  MODE=$MODE NAME=$NAME EXTRA=$EXTRA out=$out"
log=$out/train.log
if [ "$MODE" = joint ]; then
    python -u "$CODE_DIR/collab/joint_reader_selector.py" --out_dir "$out" $EXTRA >> "$log" 2>&1
else
    python -u "$CODE_DIR/collab/llm_selector_grpo.py" --out_dir "$out" $EXTRA >> "$log" 2>&1
fi
status=$?
tail -5 "$log"
exit $status
