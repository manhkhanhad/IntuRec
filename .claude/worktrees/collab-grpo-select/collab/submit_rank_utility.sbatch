#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue                   # resumes from rank_utility_<tag>.partial.npz
#SBATCH --constraint="l40|l40s|a40"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=RankUtility

# Ranking-aware motive utilities (collab/rank_utility.py): target + 20 SASRec hard negatives under
# null + the retriever's top-10 motives, scored by the cross-fitted judge of each fold.
#   TAG=train0 | train1 | test   sbatch --export=ALL collab/submit_rank_utility.sbatch
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-grpo-select}
[ -n "$TAG" ] || { echo "set TAG"; exit 1; }
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
out=/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/Toys_and_Games/CollabRec_GRPO_select/rank_reward
mkdir -p "$out"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  TAG=$TAG"
python -u "$CODE_DIR/collab/rank_utility.py" --tag "$TAG" >> "$out/rank_utility_${TAG}.log" 2>&1
status=$?
tail -3 "$out/rank_utility_${TAG}.log"
exit $status
