#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=MotiveUtility

# Utility judge (collab/motive_utility.py): u(m) = log p(target | history, m) - log p(target | history, null)
# under a trained O2 model. Usage:
#   DATASET=Toys_and_Games CKPT=<O2 run folder> sbatch --export=ALL collab/submit_motive_utility.sbatch
export DATASET=${DATASET:-Toys_and_Games}
export SPLIT=${SPLIT:-test}
export FOLD=${FOLD:-}   # "i/n": score only users in fold i (CKPT must be the judge trained WITHOUT fold i)
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-refined-infonce}
: "${CKPT:?set CKPT to the O2 run folder}"

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd "$CODE_DIR" || exit 1
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "DATASET=$DATASET SPLIT=$SPLIT CKPT=$CKPT"
python -u collab/motive_utility.py --ckpt "$CKPT" --dataset $DATASET --split $SPLIT --batch 32 ${FOLD:+--fold $FOLD}
status=$?
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out "$CKPT/motive_utility/slurm-$SLURM_JOB_ID.out" 2>/dev/null
exit $status
