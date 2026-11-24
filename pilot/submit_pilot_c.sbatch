#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=PilotC_R3

# Pilot C: R3 interventions on an original LatentR3 checkpoint (finished steps are skipped on requeue).
#   sbatch --export=ALL,DATASET=Toys_and_Games,MODEL=LatentR3-RL pilot/submit_pilot_c.sbatch
export DATASET=${DATASET:-CDs_and_Vinyl}
export MODEL=${MODEL:-LatentR3-RL}
export LIMIT=${LIMIT:--1}

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec || exit 1

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "DATASET=$DATASET MODEL=$MODEL LIMIT=$LIMIT"
python -u pilot/pilot_c_r3.py --dataset $DATASET --model $MODEL --limit $LIMIT
