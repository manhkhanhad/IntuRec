#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=IRLLRecMotive

# E8h: IRLLRec-faithful (lightgcn_int) motive model on v6 P1000 motives, Toys. Rows = user nodes.
# Run 1: full IRLLRec losses. Run 2: plain LightGCN ablation (all text-alignment weights 0).
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec/collab || exit 1
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[ -f ../data/Toys_and_Games/collab/motive_corpus_v6/P1000/retriever/irllrec_irllrec/results.json ] || python -u irllrec_motive.py --stage train --data-root ../data --tag irllrec
[ -f ../data/Toys_and_Games/collab/motive_corpus_v6/P1000/retriever/irllrec_plain/results.json ] || python -u irllrec_motive.py --stage train --data-root ../data --tag plain \
    --kd-weight 0 --kd-int-weight 0 --kd-int-weight-2 0 --kd-int-weight-3 0
