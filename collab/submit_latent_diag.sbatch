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
#SBATCH --requeue                   # restart if preempted (finished decodes are reused)
#SBATCH --constraint="l40|l40s|a40"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=LatentDiag     # Give the job a name

# Collaborative structure + interventions of the motive-refined thought for one CollabRec checkpoint
# (collab/latent_diagnostics.py). Outputs go to $RUN_DIR/diagnostics/ (decodes reused after a requeue).
#   RUN_DIR=<checkpoint folder> DATASET=Toys_and_Games SUPPORT_DIR=data/<dataset>/collab/motive_corpus/stage2
export DATASET=${DATASET:-Toys_and_Games}
export SUPPORT_DIR=${SUPPORT_DIR:-data/$DATASET/collab/motive_corpus/stage2}
export CONDITIONS=${CONDITIONS:-raw,mask,mean,shuffle,swap_motives,random_motives,oracle_motives}
[ -n "$RUN_DIR" ] || { echo "set RUN_DIR"; exit 1; }

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-refined-infonce || exit 1
mkdir -p $RUN_DIR/diagnostics
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "RUN_DIR=$RUN_DIR DATASET=$DATASET SUPPORT_DIR=$SUPPORT_DIR CONDITIONS=$CONDITIONS"
python -u collab/latent_diagnostics.py --run_dir $RUN_DIR --dataset $DATASET --support_dir $SUPPORT_DIR \
    --conditions $CONDITIONS --eval_batch_size 8 $EXTRA_ARGS --out_dir $RUN_DIR/${OUT_NAME:-diagnostics} >> $RUN_DIR/diagnostics/diag${OUT_NAME:+_$OUT_NAME}.log 2>&1
status=$?
grep -v "it/s\]\|s/it\]" $RUN_DIR/diagnostics/diag${OUT_NAME:+_$OUT_NAME}.log | tail -30
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $RUN_DIR/diagnostics/slurm-$SLURM_JOB_ID.out
exit $status
