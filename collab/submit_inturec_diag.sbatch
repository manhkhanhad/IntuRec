#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=20           # Number of CPU to request for the job
#SBATCH --mem=64GB                 # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=01-12:00:00         # How long to run the job for? Jobs exceed this time will be terminated
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out          # Where should the log files go?
#SBATCH --requeue                   # restart if preempted (finished decodes are reused)
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=IntuRecDiag     # Give the job a name

# Latent interventions on one IntuRec stage-2 checkpoint (collab/inturec_latent_diagnostics.py; IntuRec files
# untouched). Outputs: $CKPT_DIR/diagnostics/ (decodes reused after a requeue).
#   CKPT_DIR=<IntuRecF2 run folder> DATASET=Toys_and_Games
export DATASET=${DATASET:-Toys_and_Games}
[ -n "$CKPT_DIR" ] || { echo "set CKPT_DIR"; exit 1; }

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1
mkdir -p $CKPT_DIR/diagnostics
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CKPT_DIR=$CKPT_DIR DATASET=$DATASET"
python -u collab/inturec_latent_diagnostics.py --ckpt_dir $CKPT_DIR --dataset $DATASET --eval_batch_size 8 \
    >> $CKPT_DIR/diagnostics/diag.log 2>&1
status=$?
grep -v "it/s\]\|s/it\]" $CKPT_DIR/diagnostics/diag.log | tail -20
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $CKPT_DIR/diagnostics/slurm-$SLURM_JOB_ID.out
exit $status
