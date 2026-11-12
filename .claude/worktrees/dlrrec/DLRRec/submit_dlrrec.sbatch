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
#SBATCH --time=02-00:00:00         # How long to run the job for? Jobs exceed this time will be terminated
                                # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/dlrrec/DLRRec/logs/slurm/%x.%j.out          # Where should the log files go?
                                # You must provide an absolute path eg /common/home/module/username/
                                # If no paths are provided, the output file will be placed in your current working directory
# Remove if you do not want the workload scheduler to requeue your job after preemption --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=DLRRec     # Give the job a name

# DLRRec: discrete latent reasoning (motive codes) for recommendation.
#   sbatch --export=ALL,STAGE=align DLRRec/submit_dlrrec.sbatch                      stage 1: alignment
#   sbatch --export=ALL,STAGE=sft,REASONING=codes DLRRec/submit_dlrrec.sbatch        stage 2: SFT + test (generated & gold codes)
#   sbatch --export=ALL,STAGE=sft,REASONING=none DLRRec/submit_dlrrec.sbatch         control: same SFT without codes
#   sbatch --export=ALL,STAGE=test,REASONING=codes,EVAL_CKPT=epoch_2 DLRRec/submit_dlrrec.sbatch
# Chain stage 2 after stage 1 with --dependency=afterok:<align job id>.
export STAGE=${STAGE:-sft}
export REASONING=${REASONING:-codes}
export LAMBDA_CODE=${LAMBDA_CODE:-1.0}
export MICRO=${MICRO:-8}
export EVAL_CKPT=${EVAL_CKPT:-}

ROOT=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/dlrrec
export PATH=/common/home/users/m/mk.ngohuu.2024/.conda/envs/rrec/bin:$PATH
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
export PYTHONPATH=$ROOT TOKENIZERS_PARALLELISM=false
cd $ROOT || exit 1
mkdir -p DLRRec/logs/slurm

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "STAGE=$STAGE REASONING=$REASONING LAMBDA_CODE=$LAMBDA_CODE MICRO=$MICRO EVAL_CKPT=$EVAL_CKPT"
case $STAGE in
  align)
    log=DLRRec/logs/align.log
    python -u DLRRec/align.py > $log 2>&1 ;;
  sft)
    log=DLRRec/logs/sft-$REASONING-lam_$LAMBDA_CODE.log
    python -u DLRRec/train_sft.py --reasoning $REASONING --lambda_code $LAMBDA_CODE \
        --train_micro_batch_size $MICRO --eval_batch_size $MICRO > $log 2>&1 ;;
  test)
    log=DLRRec/logs/test-$REASONING-lam_$LAMBDA_CODE${EVAL_CKPT:+-$EVAL_CKPT}.log
    python -u DLRRec/train_sft.py --reasoning $REASONING --lambda_code $LAMBDA_CODE --train_or_test test \
        --eval_ckpt "$EVAL_CKPT" --eval_batch_size $MICRO > $log 2>&1 ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
status=$?
echo "log: $log"
grep -E "mean loss|codes vs gold|test metrics|Recall|NDCG" $log
exit $status
