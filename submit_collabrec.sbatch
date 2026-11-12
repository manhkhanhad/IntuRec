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
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out          # Where should the log files go?
                                # You must provide an absolute path eg /common/home/module/username/
                                # If no paths are provided, the output file will be placed in your current working directory
# Remove if you do not want the workload scheduler to requeue your job after preemption --requeue
#SBATCH --requeue                   # restart from scratch if the job is preempted
#SBATCH --constraint="l40|a100|a40|l40s"
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --qos=research-1-qos      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=CollabRec     # Give the job a name

# CollabRec stage 2 (run_collabrec.py). Override per job, e.g. for ablations:
#   sbatch --export=ALL,SUPPORT_K=0,BETA=0 submit_collabrec.sbatch
#   sbatch --export=ALL,DATASET=Toys_and_Games,SUPPORT_K=0,BETA=0,SUPPORT_DIR= submit_collabrec.sbatch   (other dataset)
#   sbatch --export=ALL,SUPPORT_MODE=emb submit_collabrec.sbatch   (first version: projected embeddings)
#   sbatch --export=ALL,SUPPORT_DIR=data/CDs_and_Vinyl/collab/motive_corpus/stage2,RUN_TAG=motive submit_collabrec.sbatch
#   sbatch --export=ALL,TRAIN_OR_TEST=test,EVAL_CKPT=epoch_2,RUN_SUFFIX=job335212 submit_collabrec.sbatch   (decode a saved epoch of that run)
#   sbatch --export=ALL,ORACLE=own,SUPPORT_K=1,BETA=0 submit_collabrec.sbatch   (oracle analysis)
#   sbatch --export=ALL,SUPPORT_DIR=data/CDs_and_Vinyl/collab/stage2_item,RUN_TAG=item,GAMMA=1.0,SUP_ANCHOR=title submit_collabrec.sbatch
# All constrained GPUs have >= 40GB, so micro-batch 8 fits (effective batch stays 128).
export DATASET=${DATASET:-CDs_and_Vinyl}
export SUPPORT_MODE=${SUPPORT_MODE:-text}
export SUPPORT_K=${SUPPORT_K:-10}
export BETA=${BETA:-0.1}
export REASON_STEP=${REASON_STEP:-1}
export MICRO=${MICRO:-8}
export SUPPORT_DIR=${SUPPORT_DIR-data/$DATASET/collab/stage2}   # may be empty (e.g. O0 without supports)
export RUN_TAG=${RUN_TAG:-}
export TRAIN_OR_TEST=${TRAIN_OR_TEST:-train}
export EVAL_CKPT=${EVAL_CKPT:-}   # test only: e.g. epoch_2
export ORACLE=${ORACLE:-}         # analysis only: own | own_motive (use with BETA=0)
export GAMMA=${GAMMA:-0}          # contrastive support loss weight (0 = off)
export SUP_ANCHOR=${SUP_ANCHOR:-title}   # title | motive
export COLLAB_WEIGHT=${COLLAB_WEIGHT:-0}  # collaborative InfoNCE on the refined thought, per relation (0 = off)
export COLLAB_RELATIONS=${COLLAB_RELATIONS:-user,target,motive}
export COLLAB_MIN_ACC=${COLLAB_MIN_ACC:-0.5}   # queue gate: share of target tokens predicted correctly
# Motive utilization: readout inject | gated | soft; motive supports "" | predicted | noisy (noisy oracle)
export READOUT=${READOUT:-inject}
export SEL_WEIGHT=${SEL_WEIGHT:-1.0}
export MOTIVE_SOURCE=${MOTIVE_SOURCE:-}
export NOISY_P=${NOISY_P:-1.0}
export NOISY_P_START=${NOISY_P_START:--1}   # >= 0: curriculum, train precision NOISY_P_START -> NOISY_P per epoch
export UTILITY_FILES=${UTILITY_FILES:-}   # comma-separated utility_train_fold*.npz: gate labels = u > UTILITY_MARGIN
export UTILITY_MARGIN=${UTILITY_MARGIN:-0}
export RESUME_EVERY=${RESUME_EVERY:-0}   # >0: save a ~9 GB resume state every N steps (off by default: storage)
export TRAIN_FOLD=${TRAIN_FOLD:-}   # "i/n": train on users outside fold i (cross-fitted utility judge)
export EXTRA_PRED=${EXTRA_PRED:-0}   # 1: also decode test with the retriever's top-k motives (result/.../predicted)
# Code checkout to run (this worktree by default); data/ and run_logs/ are shared with the main checkout.
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-refined-infonce}
# Checkpoints: $CKPT_ROOT/<dataset>/$EXPERIMENT/<run_name>/ ; results: result_data/<dataset>/$EXPERIMENT/<run_name>/
export CKPT_ROOT=${CKPT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints}
export EXPERIMENT=${EXPERIMENT:-$( [ -n "$ORACLE" ] || [ "$MOTIVE_SOURCE" = noisy ] && echo CollabRec_oracle || echo CollabRec )}
export RUN_SUFFIX=${RUN_SUFFIX:-job${SLURM_JOB_ID:-local}}

# Nodes have no system CUDA: load the toolkit matching rrec's torch (cu124) so CUDA_HOME is set
# (DeepSpeed, imported by transformers' Trainer / save_pretrained, requires it).
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd "$CODE_DIR" || exit 1

dataset="$DATASET"
run=$SUPPORT_MODE${RUN_TAG:+-$RUN_TAG}${ORACLE:+-oracle_$ORACLE}$( [ "$GAMMA" != 0 ] && echo "-gamma${GAMMA}_$SUP_ANCHOR" )$( [ -n "$MOTIVE_SOURCE" ] && echo "-$MOTIVE_SOURCE$( [ "$MOTIVE_SOURCE" = noisy ] && { [ "${NOISY_P_START:0:1}" != - ] && echo ${NOISY_P_START}to$NOISY_P || echo $NOISY_P; } )" )$( [ "$READOUT" != inject ] && echo "-$READOUT$( [ "$READOUT" = gated ] && echo $SEL_WEIGHT )${UTILITY_FILES:+-util$UTILITY_MARGIN}" )$( [ "$COLLAB_WEIGHT" != 0 ] && echo "-cq${COLLAB_WEIGHT}_${COLLAB_RELATIONS//,/+}" )${TRAIN_FOLD:+-fold${TRAIN_FOLD/\//of}}_k${SUPPORT_K}_beta${BETA}_rs${REASON_STEP}_lr0.0003_$RUN_SUFFIX
# One folder per run: epoch_N/ checkpoints, result/ predictions, train.log, slurm-<job>.out.
RUN_DIR=$CKPT_ROOT/$dataset/$EXPERIMENT/$run
log=$RUN_DIR/train.log
[ "$TRAIN_OR_TEST" = test ] && log=$RUN_DIR/test${EVAL_CKPT:+_$EVAL_CKPT}_job${SLURM_JOB_ID:-local}.log
mkdir -p $RUN_DIR
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "SUPPORT_MODE=$SUPPORT_MODE SUPPORT_K=$SUPPORT_K BETA=$BETA REASON_STEP=$REASON_STEP MICRO=$MICRO SUPPORT_DIR=$SUPPORT_DIR RUN_TAG=$RUN_TAG COLLAB_WEIGHT=$COLLAB_WEIGHT COLLAB_RELATIONS=$COLLAB_RELATIONS COLLAB_MIN_ACC=$COLLAB_MIN_ACC READOUT=$READOUT SEL_WEIGHT=$SEL_WEIGHT MOTIVE_SOURCE=$MOTIVE_SOURCE NOISY_P=$NOISY_P EXTRA_PRED=$EXTRA_PRED"
echo "code: $CODE_DIR"
echo "run folder: $RUN_DIR"
echo "training log: $log"
python -u run_collabrec.py \
    --device cuda:0 \
    --dataset $dataset \
    --base_model Qwen/Qwen2.5-1.5B \
    --support_dir "$SUPPORT_DIR" \
    --run_tag "$RUN_TAG" \
    --train_or_test $TRAIN_OR_TEST \
    --eval_ckpt "$EVAL_CKPT" \
    --oracle "$ORACLE" \
    --gamma $GAMMA \
    --sup_anchor $SUP_ANCHOR \
    --collab_weight $COLLAB_WEIGHT \
    --collab_relations $COLLAB_RELATIONS \
    --collab_min_token_accuracy $COLLAB_MIN_ACC \
    --support_mode $SUPPORT_MODE \
    --readout $READOUT \
    --sel_weight $SEL_WEIGHT \
    --motive_source "$MOTIVE_SOURCE" \
    --noisy_precision $NOISY_P \
    --noisy_precision_start $NOISY_P_START \
    --train_fold "$TRAIN_FOLD" \
    --resume_every $RESUME_EVERY \
    --utility_files "$UTILITY_FILES" --utility_margin $UTILITY_MARGIN \
    $( [ "$EXTRA_PRED" = 1 ] && echo --extra_test_predicted ) \
    --support_k $SUPPORT_K \
    --beta $BETA \
    --reason_step $REASON_STEP \
    --lr 3e-4 \
    --train_batch_size 128 \
    --train_micro_batch_size $MICRO \
    --eval_batch_size $MICRO \
    --ckpt_root $CKPT_ROOT \
    --experiment $EXPERIMENT \
    --run_name "$run" >> $log 2>&1   # append: a requeued (resumed) run keeps its earlier log
status=$?
grep -E "Recall|NDCG" $log
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $RUN_DIR/slurm-$SLURM_JOB_ID.out 2>/dev/null
exit $status
