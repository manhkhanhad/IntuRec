#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue                   # preempted runs resume from <run>/resume/state.pt (RESUME_EVERY steps)
#SBATCH --constraint="l40|l40s|a40" # Toys needs 48 GB
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=mk.ngohuu.2024@phdcs.smu.edu.sg
#SBATCH --job-name=CollabRecGRPO

# GRPO on a CollabRec SFT checkpoint (R3 stage-2 noise GRPO; run_collabrec_grpo.py).
#   sbatch --export=ALL submit_collabrec_grpo.sbatch                              (defaults: Toys, 345291 init)
#   sbatch --export=ALL,PROBE=0.01,0.05,0.2,0.5,1.5 submit_collabrec_grpo.sbatch  (reward stats per noise std)
#   sbatch --export=ALL,NOISE_STD=0.5,NOISE_REL=1 submit_collabrec_grpo.sbatch    (noise relative to ||thought||)
#   sbatch --export=ALL,LR_LLM=1e-6 submit_collabrec_grpo.sbatch                  (also train the LLM; 5e-5 diverged)
# NOTE: values with commas (PROBE) split under --export=...; set them in the environment and use --export=ALL.
CK=/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints
export DATASET=${DATASET:-Toys_and_Games}
export INIT_RUN=${INIT_RUN:-$CK/Toys_and_Games/CollabRec/text-motive_k10_beta0.1_rs1_lr0.0003_job345291}
export SUPPORT_DIR=${SUPPORT_DIR-data/$DATASET/collab/motive_corpus/stage2}
export SUPPORT_K=${SUPPORT_K:-10}
export RUN_TAG=${RUN_TAG:-motive}
export MOTIVE_SOURCE=${MOTIVE_SOURCE:-}
export NOISE_STD=${NOISE_STD:-1.5}
export NOISE_REL=${NOISE_REL:-0}
export NUM_GEN=${NUM_GEN:-8}
export BATCH=${BATCH:-256}
export KL=${KL:-0.01}
export LR=${LR:-5e-5}
export LR_LLM=${LR_LLM:-0}   # 0 = LLM frozen, latent module only (R3); >0 also trains the LLM
export FP32_LOGPS=${FP32_LOGPS:-0}   # 1: fp32 log-probs / reward (R3: bf16)
export NUM_ITER=${NUM_ITER:-1}
export EPOCHS=${EPOCHS:-1}
export SAMPLE=${SAMPLE:--1}
export RESUME_EVERY=${RESUME_EVERY:-100}
export VALID_EVERY=${VALID_EVERY:-200}
export PROBE=${PROBE:-}
export TRAIN_OR_TEST=${TRAIN_OR_TEST:-train}
export EXPERIMENT=${EXPERIMENT:-CollabRec_GRPO}
export RUN_SUFFIX=${RUN_SUFFIX:-job${SLURM_JOB_ID:-local}}
export CODE_DIR=${CODE_DIR:-/common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/collab-refined-infonce}

module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
cd "$CODE_DIR" || exit 1

init=$(basename "$INIT_RUN")
if [ -n "$PROBE" ]; then
    run=${RUN_NAME:-probe-init_${init}$( [ "$NOISE_REL" = 1 ] && echo _rel )_$RUN_SUFFIX}
else
    run=${RUN_NAME:-grpo${RUN_TAG:+-$RUN_TAG}-init_${init}_noise${NOISE_STD}$( [ "$NOISE_REL" = 1 ] && echo rel )_g${NUM_GEN}_kl${KL}_lr${LR}_llm${LR_LLM}$( [ "$FP32_LOGPS" = 1 ] && echo _fp32 )$( [ "$SAMPLE" != -1 ] && echo _n$SAMPLE )_$RUN_SUFFIX}
fi
RUN_DIR=$CK/$DATASET/$EXPERIMENT/$run
mkdir -p "$RUN_DIR"
log=$RUN_DIR/train.log
[ -n "$PROBE" ] && log=$RUN_DIR/probe.log
[ "$TRAIN_OR_TEST" = test ] && log=$RUN_DIR/test_job${SLURM_JOB_ID:-local}.log
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "DATASET=$DATASET INIT_RUN=$INIT_RUN SUPPORT_DIR=$SUPPORT_DIR SUPPORT_K=$SUPPORT_K RUN_TAG=$RUN_TAG MOTIVE_SOURCE=$MOTIVE_SOURCE NOISE_STD=$NOISE_STD NOISE_REL=$NOISE_REL NUM_GEN=$NUM_GEN BATCH=$BATCH KL=$KL LR=$LR LR_LLM=$LR_LLM FP32_LOGPS=$FP32_LOGPS NUM_ITER=$NUM_ITER EPOCHS=$EPOCHS SAMPLE=$SAMPLE PROBE=$PROBE TRAIN_OR_TEST=$TRAIN_OR_TEST"
echo "code: $CODE_DIR"
echo "run folder: $RUN_DIR"
echo "log: $log"
python -u run_collabrec_grpo.py \
    --device cuda:0 \
    --dataset "$DATASET" \
    --init_run_dir "$INIT_RUN" \
    --support_dir "$SUPPORT_DIR" \
    --support_k $SUPPORT_K \
    --motive_source "$MOTIVE_SOURCE" \
    --run_tag "$RUN_TAG" \
    --noise_std $NOISE_STD \
    $( [ "$NOISE_REL" = 1 ] && echo --noise_relative ) \
    $( [ "$FP32_LOGPS" = 1 ] && echo --fp32_logps ) \
    --num_generations $NUM_GEN \
    --batch_size $BATCH \
    --kl_beta $KL \
    --lr $LR \
    --lr_llm $LR_LLM \
    --num_iterations $NUM_ITER \
    --nums_epoch $EPOCHS \
    --sample $SAMPLE \
    --resume_every $RESUME_EVERY \
    --valid_every $VALID_EVERY \
    ${PROBE:+--probe_noise $PROBE} \
    --train_or_test $TRAIN_OR_TEST \
    --eval_batch_size 8 \
    --ckpt_root $CK \
    --experiment $EXPERIMENT \
    --run_name "$run" >> "$log" 2>&1   # append: a requeued run keeps its earlier log
status=$?
grep -E "probe|Recall|NDCG" "$log"
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out "$RUN_DIR/slurm-$SLURM_JOB_ID.out" 2>/dev/null
exit $status
