#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --job-name=IntuRecTest

# Test-only decode of a saved IntuRec F2 checkpoint, using the UNMODIFIED main-repo run_inturec_f2.py
# (IntuRec's own submit script hard-codes --train_or_test train, and we never edit IntuRec files).
# The run folder's model.safetensors is IntuRec's best-valid-NTP save (= epoch 2 for CDs 349506).
# Writes only new files: <RUN_DIR>/result/standard/<TAG>/meta.json and <RUN_DIR>/test_<TAG>_job<id>.log.
#   sbatch --export=ALL,RUN_DIR=/.../IntuRecF2_micro2_job349506,TAG=epoch_2 collab/test_inturec_ckpt.sbatch
export DATASET=${DATASET:-CDs_and_Vinyl}
export MICRO=${MICRO:-2}
export ALPHA=${ALPHA:-1.0}
export INTUITION_NUM=${INTUITION_NUM:-15}
export REASON_STEP=${REASON_STEP:-1}
export TAG=${TAG:-epoch_2}
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
[ -f "$RUN_DIR/model.safetensors" ] || { echo "no checkpoint in RUN_DIR=$RUN_DIR"; exit 1; }
module purge
eval "$(conda shell.bash hook)"
module load CUDA/12.4.0
conda activate rrec
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec || exit 1
log=$RUN_DIR/test_${TAG}_job${SLURM_JOB_ID:-local}.log
echo "host $(hostname)  RUN_DIR=$RUN_DIR TAG=$TAG MICRO=$MICRO ALPHA=$ALPHA INTUITION_NUM=$INTUITION_NUM REASON_STEP=$REASON_STEP"
echo "log: $log"
python -u run_inturec_f2.py \
    --device cuda:0 \
    --dataset $DATASET \
    --base_model Qwen/Qwen2.5-1.5B \
    --ckpt_dir $RUN_DIR \
    --result_dir $RUN_DIR/result/standard/$TAG \
    --train_or_test test \
    --decode_method standard \
    --lr 3e-4 \
    --intuition_num $INTUITION_NUM \
    --reason_step $REASON_STEP \
    --alpha $ALPHA \
    --train_batch_size 128 \
    --train_micro_batch_size $MICRO \
    --eval_batch_size $MICRO > $log 2>&1
status=$?
grep -E "Recall|NDCG" $log
exit $status
