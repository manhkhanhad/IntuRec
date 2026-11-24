#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/%x.%j.out
#SBATCH --requeue
#SBATCH --constraint="l40|a100|a40|l40s"
#SBATCH --partition=researchshort
#SBATCH --qos=research-1-qos
#SBATCH --job-name=GEARec

# GEARec v2 target reasoning for any dataset (input of motive v6); same settings as Toys/CDs:
# gemma-3-4b-it, Best-of-8, full history, temperature 0.8, seed 42, mxbai reachability rank (V3). Generate + validate only.
#   sbatch --export=ALL,DATASET=Steam,DOMAIN=steam,SPLIT=train collab/gearec_v2/submit_gearec.sbatch
# Needs data/<DATASET>/collab/gearec_v2/items.json (--stage items, CPU). Writes full_<split>_bo8/. Resumes per stage.
DATASET=${DATASET:?set DATASET}; DOMAIN=${DOMAIN:?set DOMAIN}
SPLIT=${SPLIT:?set SPLIT=train|valid|test}
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec/.claude/worktrees/motiverec || exit 1
export LD_LIBRARY_PATH=$HOME/.conda/envs/racer/lib:$LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-/common/home/users/m/mk.ngohuu.2024/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/.conda/envs/racer/bin/python
D=data/$DATASET/collab/gearec_v2
OUT=$D/full_${SPLIT}_bo8
COMMON="--domain $DOMAIN --data_dir data/$DATASET --items $D/items.json --item_emb $D/item_emb_mxbai.pt"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  split $SPLIT  $DATASET"
set -e
if [ ! -f $D/item_emb_mxbai.pt ]; then   # first job to get here builds it (atomic rename)
    $PY collab/gearec_v2/target_reasoning.py --stage item_emb $COMMON --item_emb $D/item_emb_mxbai.$SLURM_JOB_ID.pt --out $OUT
    mv -n $D/item_emb_mxbai.$SLURM_JOB_ID.pt $D/item_emb_mxbai.pt; rm -f $D/item_emb_mxbai.$SLURM_JOB_ID.pt
fi
if [ ! -f $OUT/candidates.jsonl ]; then
    $PY collab/gearec_v2/target_reasoning.py --stage generate $COMMON --out $OUT --split $SPLIT --n_samples 8 --seed 42
fi   # (no pkill of vLLM engines: other jobs of this user may share the node)
$PY collab/gearec_v2/target_reasoning.py --stage validate $COMMON --out $OUT --split $SPLIT --rank_max 100
$PY - <<EOF
import json, statistics as st
S = [json.loads(l) for l in open("$OUT/validated.jsonl")]
r = [s["best"]["rank"] for s in S if s["best"]]
print(f"$SPLIT: {len(S)} rows, with best {len(r)}, median rank {st.median(r):.0f}, "
      f"top20 {sum(x <= 20 for x in r) / len(r):.3f}, top100 {sum(x <= 100 for x in r) / len(r):.3f}")
EOF
cp -f /common/home/users/m/mk.ngohuu.2024/research/IntuRec/run_logs/$SLURM_JOB_NAME.$SLURM_JOB_ID.out $OUT/slurm-$SLURM_JOB_ID.out 2>/dev/null || true
