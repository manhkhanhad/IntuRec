#!/usr/bin/env bash
# Move a finished run's checkpoint and predictions into the <dataset>/<experiment>/<run_name> layout.
#   collab/migrate_run.sh <old_ckpt_dir> <old_result_dir> <experiment> <run_name>
# Checkpoints go to $CKPT_ROOT/CDs_and_Vinyl/<experiment>/<run_name> (copy, verify sizes, then delete the
# source); a best checkpoint identical to an epoch_N/ copy becomes a symlink. Predictions go to
# result_data/CDs_and_Vinyl/<experiment>/<run_name>/standard.
set -euo pipefail
cd /common/home/users/m/mk.ngohuu.2024/research/IntuRec
CKPT_ROOT=${CKPT_ROOT:-/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints}
src=$(readlink -f "$1"); old_res=$2; exp=$3; run=$4
dst=$CKPT_ROOT/CDs_and_Vinyl/$exp/$run
[ -e "$dst" ] && { echo "destination exists: $dst"; exit 1; }
mkdir -p "$(dirname "$dst")"
cp -a "$src" "$dst"
diff <(cd "$src" && find . -type f -printf "%p %s\n" | sort) <(cd "$dst" && find . -type f -printf "%p %s\n" | sort)
rm -rf "$src"
for e in 1 2 3; do
    if [ -f "$dst/model.safetensors" ] && [ ! -L "$dst/model.safetensors" ] && cmp -s "$dst/model.safetensors" "$dst/epoch_$e/model.safetensors"; then
        rm "$dst/model.safetensors"; ln -s "epoch_$e/model.safetensors" "$dst/model.safetensors"; break
    fi
done
if [ -d "$old_res/standard" ]; then
    mkdir -p "result_data/CDs_and_Vinyl/$exp/$run"
    mv "$old_res/standard" "result_data/CDs_and_Vinyl/$exp/$run/standard"
fi
echo "migrated $exp/$run"
