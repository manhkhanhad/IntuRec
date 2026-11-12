export CUDA_VISIBLE_DEVICES=0


datasets=(""CDs_and_Vinyl"")
train_or_test="train"
decode_method="standard"
for dataset in "${datasets[@]}"; do
    port=$((RANDOM % 5000 + 10000))
    torchrun \
        --nproc_per_node=1 \
        --nnodes=1 \
        --master_port=$port \
        run_inturec_f1.py \
            --data_path "/common/home/users/m/mk.ngohuu.2024/research/R3/data" \
            --dataset $dataset \
            --lr 3e-4 \
            --base_model Qwen/Qwen2.5-1.5B \
            --train_batch_size 128 \
            --train_micro_batch_size 12 \
            --eval_batch_size 12 \
            --train_or_test $train_or_test \
            --decode_method $decode_method \
    > log/$dataset/inturec-f1-lr_3e-4-$train_or_test-$decode_method.log 2>&1 &
    pre_pid=$!
    wait $pre_pid
done