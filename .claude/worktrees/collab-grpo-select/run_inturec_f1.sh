export CUDA_VISIBLE_DEVICES=0,1


datasets=(""Books"")
train_or_test="train"
decode_method="standard"
for dataset in "${datasets[@]}"; do
    port=$((RANDOM % 5000 + 10000))
    torchrun \
        --nproc_per_node=2 \
        --nnodes=1 \
        --master_port=$port \
        run_inturec_f1.py \
            --dataset $dataset \
            --lr 3e-4 \
            --train_batch_size 128 \
            --train_micro_batch_size 24 \
            --eval_batch_size 12 \
            --train_or_test $train_or_test \
            --decode_method $decode_method \
    > log/$dataset/inturec-f1-lr_3e-4-$train_or_test-$decode_method.log 2>&1 &
    pre_pid=$!
    wait $pre_pid
done