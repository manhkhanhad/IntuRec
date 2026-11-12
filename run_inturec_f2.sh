decode_method="standard"
train_or_test="train"
dataset="CDs_and_Vinyl" # Books, CDs_and_Vinyl, Musical_Instruments, Toys_and_Games, Video_Games
intuition_num=15
alpha=1.0 # 1.0 0.5 0.3 0.1 0.05 0.03 0.01 0.005 0.003 0.001
reason_step=1 # 1 2 3 4 5
python -u run_inturec_f2.py \
    --device cuda:0 \
    --dataset $dataset \
    --base_model Qwen/Qwen2.5-1.5B \
    --train_or_test $train_or_test \
    --decode_method $decode_method \
    --lr 3e-4 \
    --intuition_num $intuition_num \
    --reason_step $reason_step \
    --alpha $alpha \
    --train_batch_size 128 \
    --train_micro_batch_size 2 \
    --eval_batch_size 2 > log/$dataset/inturec-f2-lr_3e-4-intunum_$intuition_num-alpha_$alpha-$train_or_test-$decode_method-rs_$reason_step.log 2>&1 &