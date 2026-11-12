dataset=CDs_and_Vinyl # CDs_and_Vinyl, Musical_Instruments, Toys_and_Games, Video_Games, Books
dataset_type=test # train, valid, test
beam_num=16 # If you want to use N intu items in the second stage, you should set this parameter to N+1 for train mode, and N for both valid and test mode.

for dataset_type in valid test; do
    if [ $dataset_type == "train" ]; then
        beam_num=16
    else
        beam_num=15
    fi
    python -u genf1midres.py \
        --device cuda:0 \
        --dataset $dataset \
        --dataset_type $dataset_type \
        --beam_num $beam_num \
        --f1_ckpt_path "/common/home/users/m/mk.ngohuu.2024/research/IntuRec/checkpoints/CDs_and_Vinyl/IntuRecF1/lr_0.0003" \
        --batch_size 8 > log/$dataset/genf1midres-$dataset_type-beam_$beam_num.log 2>&1
done