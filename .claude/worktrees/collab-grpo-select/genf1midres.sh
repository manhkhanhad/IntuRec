dataset=Books # CDs_and_Vinyl, Musical_Instruments, Toys_and_Games, Video_Games, Books
dataset_type=train
beam_num=16 # If you want to use N intu items in the second stage, you should set this parameter to N+1 for train mode, and N for both valid and test mode.
# nohup python -u genf1midres.py --device cuda:0 --dataset $dataset --dataset_type $dataset_type --beam_num $beam_num --batch_size 8 > log/$dataset/genf1midres-$dataset_type-beam_$beam_num.log 2>&1 &