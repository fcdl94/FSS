#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
alias exp="python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 run.py --num_workers 8"
shopt -s expand_aliases

gen_par="--task spn --dataset coco --epochs 20 --batch_size 24 --crop_size 512 --val_interval 1"
#gen_par="--task 15-5 --dataset voc --epochs 30 --batch_size 24 --crop_size 512 --val_interval 1"
#gen_par="--task voc --dataset coco-stuff --lr 0.01 --epochs 20 --batch_size 24 --crop_size 512 --val_interval 1"


#path=checkpoints/step/15-5-voc

lr=0.01
if [ $3 -eq 0 ]; then
 exp --method FT --name FT ${gen_par} --step 0 --lr ${lr}
elif [ $3 -eq 1 ]; then
 exp --method COS --name COS ${gen_par} --step 0 --lr ${lr}
elif [ $3 -eq 2 ]; then
 exp --method SPN --name SPN ${gen_par} --step 0 --lr ${lr}
elif [ $3 -eq 3 ]; then
 exp --method COS --name COS${gen_par} --step 0 --lr ${lr}
#elif [ $3 -eq 4 ]; then
# exp --method SPN --name SPN_in ${gen_par} --step 0 --norm_act ain --no_pooling
#elif [ $3 -eq 5 ]; then
# exp --method COS --name COS_in ${gen_par} --step 0 --norm_act ain --no_pooling
fi