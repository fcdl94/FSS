#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
alias exp="python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 run.py --opt_level O1 --num_workers 8"
shopt -s expand_aliases

gen_par="--task voc --dataset coco --lr 0.01 --epochs 20 --batch_size 24 --crop_size 512 --val_interval 1"
#gen_par="--task 15-5 --dataset voc --lr 0.01 --epochs 30 --batch_size 24 --crop_size 512 --val_interval 1"

#path=checkpoints/step/15-5-voc
#ckpt=${path}/PR_0.pth
if [ $1 -eq 0 ]; then
 exp --method SPN --name SPN ${gen_par} --step 0
else
 exp --method COS --name COS ${gen_par} --step 0
fi