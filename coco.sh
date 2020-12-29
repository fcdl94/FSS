#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
alias exp="python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 run.py --num_workers 8"
shopt -s expand_aliases

gen_par="--dataset coco --epochs 20 --batch_size 24 --crop_size 512 --val_interval 1"
#gen_par="--task ${task} --dataset voc --epochs 30 --batch_size 24 --crop_size 512 --val_interval 1"
#gen_par="--task voc --dataset coco-stuff --lr 0.01 --epochs 20 --batch_size 24 --crop_size 512 --val_interval 1"

lr=0.01

exp --method SPN --name SPN ${gen_par} --step 0 --lr ${lr} --task 20-0
exp --method SPN --name SPN ${gen_par} --step 0 --lr ${lr} --task 20-1
exp --method SPN --name SPN ${gen_par} --step 0 --lr ${lr} --task 20-2


#export CUDA_VISIBLE_DEVICES=$1
#port=$2
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
#shopt -s expand_aliases
#
#task=7mc
#ds=coco
#
#path=checkpoints/step/spn-${ds}
#
#gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
#
#lr=0.0001 # for 1-shot, 15-1
#iter=200
#
#for is in 0 1 2 3 4; do
#inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5 --no_pooling"
#for ns in 1; do
#  exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
#  exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
#    for step in 2 3 4 5 6 7; do
#        exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
#        exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
#    done
#  done
#done
