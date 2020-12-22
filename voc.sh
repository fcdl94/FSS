#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

task=$3
ds=voc

path=checkpoints/step/15-5-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10"

lr=0.001 # for 1,2,5-shot, 15-5
iter=1000
#lr=0.0001 # for 2,5-shot, 15-1
#iter=200

for is in 0 1 2; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
for ns in 1 2 5; do

    exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

    exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
    exp --method WI --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
    exp --method GIFS --name GIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth

    exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

  done
done


