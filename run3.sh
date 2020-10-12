#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

ishot=$3
task=15-5
ds=voc

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10  --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 25 --ckpt_interval 5"

lr=0.001
iter=1000
oname=COS_ns
for ns in 1 5 10; do
  exp --method WI --name WI_FT_lr${lr}_iter${iter} --iter ${iter}  --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done

lr=0.0001
iter=5000
oname=COS_ns
for ns in 1 5 10; do
  exp --method WI --name WI_FT_lr${lr}_iter${iter} --iter ${iter}  --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done