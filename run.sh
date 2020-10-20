#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

ishot=$3
task=15-5
ds=voc

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 1000 --ckpt_interval 5"

lr=0.001
iter=20
#FT
for ns in 1; do
  exp --method FT  --name FT_new_kd  --loss_kd 10  --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_ns_0.pth
  exp --method SPN --name SPN_new_kd --loss_kd 10  --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_ns_0.pth
  exp --method COS --name COS_new_kd  --loss_kd 10 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_new --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method AMP --name AMP_025_new --iter 0 --amp_alpha 0.25 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_ns_0.pth
#  exp --method WI  --name WI_new   --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
done
