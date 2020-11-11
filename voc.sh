#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

task=15-5
ds=voc

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10"

lr=0.001
iter=1000
is=$3

for is in 0 1 2 3 4; do
  inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
  for ns in 1 2 5; do
    exp --method WI --name WI-FT-iabr-kd10 --norm_act iabr --loss_kd 10 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
  done
done
