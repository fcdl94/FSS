#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py"
shopt -s expand_aliases

task=$3  # 20-0, 20-1, 20-2, 20-3
ds=coco

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"

lr=0.001 # for 1-shot, 15-1
iter=2000

for is in 0; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
for ns in 1; do
#  exp --method WI  --name WI  --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
#exp --method GIFS --name GIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
#  exp --method WI  --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
####
#  exp --method AMP --name AMP --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
#  exp --method FT  --name FT  --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
#  exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
#  exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
  exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

  exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth

  done
done
