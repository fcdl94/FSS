#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
shopt -s expand_aliases

ds=voc
task=$2

gen_par="--task ${task} --dataset ${ds} --batch_size 10"
lr=0.001 # for 1,2,5-shot, 15-5
iter=1000
path=checkpoints/step/${task}-${ds}
for ns in 1 2 5; do
  for is in 0 1 2; do
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"

#      exp --method FT --name FT_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking

#      exp --method WI --name WI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
#      exp --method DWI --name DWI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/DWI_0.pth --masking
#      exp --method RT --name RT_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/RT_0.pth  --masking

#      exp --method SPN --name SPN_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth --masking
#      exp --method AMP --name AMP_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth  --masking

#      exp --method LWF --name LWF_masks --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking
#      exp --method ILT --name ILT_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking
#      exp --method MIB --name MIB_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking

#      exp --method GIFS --name GIFS_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
#      exp --method GIFS --name GIFS_with_UCE_mask --mib_ce --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking


  done
done
