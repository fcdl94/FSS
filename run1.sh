#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py"
shopt -s expand_aliases

ishot=$3
task=15-5
ds=voc

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 20 --ckpt_interval 5"

lr=0.001
iter=1000
for is in 0 1 2 3 4; do
  inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
  ns=$3
  exp --method WI --name WI-FT-iabr-kd-10 --loss_kd 10 --norm_act iabr --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
done



#  exp --method FT  --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_ns_0.pth
#  exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_ns_0.pth
#  exp --method COS --name COS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
##  exp --method COS --name COS_img --supp_dataset imagenet --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI --name WI-FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_img --supp_dataset imagenet --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_COCO --supp_dataset coco --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_mom0 --bn_momentum 0 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth

