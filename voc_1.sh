#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

task=15-1
ds=voc

path=checkpoints/step/15-5-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10"

#lr=0.001 # for 1,2,5-shot, 15-5
#lr=0.0001 # for 2,5-shot, 15-1

lr=0.00001 # for 1-shot, 15-1
iter=200

for is in 1 2 3 4; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
for ns in 1; do
  exp --method WI --name WI_np --no_pooling --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_nopool_0.pth
  exp --method GL --name GL_np --no_pooling --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_nopool_0.pth
    for step in 2 3 4 5; do
        exp --method WI --name WI_np --no_pooling --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
        exp --method GL --name GL_np --no_pooling --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
    done
  done
done


#ishot=$3
#task=15-5
#ds=voc
#
#path=checkpoints/step/${task}-${ds}
#
#gen_par="--task ${task} --dataset ${ds} --batch_size 10"
#inc_par="--ishot ${ishot} --input_mix novel --val_interval 1000 --ckpt_interval 5"
#
##oname=COS_ns
##lr=0.01
##exp --method WM --name WM --epochs 30 --lr ${lr} ${gen_par} --step 0 --ckpt ${path}/${oname}_0.pth
#
#lr=0.001
#iter=1000
#for is in 0 1 2 3 4; do
#  inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
#  ns=$3
#  exp --method WI --name WI-FT-iabr-l1-0.1 --l1_loss 0.1 --norm_act iabr --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#done
