#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0 --num_workers 2"
shopt -s expand_aliases

ishot=$3
task=15-5
ds=voc

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds}"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 200 --ckpt_interval 5"

#oname=COS_ns
#lr=0.01
#exp --method WM --name WM --epochs 30 --lr ${lr} ${gen_par} --step 0 --ckpt ${path}/${oname}_0.pth

oname=COS_ns
lr=0.001
iter=1000
#for ns in 1 5 10; do
#  exp --method LC --name LC-iter${iter} --iter ${iter}  --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
for ns in 1 5 10; do
  exp --method COS --name COS-hnm --hnm --iter ${iter}  --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done

#oname=SPN_ns
#for ns in 1 5 10; do
#  exp --method LS --name LS-iter${iter} --iter ${iter}  --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done




