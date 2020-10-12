#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

ishot=$3
lr=0.0001
task=voc
ds=coco

path=checkpoints/step/voc-coco

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 25 --ckpt_interval 5"


iter=10000
lr=0.001
oname=SPN
for ns in 1 5 10; do
  exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done