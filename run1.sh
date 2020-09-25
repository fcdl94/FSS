#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

lr=1e-4
task=15-5
path=checkpoints/step/15-5-voc

gen_par="--task ${task} --batch_size 10 --lr ${lr} --crop_size 512"
inc_par="--input_mix novel --val_interval 10 --ckpt_interval 10"

oname=SPN_ns
for is in 0 1 2 3 4; do
  for ns in 1 5 10; do
    exp --method SPN --name SPN_notrain_ns --epochs 0 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --ishot ${is} --step_ckpt ${path}/${oname}_0.pth
  done
done