#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

task=7mc
ds=coco

path=checkpoints/step/spn-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"

lr=0.0001
iter=200

for is in 0 1 2 3 4; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5 --no_pooling"
for ns in 1; do
  exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth
    for step in 2 3 4 5 6 7; do
        exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
    done
  done
done
