#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py"
shopt -s expand_aliases

task=7mc
ds=coco

path=checkpoints/step/spn-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"

lr=0.0001 # for 1-shot, 15-1
iter=200

exp --method FT  --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 0


for is in 0 1 2 3 4; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
for ns in 2 5; do
  exp --method WI --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
    for step in 2 3 4 5 6 7; do
        exp --method WI --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step ${step} --nshot ${ns}
    done
  done
done

#  exp --method FT  --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_ns_0.pth
#  exp --method COS --name COS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method COS --name COS_img --supp_dataset imagenet --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI --name WI-FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_img --supp_dataset imagenet --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_COCO --supp_dataset coco --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth
#  exp --method WI  --name WI-FT_mom0 --bn_momentum 0 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_ns_0.pth

