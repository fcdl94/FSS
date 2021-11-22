#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
shopt -s expand_aliases

ds=coco
task=$2

exp --method FT --name FT --epochs 20 --lr 0.01 --batch_size 24
exp --method COS --name COS --epochs 20 --lr 0.01 --batch_size 24
exp --method SPN --name SPN --epochs 20 --lr 0.01 --batch_size 24
exp --method DWI --name DWI --epochs 20 --lr 0.01 --batch_size 24 --ckpt checkpoints/step/${task}-voc/COS_0.pth
exp --method RT --name RT --epochs 40 --lr 0.01 --batch_size 24 --ckpt checkpoints/step/${task}-voc/FT_0.pth --born_again

gen_par="--task ${task} --dataset ${ds} --batch_size 10"
lr=0.001
iter=2000
path=checkpoints/step/${task}-${ds}
for ns in 1 2 5; do  # shot 1/2/5 images
  for is in 0 1 2; do  # image samples
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"

      exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

      exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
      exp --method DWI --name DWI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/DWI_0.pth
      exp --method RT --name RT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/RT_0.pth

      exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth
      exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

      exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
      exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
      exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

      exp --method PIFS --name PIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
  done
done
