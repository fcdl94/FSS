#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
shopt -s expand_aliases

ds=voc
task=$2  # 5-0 5-1 5-2 5-3

exp --method FT --name FT --epochs 30 --lr 0.01 --batch_size 24
exp --method COS --name COS --epochs 30 --lr 0.01 --batch_size 24
exp --method SPN --name SPN --epochs 30 --lr 0.01 --batch_size 24
exp --method DWI --name DWI --epochs 30 --lr 0.01 --batch_size 24 --ckpt checkpoints/step/${task}-voc/COS_0.pth
exp --method RT --name RT --epochs 60 --lr 0.01 --batch_size 24 --ckpt checkpoints/step/${task}-voc/FT_0.pth --born_again

path=checkpoints/step/${task}-${ds}
task="${task}m"
gen_par="--task ${task} --dataset ${ds} --batch_size 10"
lr=0.0001 # for 1,2,5-shot
iter=200

# for 1 shot, disable the batch norm pooling on the deeplab head (or it'll throw errors) (--no_pooling)
for is in 0 1 2; do
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5 --no_pooling"
    ns=1
    exp --method WI --name WI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method GIFS --name GIFS_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method GIFS --name GIFS_with_UCE_mask --mib_ce --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method MIB --name MIB_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking

    for s in 2 3 4 5; do
      exp --method WI --name WI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method GIFS --name GIFS_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method GIFS --name GIFS_with_UCE_mask --mib_ce --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method MIB --name MIB_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
    done
done

for ns in 2 5; do
  for is in 0 1 2; do
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
    exp --method WI --name WI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method GIFS --name GIFS_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method GIFS --name GIFS_with_UCE_mask --mib_ce --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --masking
    exp --method MIB --name MIB_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth --masking

    for s in 2 3 4 5; do

      exp --method WI --name WI_mask --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method GIFS --name GIFS_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method GIFS --name GIFS_with_UCE_mask --mib_ce --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
      exp --method MIB --name MIB_mask --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns} --masking
    done
  done
done
