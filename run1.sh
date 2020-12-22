#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py"
shopt -s expand_aliases

#task=spn
#ds=coco
#
#path=checkpoints/step/${task}-${ds}
#
#gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
#lr=0.001
#iter=1000
#
#
#for is in 0 1 2 3 4; do
#inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5"
#    for ns in 1 2 5; do
#      exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth
#done
#done


task=voc
ds=coco

path=checkpoints/step/${task}-${ds}

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"

lr=0.001
iter=2000 # fixme 1000 for VOC, COCO73

for ns in 1 2 5; do
  is=$3
  inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
#  exp --method WI --name WI --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
  exp --method WI --name WIT-iabn_l2_0.0001 --iter ${iter} --l2_loss 0.0001 --norm_act iabn --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
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

