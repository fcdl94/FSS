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

lr=0.001
iter=10000
#FT
oname=FT
for ns in 1 5 10; do
  exp --method FT --name FT_lr${lr}_iter${iter} --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done

##AMP
#for ns in 1 5 10; do
#  exp --method AMP --name AMP --iter 0 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
#AMP alpha 0.25
#for ns in 1 5 10; do
#  exp --method AMP --name AMP_alpha025 --iter 0 --amp_alpha 0.25 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
##SPN
#oname=SPN
#for ns in 1 5 10; do
#  exp --method SPN --name SPN --iter ${iter} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
#COS
#oname=COS
#for ns in 1 5 10; do
#  exp --method COS --name COS --iter ${iter} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
#WI
#for ns in 1 5 10; do
#  exp --method WI --name WI --iter 0 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
#WI-FT
#for ns in 1 5 10; do
#  exp --method WI --name WI-FT --iter ${iter} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done

#met=MIB-WI
#oname=COS_ns
#name=${met}ft_ns
#gen_par="--task ${task} --no_mask --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --crop_size 512 --crop_size_test 512"
#inc_par="--ishot ${ishot} --input_mix novel --iter ${iter} --val_interval 50 --ckpt_interval 10"
#for ns in 1 5 10; do
#  exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${ckpt}
##  for s in 2 3 4 5; do
##   exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step $s --nshot ${ns}
##  done
#done