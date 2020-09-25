#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

ishot=$3
lr=1e-4
epochs=50
task=15-5
path=checkpoints/step/15-5-voc

gen_par="--task ${task} --no_mask --masking 255 --batch_size 10 --lr ${lr} --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 10 --ckpt_interval 10"

#FT
oname=FT_ns
#for ns in 1 5 10; do
#  exp --method FT --name FT_ns --epochs ${epochs} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#done
#AMP
for ns in 1 5 10; do
  exp --method AMP --name AMP_ns --epochs 0 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done
#AMP alpha 0.25
for ns in 1 5 10; do
  exp --method AMP --name AMP_alpha025_ns --epochs 0 --amp_alpha 0.25 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done
#SPN
oname=SPN
for ns in 1 5 10; do
  exp --method SPN --name SPN_ns --epochs ${epochs} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done
#COS
oname=COS_ns
for ns in 1 5 10; do
  exp --method COS --name COS_ns --epochs ${epochs} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done
#WI
for ns in 1 5 10; do
  exp --method WI --name WI_ns --epochs 0 ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done
#WI-FT
for ns in 1 5 10; do
  exp --method WI --name WI-FT_ns --epochs ${epochs} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done


#met=MIB-WI
#oname=COS_ns
#name=${met}ft_ns
#gen_par="--task ${task} --no_mask --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --crop_size 512 --crop_size_test 512"
#inc_par="--ishot ${ishot} --input_mix novel --epochs ${epochs} --val_interval 50 --ckpt_interval 10"
#for ns in 1 5 10; do
#  exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${ckpt}
##  for s in 2 3 4 5; do
##   exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step $s --nshot ${ns}
##  done
#done