#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
alias exp='python -m torch.distributed.launch --master_port 6666 --nproc_per_node=2 run.py --opt_level O0'
shopt -s expand_aliases

met=COS
oname=COS_nset
name=${met}_nset
ishot=0
lr=1e-2
epochs=30
task=15-5

gen_par="--task ${task} --no_mask --batch_size 12 --lr ${lr} --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --epochs ${epochs} --val_interval 50 --ckpt_interval 10"

#ckpt=checkpoints/step/15-5-voc/${oname}_0.pth
#exp --method COS --name COS_ns   ${gen_par} --epochs 30 --step 0 --val_interval 1 # --test --ckpt ${ckpt}
#exp --method COS --name COS_ns_relu --relu ${gen_par} --epochs 30 --step 0 --val_interval 1 # --test --ckpt ${ckpt}
exp --method SPN --name SPN_ns   ${gen_par} --epochs 30 --step 0 --val_interval 1 # --test --ckpt ${ckpt}
#exp --method SPN --name SPN_ns_relu --relu  ${gen_par} --epochs 30 --step 0 --val_interval 1 # --test --ckpt ${ckpt}


#for ns in 1 5 10; do
#  exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${ckpt}
##  for s in 2 3 4 5; do
##   exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step $s --nshot ${ns}
##  done
#done
