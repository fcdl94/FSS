#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
alias exp='python -m torch.distributed.launch --master_port 1994 --nproc_per_node=1 run.py --opt_level O0'
shopt -s expand_aliases

met=COS
oname=COS_disj
name=${met}_disj
ishot=0
lr=1e-4
epochs=50
task=15-5

gen_par="--task ${task} --no_mask --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --crop_size 320 --crop_size_test 512"
inc_par="--ishot ${ishot} --input_mix novel --epochs ${epochs} --val_interval 50 --ckpt_interval 10"

ckpt=checkpoints/step/15-5-voc/${oname}_0.pth
exp --method ${met} --name ${name} ${gen_par} --epochs 21 --step 0 --val_interval 21 # --test --ckpt ${ckpt}

for ns in 1 5 10; do
  exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${ckpt}
#  for s in 2 3 4 5; do
#   exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step $s --nshot ${ns}
#  done
done
