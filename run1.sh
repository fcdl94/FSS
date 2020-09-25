#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
alias exp='python -m torch.distributed.launch --master_port 6665 --nproc_per_node=1 run.py --opt_level O0'
shopt -s expand_aliases

met=AMP
oname=FT_ns
name=${met}alpha025_ns
ishot=0
lr=1e-4
epochs=50
task=15-5

gen_par="--task ${task} --no_mask --masking 255 --batch_size 10 --lr ${lr} --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --epochs ${epochs} --val_interval 10 --ckpt_interval 10"

ckpt=checkpoints/step/15-5-voc/${oname}_0.pth

for ns in 1 5 10; do
  exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${ckpt} --amp_alpha 0.25
##  for s in 2 3 4 5; do
##   exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step $s --nshot ${ns}
##  done
done
