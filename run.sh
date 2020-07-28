#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
alias exp='python -m torch.distributed.launch --nproc_per_node=1 run.py --opt_level O1'
shopt -s expand_aliases

met=MIB
oname=FT_bgm0
name=MIB_mib
ishot=0
lr=1e-4
gen_par="--masking 0 --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --crop_size 320 --crop_size_test 512"
inc_par="--ishot ${ishot} --input_mix novel --epochs 50 --val_interval 50 --ckpt_interval 10"

#exp --method ${met} --name ${name} ${gen_par} --epochs 21 --step 0 --val_interval 21
ckpt=checkpoints/step/15-5-voc/${oname}_0.pth
#exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot 1  --step_ckpt ${ckpt}
exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot 2  --step_ckpt ${ckpt}
exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot 5  --step_ckpt ${ckpt}
exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot 10 --step_ckpt ${ckpt}
exp --method ${met} --name ${name} ${gen_par} ${inc_par} --step 1 --nshot 20 --step_ckpt ${ckpt}
