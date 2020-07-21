#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
alias exp='python -m torch.distributed.launch --nproc_per_node=1 run.py --opt_level O1'
shopt -s expand_aliases

met=FT
name=FT_usebg
lr=1e-4
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 21 --step 0 --val_interval 20 --crop_size 320 --crop_size_test 512
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 1000 --step 1 --input_mix both --nshot 1 --val_interval 100 --crop_size 320 --crop_size_test 512
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 500 --step 1 --input_mix both --nshot 2 --val_interval 50 --crop_size 320 --crop_size_test 512
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 200 --step 1 --input_mix both --nshot 5 --val_interval 20 --crop_size 320 --crop_size_test 512
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 100 --step 1 --input_mix both --nshot 10 --val_interval 10 --crop_size 320 --crop_size_test 512
exp --method ${met} --name ${name} --use_bkg --fix_bn --batch_size 10 --lr ${lr} --weight_decay 5e-4 --epochs 50 --step 1 --input_mix both --nshot 20 --val_interval 5 --crop_size 320 --crop_size_test 512
