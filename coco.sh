#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$2
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

ishot=0
task=voc
ds=coco

path=checkpoints/step/voc-coco

gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"
inc_par="--ishot ${ishot} --input_mix novel --val_interval 1000 --ckpt_interval 5"


#epochs=20
#lr=0.01
#oname=COS
#exp --method WR --name WR --epochs ${epochs} --lr ${lr} ${gen_par} --batch_size 24 --step 0 --ckpt ${path}/${oname}_0.pth

iter=5000
lr=0.001
oname=COS
for ns in 1 5 10; do
  #exp --method WI --name WI-FT_supp --supp_dataset --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#  exp --method WI --name WI-FT-2000 --iter 2000 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#  exp --method WI --name WI-FT_randbn-2000 --norm_act riabn_sync2 --iter 2000 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
#  exp --method WI --name WI+mix-FT_randbn --weight_mix --norm_act riabn_sync2 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
 exp --method COS  --name COS_mom0 --bn_momentum 0 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
 exp --method WI  --name WI_FT_mom0 --bn_momentum 0 --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/${oname}_0.pth
done