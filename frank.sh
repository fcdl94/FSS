#!/bin/bash
#PBS -l select=1:ncpus=15:mem=15GB:ngpus=2
#PBS -l walltime=48:00:00
#PBS -e err_file.txt
#PBS -o out_file.txt
#PBS -N GIFS_COCO
#PBS -q gpu

# setup env
module load anaconda/3.2020.2
source activate /home/fcermelli/.conda/envs/deep_env/
cd /work/fcermelli/fcdl/FSS/

port=$1
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --opt_level O0"
shopt -s expand_aliases

task=$2  # 20-0, 20-1, 20-2, 20-3
ds=coco

path=checkpoints/step/${task}-${ds}
task=${task}m
gen_par="--task ${task} --dataset ${ds} --batch_size 10 --crop_size 512"

lr=0.001 # for 1-shot, 15-1
iter=1000

for is in 0 1 2; do
inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"
  for ns in 1 5; do
    exp --method WI  --name WI  --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
   exp --method GIFS --name GIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
    exp --method WI  --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth

    exp --method AMP --name AMP --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method FT  --name FT  --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

    exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

    exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth

      for s in 2 3 4; do
      exp --method WI  --name WI  --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
     exp --method GIFS --name GIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method WI  --name WIT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method AMP --name AMP --iter 0       --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method FT  --name FT  --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      done
  done
done