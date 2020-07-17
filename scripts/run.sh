#!/bin/bash

## be careful, change O1!!!!!
alias exp='python -m torch.distributed.launch --nproc_per_node=2 run.py --batch_size 12 --opt_level O1'
shopt -s expand_aliases

exp --dataset voc --name FT_exemplars --method FT --task 15-5 --step 1 --lr 0.001 --random_seed 94 --epochs 1 --exemplars 150