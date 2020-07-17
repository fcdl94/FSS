# FSS
## Few Shot Learning in Semantic Segmentation

# How to download data

> cd <target folder>
> ../data/download_voc.sh
  
# How to run the training

> python -m torch.distributed.launch --nproc_per_node="total GPUs" train.py --data_root "folder where you downloaded the data" --name "name of exp" --batch_size=4 --num_workers=1 --other_args

The default folder for the logs is logs/"name of exp". The log is in the format of tensorboard.

The default is to use a pretraining for the backbone used, that is searched in the pretrained folder of the project. If you don't want to use pretrained, please use --no-pretrained.
