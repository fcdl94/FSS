import utils
import argparser
import os
from utils.logger import Logger

from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import get_dataset
from metrics import StreamSegMetrics
from task import Task

from methods import get_method
import time


def save_ckpt(path, model, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_score": best_score,
    }

    torch.save(state, path)


def get_step_ckpt(opts, logger, task_name):
    # xxx Get step checkpoint
    step_checkpoint = None
    if opts.step_ckpt is not None:
        path = opts.step_ckpt
    else:
        path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"

    # generate model from path
    if os.path.exists(path):
        step_checkpoint = torch.load(path, map_location="cpu")
        step_checkpoint['path'] = path
    elif opts.debug:
        logger.info(
            f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
    else:
        raise FileNotFoundError(f"Step checkpoint not found in {path}")

    return step_checkpoint


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    task = Task(opts)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    if task.nshot != -1:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}-s{task.nshot}/"
    else:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    checkpoint_path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step}.pth"
    os.makedirs(f"checkpoints/step/{task_name}", exist_ok=True)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, test_dst = get_dataset(opts, task)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {opts.num_classes}")

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)

    opts.max_iter = opts.epochs * len(train_loader)
    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    logger.info(f"Train loader contains {len(train_loader)} iterations per epoch")
    logger.info(f"Total iterations are {opts.max_iter}, corresponding to {opts.epochs} epochs")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    model = get_method(opts, task, device, logger)
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")
    # IF step > 0 you need to reload pretrained
    if task.step > 0:
        step_ckpt = get_step_ckpt(opts, logger, task_name)
        model.load_state_dict(step_ckpt['model_state'], strict=False)  # False because of incr. classifiers
        logger.info(f"[!] Previous model loaded from {step_ckpt['path']}")
        # clean memory
        del step_ckpt

    logger.debug(model)

    # xxx Handle checkpoint to resume training
    best_score = 0.0
    cur_epoch = 0
    if opts.continue_ckpt:
        opts.ckpt = checkpoint_path
    if opts.ckpt is not None:
        assert os.path.isfile(opts.ckpt), "Error, ckpt not found. Check the correct directory"
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        model.load_state_dict(checkpoint["model_state"])
        logger.info("[!] Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    val_metrics = StreamSegMetrics(len(task.get_order()))
    val_score = None
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    while cur_epoch < opts.epochs and not opts.test:
        # =====  Train  =====
        train_loader.sampler.set_epoch(cur_epoch)  # setup dataloader sampler
        start = time.time()
        epoch_loss = model.train(cur_epoch=cur_epoch, train_loader=train_loader, print_int=opts.print_interval)
        end = time.time()

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]} "
                    f"-- time: {int(end-start)//60}:{int(end-start)%60}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            val_loss, val_score, ret_samples = model.validate(loader=val_loader, metrics=val_metrics,
                                                              ret_samples_ids=sample_ids)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0] + val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]} ")

            logger.info(val_metrics.to_str(val_score))

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0] + val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            for k, (img, target, pred) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                pred = label2color(pred).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, pred), axis=2)  # concat along width
                logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

        # =====  Save Model  =====
        if rank == 0:  # save best model at the last iteration
            score = val_score['Mean IoU'] if val_score is not None else 0.  # use last score we have
            # best model to build incremental steps
            if not opts.debug:
                save_ckpt(checkpoint_path, model, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

        cur_epoch += 1

    torch.distributed.barrier()

    # xxx Test code!
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    # load best model
    if opts.test:
        # Put the model on GPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {checkpoint_path}")
        del checkpoint

    val_loss, val_score, _ = model.validate(loader=test_loader, metrics=val_metrics)
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix_Recall", val_score['Confusion Matrix'])
    logger.add_figure("Test_Confusion_Matrix_Precision", val_score["Confusion Matrix Pred"])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    results["T-Prec"] = val_score['Class Prec']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'])
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'])
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'])

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    main(opts)
