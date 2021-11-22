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


def save_ckpt(path, model, epoch):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }

    torch.save(state, path)


def get_step_ckpt(opts, logger, task_name, name):
    # xxx Get step checkpoint
    step_checkpoint = None
    if opts.step_ckpt is not None:
        path = opts.step_ckpt
    else:
        if opts.step - 1 == 0:
            path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"
        else:
            path = f"checkpoints/step/{task_name}/{name}_{opts.step - 1}.pth"

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


# =====  Log metrics on Tensorboard =====
def log_val(logger, val_metrics, val_score, val_loss, cur_epoch):
    logger.info(val_metrics.to_str(val_score))

    # visualize validation score and samples
    logger.add_scalar("V-Loss", val_loss, cur_epoch)
    logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
    logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
    logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
    logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
    # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)


def log_samples(logger, ret_samples, denorm, label2color, cur_epoch):
    for k, (img, target, pred) in enumerate(ret_samples):
        img = (denorm(img) * 255).astype(np.uint8)
        target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
        pred = label2color(pred).transpose(2, 0, 1).astype(np.uint8)

        concat_img = np.concatenate((img, target, pred), axis=2)  # concat along width
        logger.add_image(f'Sample_{k}', concat_img, cur_epoch)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    if opts.device is not None:
        device_id = opts.device
    else:
        device_id = opts.local_rank
    device = torch.device(device_id)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    if opts.device is not None:
        torch.cuda.set_device(opts.device)
    else:
        torch.cuda.set_device(device_id)
    opts.device_id = device_id

    task = Task(opts)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    if task.nshot != -1:
        logdir_full = f"{opts.logdir}/{task_name}/{name}/"
    else:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    checkpoint_path = f"checkpoints/step/{task_name}/{name}_{opts.step}.pth"
    os.makedirs(f"checkpoints/step/{task_name}", exist_ok=True)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, train_dst_no_aug = get_dataset(opts, task, train=True)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    train_loader = data.DataLoader(train_dst, batch_size=min(opts.batch_size, len(train_dst)),
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=min(opts.batch_size, len(val_dst)),
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)

    train_iterations = 1 if task.step == 0 else 20 // task.nshot
    if opts.iter is not None:
        opts.epochs = opts.iter // (len(train_loader) * train_iterations)
    opts.max_iter = opts.epochs * len(train_loader) * train_iterations
    if opts.max_iter == 0:
        opts.max_iter = 1
    logger.info(f"Total batch size is {min(opts.batch_size, len(train_dst)) * world_size}")
    logger.info(f"Train loader contains {len(train_loader)} iterations per epoch, multiplied by {train_iterations}")
    logger.info(f"Total iterations are {opts.max_iter}, corresponding to {opts.epochs} epochs")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    model = get_method(opts, task, device, logger)
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")
    # IF step > 0 you need to reload pretrained
    if task.step > 0:
        step_ckpt = get_step_ckpt(opts, logger, task_name, name)
        assert step_ckpt is not None, "Step checkpoint is None!"
        model.load_state_dict(step_ckpt['model_state'], strict=False)  # False because of incr. classifiers
        logger.info(f"[!] Previous model loaded from {step_ckpt['path']}")
        # clean memory
        del step_ckpt

    # xxx Model warm up
    logger.debug(model)
    if task.step > 0 and not opts.continue_ckpt and opts.ckpt is None:
        logger.info("Warm up lap!")
        model.warm_up(train_dst_no_aug)

    # put the model on DDP
    model.distribute()

    # xxx Handle checkpoint to resume training
    cur_epoch = 0
    if opts.continue_ckpt:
        opts.ckpt = checkpoint_path
    if opts.ckpt is not None:
        assert os.path.isfile(opts.ckpt), "Error, ckpt not found. Check the correct directory"
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        cur_epoch = checkpoint["epoch"] + 1 if not opts.born_again else 0
        model.load_state_dict(checkpoint["model_state"])
        logger.info("[!] Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        logger.info("[!] Train from the beginning of the task")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    train_metrics = StreamSegMetrics(len(task.get_order()), task.get_n_classes()[0])
    val_metrics = StreamSegMetrics(len(task.get_order()), task.get_n_classes()[0])
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))

    # train/val here
    while cur_epoch < opts.epochs and not opts.test:
        # =====  Train  =====
        start = time.time()
        epoch_loss = model.train(cur_epoch=cur_epoch, train_loader=train_loader,
                                 metrics=train_metrics, print_int=opts.print_interval,
                                 n_iter=train_iterations)
        train_score = train_metrics.get_results()
        end = time.time()

        len_ep = int(end - start)
        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]:.4f}, "
                    f"Class Loss={epoch_loss[0]:.4f}, Reg Loss={epoch_loss[1]}\n"
                    f"Train_Acc={train_score['Overall Acc']:.4f}, Train_Iou={train_score['Mean IoU']:.4f} "
                    f"\n -- time: {len_ep // 60}:{len_ep % 60} -- ")
        logger.info(f"I will finish in {len_ep * (opts.epochs - cur_epoch) // 60} minutes")
        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0 and False:
            logger.info("validate on val set...")
            val_loss, _ = model.validate(loader=val_loader, metrics=val_metrics, ret_samples_ids=None)
            val_score = val_metrics.get_results()

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss}")
            log_val(logger, val_metrics, val_score, val_loss, cur_epoch)

        # =====  Save Model  =====
        if rank == 0 and (cur_epoch + 1) % opts.ckpt_interval == 0:  # save best model at the last iteration
            # best model to build incremental steps
            save_ckpt(checkpoint_path, model, cur_epoch)
            logger.info("[!] Checkpoint saved.")

        cur_epoch += 1

    if not opts.test:
        # =====  Finalize Model  =====
        logger.info("Cooling down...")
        if rank == 0:
            model.cool_down(train_dst)

    # =====  Save Model  =====
    if not opts.test and rank == 0:  # save best model at the last iteration
        save_ckpt(checkpoint_path, model, cur_epoch)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx Test code!
    logger.info("*** Test the model on all seen classes...")
    test_dst_all, test_dst_novel = get_dataset(opts, task, train=False)
    logger.info(f"Dataset: {opts.dataset}, Test set: {len(test_dst_all)}")
    # make data loader for all classes
    test_loader_all = data.DataLoader(test_dst_all, batch_size=opts.test_batch_size,
                                      sampler=DistributedSampler(test_dst_all, num_replicas=world_size, rank=rank),
                                      num_workers=opts.num_workers)

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(test_loader_all), opts.sample_num, replace=False)  # sample idxs for visual.
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
    if opts.test and opts.ckpt is not None:
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    logger.info(f"*** Model restored from {checkpoint_path}")
    del checkpoint

    val_loss, ret_samples = model.validate(loader=test_loader_all, metrics=val_metrics,
                                           ret_samples_ids=sample_ids)
    val_score = val_metrics.get_results()
    conf_matrixes = val_metrics.get_conf_matrixes()
    logger.print("Done test on all")
    logger.info(f"*** End of Test on all, Total Loss={val_loss}")

    logger.info(val_metrics.to_str(val_score))

    log_samples(logger, ret_samples, denorm, label2color, 0)

    logger.add_figure("Test_Confusion_Matrix_Recall", conf_matrixes['Confusion Matrix'])
    logger.add_figure("Test_Confusion_Matrix_Precision", conf_matrixes["Confusion Matrix Pred"])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    results["T-Prec"] = val_score['Class Prec']
    logger.add_results(results)
    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'])
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'])
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'])
    ret = val_score['Mean IoU']

    logger.log_results(task=task, name=opts.name, results=val_score['Class IoU'].values())
    logger.log_aggregates(task=task, name=opts.name, results=val_score['Agg'])

    logger.close()
    return ret


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    main(opts)
