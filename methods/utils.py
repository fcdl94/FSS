import utils
import torch


def get_scheduler(opts, optim):
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optim, max_iters=opts.max_iter, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opts.lr_decay_step,
                                                    gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    return scheduler
