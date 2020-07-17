from .segmentation_module import make_model
from .method import FineTuning
from .SPNet import SPNet


def get_method(opts, task, device, logger):

    if opts.method == "FT":
        method_ = FineTuning(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "SPN":
        method_ = SPNet(task=task, device=device, logger=logger, opts=opts)
    else:
        raise NotImplementedError

    return method_

