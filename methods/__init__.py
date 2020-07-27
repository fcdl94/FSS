from .segmentation_module import make_model
from .method import FineTuning, FineTuningClassifier
from .SPNet import SPNet
from .imprinting import CosineFT, WeightImprinting, AMP


def get_method(opts, task, device, logger):

    if opts.method == "FT":
        method_ = FineTuning(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "FTC":
        method_ = FineTuningClassifier(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "SPN":
        method_ = SPNet(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "COS":
        method_ = CosineFT(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "WI":
        method_ = WeightImprinting(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == "AMP":
        method_ = AMP(task=task, device=device, logger=logger, opts=opts)
    else:
        raise NotImplementedError

    return method_

