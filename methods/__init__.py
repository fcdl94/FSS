from .segmentation_module import make_model
from .finetuning import FineTuning, FineTuningClassifier
from .SPNet import SPNet
from .imprinting import CosineFT, WeightImprinting
from .ilmethods import MiB, LwF


def get_method(opts, task, device, logger):

    if opts.method == "FT":
        method_ = FineTuning
    elif opts.method == "FTC":
        method_ = FineTuningClassifier
    elif opts.method == "SPN":
        method_ = SPNet
    elif opts.method == "COS":
        method_ = CosineFT
    elif opts.method == "WI":
        method_ = WeightImprinting
    elif opts.method == "AMP":
        raise NotImplementedError
    elif opts.method == "LWF":
        method_ = LwF
    elif opts.method == "MIB":
        method_ = MiB
    else:
        raise NotImplementedError

    return method_(task=task, device=device, logger=logger, opts=opts)

