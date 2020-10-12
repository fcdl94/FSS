from .segmentation_module import make_model
from .finetuning import FineTuning, FineTuningClassifier, AMP
from .SPNet import SPNet, SPNet_LwF, SPNet_MiB
from .imprinting import CosineFT, WeightImprinting, DynamicWI
from .ilmethods import MiB, LwF, MiB_WI


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
        method_ = AMP
    elif opts.method == "DWI":
        method_ = DynamicWI
    else:
        raise NotImplementedError

    return method_(task=task, device=device, logger=logger, opts=opts)

