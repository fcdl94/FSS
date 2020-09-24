from .segmentation_module import make_model
from .finetuning import FineTuning, FineTuningClassifier, AMP
from .SPNet import SPNet, SPNet_LwF, SPNet_MiB
from .imprinting import CosineFT, WeightImprinting, CosineFTC
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
    elif opts.method == "CFTC":
        method_ = CosineFTC
    elif opts.method == "WI":
        method_ = WeightImprinting
    elif opts.method == "AMP":
        method_ = AMP
    elif opts.method == "LWF":
        method_ = LwF
    elif opts.method == "MIB":
        method_ = MiB
    elif opts.method == "MIB-SPN":
        method_ = SPNet_MiB
    elif opts.method == "MIB-WI":
        method_ = MiB_WI
    else:
        raise NotImplementedError

    return method_(task=task, device=device, logger=logger, opts=opts)

