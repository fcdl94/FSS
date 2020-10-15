from .segmentation_module import make_model
from .finetuning import FineTuning, FineTuningClassifier, AMP
from .SPNet import SPNet, SPNet_LwF, SPNet_MiB
from .imprinting import CosineFT, CosineFTC, WeightImprinting, DynamicWI, WeightRegression, WeightMixing
from .ilmethods import MiB, LwF, MiB_WI

methods = {"FT": FineTuning, "CC": CosineFTC, "SPN": SPNet, "COS": CosineFT, "WI": WeightImprinting,
           "AMP": AMP, "DWI": DynamicWI, "WR": WeightRegression, "WM": WeightMixing}


def get_method(opts, task, device, logger):
    if opts.method not in methods:
        raise NotImplementedError
    return methods[opts.method](task=task, device=device, logger=logger, opts=opts)
