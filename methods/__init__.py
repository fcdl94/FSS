from .segmentation_module import make_model
from .finetuning import FineTuning, FineTuningClassifier, AMP
from .SPNet import SPNet, SPNet_LwF, SPNet_MiB
from .imprinting import CosineFT, CosineFTC, WeightImprinting, WIC, DynamicWI, WeightRegression, WeightMixing
from .ilmethods import MiB, LwF, LwF_WI, LwF_Cosine

methods = {"FT": FineTuning, "SPN": SPNet, "COS": CosineFT, "CC": CosineFTC, "WI": WeightImprinting, "WIC": WIC,
           "AMP": AMP, "DWI": DynamicWI, "WR": WeightRegression, "WM": WeightMixing,
           "LWF": LwF, "LC": LwF_Cosine, "LW": LwF_WI, "LS": SPNet_LwF}


def get_method(opts, task, device, logger):
    if opts.method not in methods:
        raise NotImplementedError
    return methods[opts.method](task=task, device=device, logger=logger, opts=opts)
