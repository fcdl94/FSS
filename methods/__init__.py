from .segmentation_module import make_model
from .trainer import Trainer
from .imprinting import *

methods = {"FT", "SPN", "COS", "WI", 'DWI', 'WM', "AMP", "WG", "SWG", "GIFS", "LWF", "MIB", "ILT", "RT"}


def get_method(opts, task, device, logger):
    if opts.method == 'WI':
        opts.method = 'COS'
        return WeightImprinting(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'WG':
        opts.method = 'COS'
        return ContextWiseWeightImprinting(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'SWG':
        opts.method = 'COS'
        return SpatialWeightGenerator(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'DWI':
        opts.method = 'COS'
        return DynamicWI(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'WM':
        opts.method = 'COS'
        return WeightMixing(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'AMP':
        opts.method = 'FT'
        return AMP(task=task, device=device, logger=logger, opts=opts)
    else:
        return Trainer(task=task, device=device, logger=logger, opts=opts)
