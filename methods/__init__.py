from .segmentation_module import make_model
from .trainer import Trainer
from .imprinting import *
from .generative_AFHN import AFHN
from .generative import FGI

methods = {"FT", "SPN", "COS", "WI", 'DWI', "MWI", "AMP", "WG",
           "GIFS", "LWF", "MIB", "ILT", "RT", "AFHN", "FGI", "FGI2"}


def get_method(opts, task, device, logger):
    if opts.method == 'WI':
        opts.method = 'COS'
        return WeightImprinting(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'WG':
        opts.method = 'COS'
        return WeightGenerator(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'DWI':
        opts.method = 'COS'
        return DynamicWI(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'MWI':
        opts.method = 'COS'
        return MaskedWI(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'AMP':
        opts.method = 'FT'
        return AMP(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'AFHN':
        opts.method = 'FT'
        return AFHN(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'FGI':
        opts.method = 'COS'
        opts.sum_noise = True
        return FGI(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'FGI2':
        opts.method = 'COS'
        opts.sum_noise = False
        return FGI(task=task, device=device, logger=logger, opts=opts)
    else:
        return Trainer(task=task, device=device, logger=logger, opts=opts)
